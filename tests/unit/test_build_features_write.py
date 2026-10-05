"""
Unit test for the write path in src/pipelines/build_features.py: the
features are written to a staging table and swapped in by RENAME inside one
transaction. No TRUNCATE, which would hold an exclusive lock on the live
table until commit and block readers. Uses a mock connection, so no
Postgres is needed. The live proof (readers polling during a build, and
rollback on failure) is in the commit message and README notes.
"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.pipelines import build_features as bf


def _executed_sql(cursor):
    return [c.args[0] for c in cursor.execute.call_args_list]


def test_swap_uses_staging_table_and_never_truncates():
    cursor = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    features = pd.DataFrame([{col: 0 for col in bf.FEATURE_COLUMNS}])
    features["customer_id"] = "x"

    with patch.object(bf, "execute_values") as execute_values:
        bf.write_features_atomically(conn, features)

    sql = _executed_sql(cursor)
    assert not any("TRUNCATE" in s.upper() for s in sql)
    assert any(s.startswith(f"DROP TABLE IF EXISTS {bf.STAGING_TABLE}") for s in sql)
    assert execute_values.call_args[0][1].startswith(f"INSERT INTO {bf.STAGING_TABLE} ")
    rename_steps = [s for s in sql if s.startswith("ALTER TABLE") or s.startswith("ALTER INDEX")]
    assert rename_steps == [
        f"ALTER TABLE {bf.FEATURES_TABLE} RENAME TO customer_features_old",
        f"ALTER TABLE {bf.STAGING_TABLE} RENAME TO {bf.FEATURES_TABLE}",
        f"ALTER INDEX {bf.STAGING_TABLE}_pkey RENAME TO {bf.FEATURES_TABLE}_pkey",
    ]
    assert "DROP TABLE customer_features_old" in sql
    conn.commit.assert_called_once()
    conn.rollback.assert_not_called()
