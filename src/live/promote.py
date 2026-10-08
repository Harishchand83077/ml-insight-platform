"""
Stage 3b promotion: copies an exported, gate-passed, compat-checked
candidate into the production model directory and writes VERSION.json.
Refuses outright unless BOTH the gate decision and the serving-compat
check passed - promote() never re-runs either check itself, so it can't
be fooled by a caller that forgot to run one; it just inspects the
results it's handed.

Never commits or pushes: prints the exact git commands and stops. Target
directory defaults to models/production_model, overridable via the
MODEL_DIR env var (or an explicit target_dir argument) so tests and the
Stage 3b live check promote into a temp copy instead of the real
deployment artifact.
"""

import datetime
import json
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_TARGET_DIR = "models/production_model"


class PromotionRefused(RuntimeError):
    """The gate decision or the serving-compat check didn't pass - or
    wasn't run at all. promote() raises this instead of silently
    no-op'ing, so a caller that ignores the return value still can't end
    up thinking a promotion happened."""


def resolve_target_dir(target_dir=None):
    return target_dir or os.environ.get("MODEL_DIR") or DEFAULT_TARGET_DIR


def _git_parent_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _next_version(target):
    version_path = Path(target) / "VERSION.json"
    if version_path.exists():
        try:
            return int(json.loads(version_path.read_text())["version"]) + 1
        except Exception:
            pass
    return 1


def promote(exported_model_dir, gate_result, compat_result, target_dir=None, metrics=None):
    """exported_model_dir: an mlflow.sklearn.save_model-style directory
    (e.g. from src.live.serving_compat.export_candidate) - copied
    verbatim into the target. gate_result: decide()'s return value.
    compat_result: check_serving_compat()'s return value. Both are
    required and both must show success, or this raises PromotionRefused
    without touching the filesystem."""
    if not gate_result or not gate_result.get("promoted"):
        raise PromotionRefused(f"gate did not pass: {gate_result.get('reasons') if gate_result else 'no gate result'}")
    if not compat_result or not compat_result.get("passed"):
        raise PromotionRefused(
            f"serving-compat check did not pass: max_abs_diff="
            f"{compat_result.get('max_abs_diff') if compat_result else 'no compat result'}"
        )

    target = Path(resolve_target_dir(target_dir))
    version = _next_version(target)

    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(exported_model_dir, target)

    version_info = {
        "version": version,
        "promoted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "parent_commit": _git_parent_commit(),
        "metrics": metrics or {},
        "holdout_metadata": gate_result.get("holdout_metadata", {}),
        "gate": {
            "pr_auc_diff_point": gate_result.get("pr_auc_diff_point"),
            "ci_lower": gate_result.get("ci_lower"),
            "ci_upper": gate_result.get("ci_upper"),
            "margin": gate_result.get("margin"),
        },
    }
    (target / "VERSION.json").write_text(json.dumps(version_info, indent=2))

    print(f"Promoted to {target} as version {version}.")
    print("To commit and push this model (not run automatically):")
    print(f"  git add {target}")
    print(f'  git commit -m "Promote model version {version}"')
    print("  git push")

    return version_info
