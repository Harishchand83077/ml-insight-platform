---
title: "Model Card: XGBoost Churn Classifier"
doc_type: model_card
version: "1.3"
last_updated: "2026-08-01"
---

# Model Card: XGBoost Churn Classifier

This is the production churn-prediction model served by `/predict` and
`predict_churn_tool`. It is trained on Vantrix's customer data (see the
synthetic-data note under Training Data below — the underlying dataset
is a synthetic telecom churn dataset, not real customer data from any
company).

## Model Details

- **Model type**: XGBoost gradient-boosted tree classifier
  (`xgboost.XGBClassifier`), wrapped in a scikit-learn `Pipeline` with a
  `ColumnTransformer` preprocessing step (one-hot encoding for
  categorical columns, passthrough for numeric/binary columns).
- **Hyperparameters**: default XGBoost parameters with `random_state=42`
  and `eval_metric="logloss"` — no hyperparameter tuning has been
  performed yet; this is explicitly a baseline model, tracked alongside
  a Logistic Regression baseline for comparison (see "Why XGBoost and
  Logistic Regression were both kept" in the project glossary).
- **Output**: a churn probability in [0, 1] for a given customer, plus a
  binary prediction (`Yes`/`No`) using a 0.5 decision threshold.
- **Serialization**: exported via `skops` (not a raw pickle) with
  explicit trusted types declared for `xgboost.core.Booster` and
  `xgboost.sklearn.XGBClassifier`, loaded directly at serving startup
  from `models/production_model/model.skops`.

## Training Data

The model is trained on `customer_features` joined with the `churn`
label from `customers`, both tables in the project's Postgres database.
**This is a synthetic dataset** generated for this project (the base
structure follows the well-known IBM Telco Customer Churn schema, with a
synthetic behavioral-event layer added on top — login counts, ticket
counts, usage ratios). It does not represent real Vantrix customers (a
fictional company) or any real telecom provider's customers. The overall
churn rate in the training data is **26.5%**, making this an imbalanced
classification problem.

An earlier version of the synthetic event-data generator introduced a
data leakage issue — churned customers had near-deterministic patterns
in the event-derived features that let early models score unrealistically
well (F1 ~0.96-0.99). This was identified and fixed by reworking the
generator to use randomized, overlapping distributions instead of
deterministic group-level rules; see the glossary entry "The leakage
issue we found and fixed" for the full account. All metrics in this
model card are from the **post-fix** data.

## Features Used

Three categorical (`contract`, `payment_method`, `internet_service`),
two binary (`partner`, `dependents`), and eleven numeric columns
(`tenure`, `monthly_charges`, `senior_citizen`, `num_addons_active`,
`total_logins_90d`, `recent_30d_vs_older_60d_ratio`,
`avg_session_duration_recent_30d`, `num_tickets`, `pct_unresolved`,
`avg_resolution_time_hours`, `avg_usage_count`) — see the Data
Dictionary for `customer_features` for the full definition of each.

## Performance Metrics

Evaluated on a stratified 20% held-out test split (`random_state=42`):

| Metric | XGBoost | Logistic Regression |
|---|---|---|
| Accuracy | ~0.80 | ~0.79 |
| Precision | ~0.66 | ~0.64 |
| Recall | ~0.72 | ~0.69 |
| F1 | **0.69** | 0.66 |
| ROC-AUC | ~0.84 | ~0.83 |
| PR-AUC | **0.78** | 0.77 |

PR-AUC and F1 are treated as the primary metrics, not accuracy, because
of the 26.5%/73.5% class imbalance — a model predicting "No churn" for
every customer would score ~74% accuracy while being useless. See the
glossary entry "PR-AUC vs ROC-AUC" for the full reasoning.

## Known Limitations

- **No hyperparameter tuning**: this is an untuned baseline. Expect
  meaningful headroom from tuning (tree depth, learning rate, number of
  estimators) that hasn't been explored yet.
- **Synthetic training data**: behavioral features (`total_logins_90d`,
  `avg_usage_count`, etc.) come from a synthetic event generator, not
  real usage logs. Real-world deployment on genuine customer data would
  require retraining and re-validating against real distributions —
  this model should not be treated as validated for any real telecom
  deployment.
- **Static decision threshold**: the 0.5 threshold is not tuned for any
  specific business cost tradeoff (e.g., cost of a false positive
  retention offer vs. a missed at-risk customer); a production retention
  program should tune this threshold against actual offer costs.
- **No fairness/bias audit performed**: the model has not been evaluated
  for disparate performance across demographic subgroups (e.g.,
  `senior_citizen`).
- **No monitoring for concept drift in production** beyond the data-drift
  check in `src/models/monitor_drift.py`, which compares feature
  distributions, not label-relationship drift.
