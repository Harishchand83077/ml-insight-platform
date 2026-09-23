# Glossary

Key terms and decisions from the ML Insight Platform project, written out
in full rather than as one-liners, so they carry enough context to be
useful on their own — including to the project's own retrieval-augmented
agent.

## Churn

Churn is the event of a customer ending their relationship with the
company — in this dataset, a telecom provider — represented by the
binary `Churn` column (`Yes`/`No`) on the `customers` table. In our data,
26.5% of customers have churned, which makes this an imbalanced
classification problem: a model that always predicts "No churn" would
still be "right" about three-quarters of the time while being useless.
That imbalance is the reason accuracy is a poor primary metric here and
why the project tracks precision, recall, F1, and PR-AUC instead (see the
PR-AUC vs ROC-AUC entry below). Churn is the label the baseline models
(Logistic Regression and XGBoost) are trained to predict from the
`customer_features` table.

## Tenure

Tenure is the number of months a customer has been with the company,
stored as an integer column on both `customers` and `customer_features`.
It turned out to be one of the strongest signals in the dataset during
EDA: customers in their first year churn at roughly 47%, compared to
about 9.5% for customers with 4+ years of tenure. Intuitively, the
riskiest period for losing a customer is early in the relationship,
before habits and switching costs build up. Tenure is also collinear with
`TotalCharges` (which is roughly tenure × MonthlyCharges), which is one
of the reasons `TotalCharges` was dropped from the modeling features
rather than kept alongside a variable that already captures most of the
same information.

## PR-AUC vs ROC-AUC (and why we chose PR-AUC)

Both are threshold-independent summaries of a binary classifier's ranking
quality, but they answer different questions. ROC-AUC plots true positive
rate against false positive rate across all thresholds and asks "how well
does the model separate the two classes overall?" PR-AUC (area under the
precision-recall curve) plots precision against recall and asks "when the
model says churn, how often is it right, and how much of the actual
churn does it catch?" The critical difference is how each one treats the
majority class: ROC-AUC's false-positive-rate term is diluted by the huge
number of true negatives in an imbalanced dataset (74% "No churn" here),
so a model can post a deceptively strong ROC-AUC while still being
mediocre at the thing that actually matters — correctly flagging the
minority churn class. PR-AUC has no true-negative term at all, so it
stays sensitive to exactly the class we care about. That's why this
project treats PR-AUC (alongside F1) as the primary metric and reports
ROC-AUC as a secondary, supporting number rather than the headline
figure. In our current results, XGBoost reaches PR-AUC 0.78 and F1 0.69
against Logistic Regression's PR-AUC 0.77 and F1 0.66 — a modest but real
edge for the gradient-boosted model.

## Cache-aside pattern

Cache-aside (also called lazy loading) is a caching strategy where the
application, not the cache itself, is responsible for keeping the cache
populated. On a read, the app checks the cache first; on a hit, it
returns the cached value directly; on a miss, it queries the real data
store, writes the result into the cache with an expiry (TTL), and then
returns it. The next request for the same key is fast because it's now
in cache, until the TTL expires and the cycle repeats. This project
implements cache-aside in `src/serving/feature_cache.py`:
`get_customer_features(customer_id)` checks Redis for key
`features:{customer_id}` first, and on a miss falls back to a pooled
Postgres query against `customer_features`, caching the JSON result with
a 300-second TTL. The pattern is simple and resilient — if Redis is
unavailable, the app can still fall back to Postgres — but it does mean
the first request after a miss (or after any TTL expiry) pays the full
database latency. Measured locally, cache hits landed at 2-5ms versus
45-194ms for misses, a 10-92x speedup, though once the serving API added
Postgres connection pooling that gap narrowed considerably, since pooling
removed most of the miss-side overhead too.

## Data drift

Data drift is a change over time in the statistical distribution of the
features a model sees in production, compared to the distribution it was
trained on — as opposed to concept drift, which is a change in the
relationship between features and the label itself. A model can degrade
silently from drift alone even if its underlying logic is still sound,
because the inputs it's now seeing no longer resemble what it learned
from. This project uses Evidently (`src/models/monitor_drift.py`) to
compare the training-time reference distribution (the actual X_train used
by `train_baseline.py`) against a simulated "current" snapshot — a random
20% sample of `customer_features` with mild synthetic shifts injected
into three numeric columns. Evidently picks a statistical test per column
automatically (Wasserstein distance for numeric columns, Jensen-Shannon
distance for categorical ones, given our dataset size) and flags a column
as drifted when its score crosses a threshold. In our test run, two of
the three shifted columns were correctly flagged; the third
(`avg_resolution_time_hours`, shifted +20%) fell just under the
threshold because its distribution is right-skewed and high-variance, so
a proportional mean shift has less effect on a normalized distance score
than it would on a tighter distribution — a useful reminder that "we
injected a shift" and "the shift was large enough to be statistically
detectable" are not the same claim.

## num_addons_active

`num_addons_active` is an engineered feature in `customer_features`: an
integer count (0-6) of how many of the six add-on services — Online
Security, Online Backup, Device Protection, Tech Support, Streaming TV,
and Streaming Movies — a customer currently has active. It was built
during feature engineering specifically to collapse six separate
three-valued categorical columns (`Yes` / `No` / `No internet service`)
into a single, denser numeric signal, after EDA showed all six shared the
same redundant pattern (the "No internet service" value is really just a
restatement of `InternetService = No`, repeated six times). The six raw
columns are dropped after this feature is computed, since keeping both
the summary and the raw duplicated columns would add redundant,
collinear inputs without adding real information. In practice,
`num_addons_active` has held up as a meaningful predictor — customers
with more active add-ons tend to churn less, which lines up with the
original EDA finding that customers lacking Tech Support and Online
Security churn disproportionately more.

## The leakage issue we found and fixed

Early baseline models trained on the first version of the synthetic
event data scored unrealistically well — F1 around 0.96-0.99 and PR-AUC
around 0.99, far higher than real-world churn models typically achieve.
The root cause was in how the synthetic event generator worked, not in
the models themselves: it made churned customers *near-deterministically*
show declining logins, low feature usage, and unresolved support tickets,
so those event-derived features had effectively encoded the churn label
directly rather than correlating with it the way real behavioral data
would. The models weren't cheating by seeing the label — they were
legitimately learning a relationship that happened to be almost 1:1
because of how the fake data was constructed. The fix was to rework the
generator to use per-customer randomized trends and noise (churn-
conditional probabilities with wide, overlapping ranges) instead of
deterministic group-level rules, so churned and non-churned customers'
behavior genuinely overlaps while the correlation direction stays
correct on average. After regenerating the data and retraining, F1
dropped to a realistic 0.66-0.69 and PR-AUC to 0.77-0.78, and — as a
independent confirmation the fix worked as intended — feature importance
shifted away from the event-derived features dominating and back toward
`Contract` and `InternetService`, which is what the original EDA on the
real customer data had already identified as the strongest signals.

## Why XGBoost and Logistic Regression were both kept

The project trains and tracks two baseline models rather than picking
one upfront. Logistic Regression is fast to train, its coefficients are
directly interpretable (each one has a sign and a magnitude that map
onto "this pushes toward churn" or "this pushes toward retention" in a
way a non-technical stakeholder can follow), and it acts as a sanity-
check baseline: if a more complex model can't beat it by much, that's a
signal the extra complexity may not be earning its keep. XGBoost is a
gradient-boosted tree ensemble that can capture non-linear relationships
and feature interactions that a linear model structurally cannot, at the
cost of being harder to interpret directly (mitigated here by logging a
feature-importance plot alongside each run). Keeping both, logged side
by side in MLflow with the same metrics, makes the comparison explicit
rather than assumed — and in this project's case, XGBoost does modestly
outperform Logistic Regression (F1 0.69 vs 0.66, PR-AUC 0.78 vs 0.77),
but not by such a wide margin that the simpler, more interpretable model
is irrelevant. Both are logged as full scikit-learn `Pipeline` objects
(preprocessing plus classifier together), not just the bare estimator,
so either one could be swapped into serving without extra glue code.
