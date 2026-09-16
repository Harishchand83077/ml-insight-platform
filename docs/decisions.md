# here we will note down every non obvious decisions 

## EDA findings (Day 1)

- Churn rate is 26.5% — imbalanced. Will use precision/recall/F1/PR-AUC
  instead of accuracy as primary model metrics in Week 3.

- TotalCharges has 11 blank values, all with tenure=0 (new customers,
  no billing cycle yet). Decision: impute as 0, not drop or mean-impute —
  these are legitimately zero-charge customers, not missing data.

- Strongest churn signals found: Contract type (month-to-month 42.7%
  churn vs two-year 2.8%), tenure (newest customers churn most, 47%
  in first year vs 9.5% after 4+ years), and lack of TechSupport/
  OnlineSecurity add-ons.

- Counterintuitive finding: Fiber optic customers churn more (42%)
  than DSL (19%) despite being a "premium" service — flagging for
  further investigation rather than assuming data error. Possible
  pricing/competition effect.

  ## EDA findings (Day 1, continued)

- No duplicate customerIDs — data integrity confirmed.

- TotalCharges is highly collinear with tenure (r=0.83) and
  MonthlyCharges (r=0.65) — expected, since it's roughly their product.
  Decision: [pick one — e.g. "drop TotalCharges from modeling features,
  derive tenure and MonthlyCharges as the independent signals instead"]

- All 6 add-on service columns (OnlineSecurity, OnlineBackup, etc.)
  share the same 3-value pattern, with "No internet service" being a
  redundant restatement of InternetService=No. Decision: engineer
  num_addons_active (count of Yes across the 6) and drop redundant
  categorical duplication of "No internet service" per column.

- Gender shows ~no churn gap (27% vs 26%) — sanity check confirming
  earlier signals (contract, tenure, add-ons) are real, not noise
  from an overly permissive method.

- MonthlyCharges boxplot: churners have higher median charge, but
  non-churners show a much wider low-end spread — likely a sticky
  low-cost/single-service segment. Worth a targeted look at customers
  with MonthlyCharges < 30 in feature engineering.