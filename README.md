# Customer Churn & Revenue Retention Engine (SQL Rule Baseline vs. Random Forest)

**Author:** Ugochukwu Henry Emenike | **Consultancy:** [SIA Benics Digital Solutions](https://benicsdigitalsolutions.com)

## 1. Business Objective
Identify subscription customers at high risk of churning before cancellation and prioritize retention outreach by **Annual Revenue at Risk (EUR)**. This repository benchmarks a traditional deterministic SQL/Excel business rule against a supervised **Random Forest** machine learning classifier across **7,043** customer accounts (`26.5%` baseline churn rate).

## 2. Repository Contents
* **`customer_churn_dual_path.ipynb`**: End-to-end Google Colab notebook performing data cleaning, train/test splitting (`80/20` stratified), Path 1 (SQL/Excel rule baseline), Path 2 (Random Forest classifier), ROC-AUC evaluation, and feature importance extraction.
* **`High_Risk_Retention_List.csv`**: Actionable target list of **159** holdout accounts with $\ge 60\%$ predicted churn probability, sorted descending by `Annual_Revenue_EUR` for immediate account manager outreach.
* **`churn_drivers_chart.png`**: Top 10 behavioral and billing features driving customer churn according to Random Forest Gini importance.

## 3. Dual-Track Comparison: Normal SQL Rules vs. Machine Learning (Holdout Test Set)
| Approach | Methodology | Accuracy | Churners Caught (Recall) | ROC-AUC | Annual At-Risk Revenue Caught |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Path 1: Normal Rule Baseline** | Deterministic SQL/Excel flag (`Contract = 'Month-to-month'` AND `tenure < 12`) | **75.3%** | **54.8%** | — | **EUR 156,355.80** |
| **Path 2: Random Forest ML** | Supervised ensemble (`200` trees, `max_depth=8`) trained across 20 billing, contract, and service features | **80.3%** | **50.0%** | **0.845** | **EUR 171,194.40** |

## 4. Key Insights & Executive Takeaways
* **Higher Precision & Financial ROI (+EUR 14,838.60 Revenue Protected):** While the static SQL rule casts a wide net over all short-tenure month-to-month users (`75.3%` accuracy), the Random Forest classifier eliminates false positives (`80.3%` accuracy) and isolates high-ARPU churners—capturing **+9.5% more at-risk annual revenue** (`EUR 171,194.40` vs. `EUR 156,355.80`) on the 20% test cohort alone.
* **Top Churn Drivers:** Customer **tenure**, **TotalCharges**, **MonthlyCharges**, **Fiber optic internet service**, and **Two-year / One-year contract status** are the strongest predictors of retention.
* **Operational Action Plan:** Route the **159** flagged accounts in `High_Risk_Retention_List.csv` weekly to Customer Success teams, prioritizing targeted annual-contract upgrade incentives for high-monthly-spend subscribers.

## 5. Tech Stack & Dataset
* **Tools:** Python (`pandas`, `scikit-learn`, `matplotlib`), Google Colab
* **Dataset:** [Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv` — Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
