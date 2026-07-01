 # Credit Risk Intelligence Platform


Predicting loan defaults with explainable AI — because knowing who will default isn't enough. You need to know why.



A 5-layer end-to-end credit risk pipeline built on 32,000 real borrower records. This project goes beyond a standard ML classifier — it clusters borrowers into personas, audits the model for fairness, explains individual predictions using SHAP, and uses the Groq API to auto-generate plain-English credit officer memos.


# The Problem

Lending institutions need to decide in minutes whether to approve a loan. A model that just outputs a number isn't actionable — a loan officer needs to know which borrowers are risky, why they're risky, and whether the model is treating all groups fairly.

This project builds that full decision-support system.


# 5-Layer Pipeline

Layer 1 — Data Ingestion & SQL Schema

Raw borrower data is loaded into a relational SQLite database. Missing values in interest rate and employment length are handled, outliers removed, and exploratory analysis surfaces the first insight: a 21.5% overall default rate that varies dramatically by loan purpose and grade.

Layer 2 — Feature Engineering & Borrower Clustering

31 new features are engineered across four categories — financial stress, credit risk signals, stability indicators, and intent-based risk. KMeans clustering (k=2, silhouette-selected) reveals two distinct borrower personas:

PersonaDefault RateAvg IncomeAvg LoanCautious Low-Income Borrower12%$73,654$7,269At-Risk Young Borrower41%$52,624$14,289

The two personas default for different reasons — confirmed later by per-persona SHAP analysis.

Layer 3 — XGBoost Modelling

XGBoost and Random Forest are trained with class imbalance correction. XGBoost wins with 0.9519 ROC-AUC. The decision threshold is tuned using F2-score (prioritising recall over precision) since missing a true default is costlier than a false alarm.

Layer 4 — Fairness Audit

The model is audited using Disparate Impact Ratio across age groups and home ownership categories. All groups pass the 0.80 DIR threshold — the model does not systematically discriminate. This step is what separates a production-ready model from a Kaggle submission.

Layer 5 — SHAP Explainability + AI Credit Memos

SHAP TreeExplainer computes feature attributions for every test borrower. Top global drivers: loan grade, income level, home ownership stability, and debt-to-income ratio. These SHAP values are then passed to the Groq API (openai/gpt-oss-120b) which generates a structured credit officer memo — in plain English, no ML jargon — for any borrower on demand.
