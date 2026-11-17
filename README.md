# Customer Churn-Risk Pipeline (Online Retail II)

End-to-end pipeline that turns raw retail transactions into a customer churn-risk score, risk bands and a FastAPI scorer.
Author: Xincheng (George) Ding. Full write-up: [`REPORT.md`](REPORT.md).

## Key results

| Step | Result |
|---|---|
| Raw data | 1,067,371 line items (UCI Online Retail II) |
| Feature panel | 47,933 customer-month rows x 39 features (5,212 customers, 16 monthly cutoffs), leakage unit tests, temporal split frozen before modeling |
| Deployed model | Logistic regression: test ROC-AUC 0.7637, PR-AUC 0.7077, top-decile precision 0.797, 1.70x lift |
| Sequence model | GRU (Keras on the PyTorch backend), same-subset test ROC-AUC 0.7581 vs 0.7526 for logistic regression; customer-clustered bootstrap CI for the ROC gain (0.0004, 0.0101) excludes zero, so the gain is real but marginal and the simpler model ships |
| Risk bands | Realized churn 80.3% / 74.3% / 64.8% / 32.0% from highest to lowest band |

## Notebooks

1. [`01_cleaning_online_retail_II.ipynb`](01_cleaning_online_retail_II.ipynb): de-duplication, returns netted only against earlier purchases
2. [`02_eda_online_retail_II.ipynb`](02_eda_online_retail_II.ipynb): exploratory analysis
3. [`03_features.ipynb`](03_features.ipynb): leak-safe as-of feature panel over 16 monthly cutoffs
4. [`03b_sequences.ipynb`](03b_sequences.ipynb): per-customer purchase sequences for the GRU
5. [`04_baseline.ipynb`](04_baseline.ipynb): baseline ladder, ablations, calibration and operating point
6. [`05_gru.ipynb`](05_gru.ipynb): GRU sequence model and bootstrap comparison

Shared metrics: [`ds_common.py`](ds_common.py). Batch scoring API: [`app.py`](app.py) (FastAPI).

## Data

Online Retail II (Chen, D., 2019), UCI Machine Learning Repository, CC BY 4.0: https://archive.ics.uci.edu/dataset/502/online+retail+ii . Raw data is not included; place the downloaded file under `Dataset/UCI/` to re-run.
