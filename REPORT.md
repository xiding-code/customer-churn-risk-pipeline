# Customer Churn-Risk Intelligence — Online Retail II

*A deployable workflow that turns raw retail transactions into an intelligence system to **identify, explain, prioritize, and act on** customers at risk of not repurchasing within 90 days.*

**Dataset attribution.** Online Retail II (Chen, D., 2019), UCI Machine Learning Repository, CC BY 4.0. Every number in this report is reproduced from the executed pipeline; the full 01→04 workflow is re-run live on Zerve and reconciled `allclose` against the shipped artifacts, and the sequence model (05) is verified by CPU inference.

---

## 1. Data cleaning & exploratory analysis

Two raw sheets were concatenated and de-duplicated, cancellations were split from purchases, non-product lines were removed, and each return was netted against **prior** purchases only (a return can never offset a purchase dated after it), LIFO within the eligible window.

| Stage | Result |
|---|---|
| Raw rows | 1,067,371 |
| After de-duplication | 1,033,036 |
| Returns / purchases | 19,104 / 1,013,932 |
| Valid product purchase lines | 1,003,214 |
| Lines after time-constrained netting | 770,551 (return qty 472,976; **95.2% matched**) |
| Gross+net basket events | **36,316 across 5,839 customers** |

**Purchase behaviour.** Median **3** baskets per customer (mean 6.22, max 370); repeat rate **72.4%** (Wilson 95% CI 71.2–73.5%). Per-customer median inter-purchase gap is **57 days**.

**Revenue concentration.** The top **1%** of customers (59 customers) account for **31.1%** of revenue; the top 10% account for **63.1%** — the data mixes ordinary shoppers with wholesale buyers.

**Geography.** **90.1% UK** (Germany 2.0%, Ireland 2.0%).

**Rolling churn panel.** Across **16 monthly cutoffs**, the panel holds **47,813 rows / 5,204 customers**; per-cutoff churn swings **38.0–62.3%** with the calendar (pooled **50.9%**).

**Seasonality.** Orders, revenue, and active customers peak in **November 2011**; average basket value peaks in **January 2010**.

---

## 2. Feature engineering & leakage discipline

For each of 16 monthly as-of cutoffs `T`, features are built from **only** history dated `≤ T`, and the label is taken from the forward 90-day window `(T, T+90d]`. This is enforced by explicit self-checks (`max_purchase_time_used ≤ cutoff` for every row; returns netted with the strict-before-date constraint; no forward-looking information in any feature).

| Split | Rows | Churn |
|---|---|---|
| Train | 23,987 | 50.2% |
| Validation | 9,985 | 58.0% |
| Test | 13,961 | 46.9% |

The feature panel is **(47,933 × 39)** across 5,212 customers. Regenerating the panel live on Zerve reproduces the shipped feature store **exactly** (max absolute difference `1.78e-15`, churn labels identical); the GRU sequence tensors reconcile to `0.00` maximum difference.

---

## 3. Model ladder (temporal test, scored once)

A ladder from a naive baseline up to boosted trees was evaluated on the held-out temporal test set (13,961 samples, 46.9% churn prevalence). Logistic regression is **retrained live on Zerve** and reproduces the shipped scores (`allclose`, max abs diff `1.57e-08`); the tree models are loaded from the shipped artifact and re-scored, because tree training with subsampling is path-dependent across machines.

| Model | ROC-AUC | PR-AUC | Precision@10% |
|---|---|---|---|
| Naive (prevalence) | 0.5000 | 0.4693 | — |
| Recency only | 0.6696 | 0.6300 | 0.718 |
| **Logistic Regression (deployed)** | **0.7637** | **0.7077** | **0.797** |
| XGBoost | 0.7551 | 0.7015 | 0.799 |
| Random Forest (pruned) | 0.7589 | 0.7022 | 0.799 |

Learning beats single-feature ranking (LogReg ROC 0.7637 vs recency-only 0.6696). Boosted and tree models are statistically comparable; **logistic regression is deployed** for its full-population coverage and interpretability. Top-decile lift is **1.70×** (P@10% 0.797 vs. base prevalence 0.469). *(The naive model's P@10% is undefined — all its scores are tied — so it is omitted.)*

**Reading the ROC / PR panel.** The three learned models' ROC and precision-recall curves overlap almost exactly (AUC 0.7637 / 0.7551 / 0.7589; average precision ≈ 0.70 each), so the boosted trees add no meaningful discriminative edge over logistic regression. All three sit well above the prevalence line (0.469), confirming genuine signal; the tie is what justifies deploying the simplest, most interpretable model.

---

## 4. Rank-based operating points

Selection uses **rank within each cutoff**, never a fixed probability threshold (calibration is not stable under base-rate drift). Each customer receives a `risk_decile` (1–10) and a `risk_band`: **Critical** (top 10%), **High** (80–90th percentile), **Medium** (50–80th), **Low** (bottom 50%).

**Actual churn by risk band** (full 47,933-customer scoring):

| Risk band | Customers | Actual churn |
|---|---|---|
| Critical | 4,800 | 80.3% |
| High | 4,794 | 74.3% |
| Medium | 14,376 | 64.8% |
| Low | 23,963 | 32.0% |

The banding is monotonic — the Critical band's realized churn (80.3%) is more than double the Low band's (32.0%).

**Churn reason codes** (deterministic decision rules, highest-priority first):

| Reason code | Customers | Rule |
|---|---|---|
| RECENCY_DOMINANT | 14,363 | last purchase > 90 days ago |
| STABLE | 12,378 | none of the below |
| LOW_ENGAGEMENT | 10,619 | ≤ 2 lifetime purchases |
| SPEND_DECLINING | 9,763 | 90-day spend trend negative and recent spend < half of the prior window |
| RETURN_SENSITIVE | 810 | return rate > 20% or any fully-returned order |

**High-value, high-risk segment.** Customers with `net_monetary_total ≥ £500` in the Critical/High bands number **573**, with a **79.1%** actual churn rate — the priority list for retention outreach, ranked by net value.

**Reading the risk dashboard.** The composition chart confirms the ranking is monotone — realized churn falls cleanly 80.3% → 74.3% → 64.8% → 32.0% with no inversions. The cumulative-gains curve quantifies the payoff: contacting the top 10% of ranked customers captures **17.0%** of all churners, a **1.70×** lift over random outreach. The revenue donut reframes priority: the Low band holds **£10.82M (83%)** of net revenue at only 32% churn, while Critical and High together hold just **£0.49M (4%)**. Absolute revenue-at-risk in the top bands is therefore small — the sharper target is the 573 high-value customers above, not the raw band totals.

**Customer segments (k-means, latest cutoff).** Clustering the latest-cutoff base (n=2,768) on as-of RFM + tenure + return-rate features gives a silhouette that peaks at k=2 (0.333); k=3 (0.303) is comparable and splits the base into three actionable personas: a **recency-driven lapsing** group (C0, n=1,292, 45% churn), a **low-engagement** group (C1, n=677, 56% churn), and a **stable core** (C2, n=799, 17% churn). The forward label is used only to describe the clusters, never to fit them.

---

## 5. Self-serve analytics: queryable feature store + natural-language SQL

The monthly `customer_churn_scores` table is the downstream asset. It is registered in DuckDB so analysts can self-serve; the schema doubles as a data dictionary. Three template queries ship with it:

- **Latest-cutoff Critical retention list** (top candidates by net value).
- **High-value high-risk segment** (`net_monetary_total ≥ 500` and band ∈ {Critical, High}).
- **Churn by return behaviour** (using the *actual* label, not the score):

| Return group | Customers | Actual churn |
|---|---|---|
| No returns | 27,988 | 60.7% |
| Low returns (< 20%) | 19,201 | 36.1% |
| High returns (≥ 20%) | 744 | 63.6% |

Healthy return behaviour correlates with retention (low-return customers churn at 36.1%), while high-return customers — likely experiencing product misfit — show elevated churn (63.6%).

A **natural-language-to-SQL** assistant closes the loop: a plain-English question is turned into an executable DuckDB query, run, and returned as a small report — analysts and stakeholders can explore churn patterns without hand-writing SQL.

---

## 6. Scope caveat

This is a **UK model**: the population is ~90% UK-based. Predictions and operating points are calibrated to UK customer dynamics; non-UK segments (Germany, Ireland, other) may exhibit different churn patterns and should be validated separately before the model is applied to them.

---

## 7. Complementary sequence model (GRU)

A gated-recurrent-unit sequence model was trained (locally) on the ≥2-purchase subset, ingesting up to 20 historical transactions with 7 features per event (inter-event gap, quantity, amount, item count, seasonality, first-purchase flag), left-padded and masked. On the sequence-eligible test set (**11,627** customers) it scores ROC **0.7581** / PR **0.6617** / P@10% **0.750** / Lift **1.80**. It is verified on Zerve by CPU inference (weights loaded, not retrained). On the same subset the tabular models score: LogReg 0.7526, XGBoost 0.7406, Random Forest 0.7468 (ROC-AUC) — the GRU adds a modest edge where sequence patterns exist.

---

## 8. Monitoring

A retrospective drift assessment was run over the 16 historical monthly cutoffs (June 2010 – September 2011) using the same leak-safe protocol (features `≤` cutoff; labels 90 days forward):

- **Churn prevalence** ranges 0.38–0.62 across cutoffs.
- **Calibration:** the gap between mean predicted score and realized churn widens over the later cutoffs — a calibration-degradation signal that motivates the rank-based (not threshold-based) operating points.
- **Precision@10%** stays within roughly 0.75–0.87 across the timeline.
- **Median recency** (a feature-drift proxy) ranges 33–89 days.

**Reading the monitoring panel.** The prevalence trace (top-left) swings 0.38–0.62 with the retail calendar, not with model failure. The top-right panel is the key watch-item: mean predicted score tracks actual churn on the early cutoffs, then the two diverge later (predicted peaks ~0.66 vs actual ~0.60) — the widening gap is the calibration drift that justifies rank-based operating points. Discrimination itself holds up: precision@10% stays at 0.75–0.87 throughout (bottom-left), and median recency (bottom-right) stays in a 33–89-day band with no runaway trend.

In production the same block runs on each new month's scores; rank-based bands remain more stable under drift than absolute probability thresholds.

---

## Deployment summary

The logistic-regression model is deployed behind a FastAPI scoring service; scores are delivered as rank-based risk bands (Critical/High/Medium/Low) with reason codes. A DuckDB self-serve layer with a natural-language-to-SQL interface supports ad-hoc analysis, and the GRU is available as a complementary signal for customers with ≥ 2 purchases. Full source, executed outputs, and exact cell code (`01_cleaning`–`05_gru`, `ds_common.py`, pinned `requirements.txt`) accompany this report.
