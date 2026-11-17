"""Churn risk-scoring API (Zerve deployment, spec §8).

Contract: the API receives VALIDATED as-of feature payloads (computed offline by the
03 pipeline as of a stated cutoff). No online feature engineering / return-matching.
Operating point is RANK-BASED top-k% within the scored batch (see 04: probability
calibration on val fails under base-rate drift; absolute thresholds are unstable).

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Deps: fastapi, uvicorn, joblib, numpy, pandas, scikit-learn, xgboost
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_VERSION = "baseline-2026-07-04"
ARTIFACTS = Path(__file__).resolve().parent / "baseline_artifacts.joblib"  # upload next to app.py
DEFAULT_MODEL = "logreg"          # best full-test ROC (0.7637); rf_pruned is the best-calibrated option (see 04) — both shipped
TOP_K = 0.10                      # top-10% of the scored batch = "HIGH" tier (04 operating point)

bundle = joblib.load(ARTIFACTS)
FEAT_COLS = bundle["feat_cols"]
SKEW = bundle["skew_cols"]
MED = bundle["impute_median"]
SCALER = bundle["scaler"]
MODELS = {
    "logreg": bundle["model_logreg"],
    "boosted": bundle["model_boosted"],
    "rf_pruned": bundle["model_rf_pruned"],
}

app = FastAPI(title="Customer churn risk API", version=MODEL_VERSION)


class Batch(BaseModel):
    """rows: list of feature dicts. Every dict must contain exactly the 31 features
    of the frozen schema (see feature dictionary in 03); days_since_last_return may be null."""
    rows: list[dict]
    model: str = DEFAULT_MODEL


def _matrix(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    missing = [c for c in FEAT_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in FEAT_COLS]
    if missing or extra:
        raise HTTPException(422, f"schema mismatch — missing {missing}, unexpected {extra}")
    df = df[FEAT_COLS].apply(pd.to_numeric, errors="coerce")   # JSON nulls/strings -> float NaN (tree models are NaN-native)
    return df


def _score(df: pd.DataFrame, model_name: str) -> np.ndarray:
    if model_name not in MODELS:
        raise HTTPException(422, f"unknown model '{model_name}'; use one of {list(MODELS)}")
    if model_name == "logreg":
        d = df.copy()
        d["days_since_last_return"] = d["days_since_last_return"].fillna(MED)
        for c in SKEW:
            if c in d:
                d[c] = np.log1p(d[c].clip(lower=0))
        return MODELS[model_name].predict_proba(SCALER.transform(d.values))[:, 1]
    return MODELS[model_name].predict_proba(df.values)[:, 1]  # tree models take NaN natively


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "models": list(MODELS),
            "n_features": len(FEAT_COLS), "operating_point": f"rank-based top-{int(TOP_K*100)}%"}


@app.post("/score_batch")
def score_batch(batch: Batch):
    df = _matrix(batch.rows)
    p = _score(df, batch.model)
    order = np.argsort(-p)
    k = max(1, int(len(p) * TOP_K))
    tier = np.array(["STANDARD"] * len(p), dtype=object)
    tier[order[:k]] = "HIGH"
    return {"model": batch.model, "model_version": MODEL_VERSION, "n": len(p),
            "scores": [round(float(x), 4) for x in p], "tier": tier.tolist(),
            "note": "tier = rank-based top-k within THIS batch; probabilities are uncalibrated "
                    "for absolute thresholds under base-rate drift (see 04)."}
