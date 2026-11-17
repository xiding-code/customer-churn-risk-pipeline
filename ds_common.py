"""Shared config + loaders for the Online Retail II / Plan D project.

Single source of truth so separate notebooks (01_cleaning, 02_eda, 03_features, ...)
never drift or mis-reference each other:
- paths are resolved from THIS file's location, not the notebook's cwd
- column names + decision constants are defined ONCE here
- loaders assert the expected schema and fail LOUD if an artifact is missing/stale/wrong

Usage in any notebook:
    from ds_common import *
    events = load_events()          # schema-checked, prints provenance
"""
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, brier_score_loss, log_loss)

# --- paths (resolved from this file -> robust to whatever cwd the notebook runs in) ---
HERE     = Path(__file__).resolve().parent            # .../Zerve_DS_Assignment/notebooks
DATA_DIR = HERE.parent / "Dataset" / "UCI"
PROC_DIR = DATA_DIR / "processed"
XLSX_PATH = DATA_DIR / "online_retail_II.xlsx"
RAW_PKL   = DATA_DIR / "online_retail_II_raw.pkl"
EVENTS_PKL      = PROC_DIR / "events.pkl"
CLEAN_LINES_PKL = PROC_DIR / "clean_lines.pkl"
RETURNS_PKL     = PROC_DIR / "returns.pkl"
FEATURES_PKL    = PROC_DIR / "features_tabular.pkl"   # 03_features output

# --- column names (import these; never retype the strings) ---
KEY     = "Customer ID"
INVOICE = "Invoice"
DATE    = "InvoiceDate"
QTY     = "Quantity"
PRICE   = "Price"
STOCK   = "StockCode"
DESC    = "Description"
COUNTRY = "Country"

# --- decision constants (agreed in cleaning/EDA) ---
RANDOM_SEED   = 42    # single seed for all notebooks (reproducibility)
HORIZON_DAYS  = 90    # repurchase / churn label window
# (MIN_PURCHASES removed: EDA decided not to pre-filter the label population; sequence branch handles it)

# --- schema contracts: producer (01_cleaning) must satisfy these or loaders raise ---
EVENTS_COLS      = [KEY, INVOICE, "event_time", "n_items", "basket_qty", "basket_value"]
CLEAN_LINES_COLS = [INVOICE, STOCK, DESC, QTY, DATE, PRICE, KEY, COUNTRY, "line_revenue"]


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found at {path}.\n"
            f"-> Run 01_cleaning_online_retail_II.ipynb first; it writes the processed pkls."
        )
    return path


def _check_cols(df: pd.DataFrame, expected, name: str) -> pd.DataFrame:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing expected columns {missing}. "
            f"Got {list(df.columns)}. Re-run 01_cleaning (schema changed)."
        )
    return df


def _provenance(path: Path, df: pd.DataFrame) -> None:
    mt = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    print(f"loaded {path.name}: {len(df):,} rows x {df.shape[1]} cols | saved {mt}")


def load_events(verbose: bool = True) -> pd.DataFrame:
    p = _require(EVENTS_PKL); df = pd.read_pickle(p)
    _check_cols(df, EVENTS_COLS, "events.pkl")
    if verbose: _provenance(p, df)
    return df


def load_clean_lines(verbose: bool = True) -> pd.DataFrame:
    p = _require(CLEAN_LINES_PKL); df = pd.read_pickle(p)
    _check_cols(df, CLEAN_LINES_COLS, "clean_lines.pkl")
    if verbose: _provenance(p, df)
    return df


def load_returns(verbose: bool = True) -> pd.DataFrame:
    p = _require(RETURNS_PKL); df = pd.read_pickle(p)
    if verbose: _provenance(p, df)
    return df


def load_features(verbose: bool = True) -> pd.DataFrame:
    p = _require(FEATURES_PKL); df = pd.read_pickle(p)
    _check_cols(df, [KEY, "cutoff", "split", "churn"], "features_tabular.pkl")
    if verbose: _provenance(p, df)
    return df


# --- metric suite (single source of truth; spec §6) ---
# Imported by every modeling notebook so baselines and the GRU are scored identically.
# NOTE: definitions are byte-for-byte the ones first validated inline in 04_baseline;
# keep them here only, never redefine in a notebook (prevents scoring drift).

def at_k(y_true, score, frac):
    """Precision/recall/lift among the top `frac` fraction by descending score."""
    n = max(1, int(len(score) * frac)); idx = np.argsort(-score)[:n]; sel = y_true[idx]
    prev = y_true.mean(); prec = sel.mean(); rec = sel.sum() / max(1, y_true.sum())
    return prec, rec, (prec / prev if prev > 0 else np.nan)


def ece(y_true, p, bins=10):
    """Expected calibration error over `bins` equal-width probability bins."""
    e = 0.0; edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() > 0: e += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return e


def evaluate_classifier(name, y_true, p, thr=0.5):
    """Full metric row: ROC/PR-AUC (+prevalence baseline), P/R/F1@thr, confusion,
    Brier, LogLoss, ECE, and Precision/Recall/Lift @10% & @20%. `y_true` is a 1-D array."""
    prev = y_true.mean(); yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    r = {"model": name, "n": len(y_true), "prevalence": round(prev, 4),
         "ROC_AUC": round(roc_auc_score(y_true, p), 4),
         "PR_AUC": round(average_precision_score(y_true, p), 4), "PR_AUC_base": round(prev, 4),
         "precision@.5": round(precision_score(y_true, yhat, zero_division=0), 4),
         "recall@.5": round(recall_score(y_true, yhat, zero_division=0), 4),
         "F1@.5": round(f1_score(y_true, yhat, zero_division=0), 4),
         "Brier": round(brier_score_loss(y_true, p), 4),
         "LogLoss": round(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)), 4),
         "ECE": round(ece(y_true, p), 4), "TN": tn, "FP": fp, "FN": fn, "TP": tp}
    for f in (0.1, 0.2):
        pr, rc, lf = at_k(y_true, p, f); k = int(f * 100)
        r[f"P@{k}%"] = round(pr, 3); r[f"R@{k}%"] = round(rc, 3); r[f"Lift@{k}%"] = round(lf, 2)
    return r


SEQUENCES_NPZ = PROC_DIR / "sequences.npz"   # 03b output

def load_sequences(verbose: bool = True):
    """Load the GRU tensors (03b). Returns a dict of arrays; schema-checked."""
    import numpy as np
    p = _require(SEQUENCES_NPZ); z = np.load(p, allow_pickle=False)
    expected = ["X_seq", "X_mask", "X_static", "y", "seq_len", "customer_id", "cutoff", "split"]
    missing = [k for k in expected if k not in z.files]
    if missing:
        raise ValueError(f"sequences.npz missing arrays {missing} — re-run 03b_sequences.ipynb.")
    d = {k: z[k] for k in expected}
    n, L, F = d["X_seq"].shape
    assert d["X_mask"].shape == (n, L) and len(d["y"]) == n, "sequences.npz shape mismatch"
    if verbose:
        print(f"loaded sequences.npz: n={n:,} L={L} F={F} | splits "
              f"{dict(zip(*__import__('numpy').unique(d['split'], return_counts=True)))}")
    return d
