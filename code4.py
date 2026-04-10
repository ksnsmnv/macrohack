"""
macrohack_all_fixed_v3.py
=========================
DIAGNOSIS → FIXES:
  D1 → F1  SV lambda grid constrained: λ₁∈[0.3..2.5], λ₂∈[2.0..7.0], λ₂>λ₁+0.5
  D2 → F2  VAR variable cap MAX_VAR_TOTAL_VARS=7; IV block uses only iv_pc1 (iv_n=1)
  D3 → F3  SvenssonARIMAForecaster: two-phase fit + refit_arima_only()
  D4 → F4  Remove all raw-macro VAR specs; keep pca-only
  D5 → F5  New ARIMAX M2 (macro_pca+iv_pc1 as exog) + RW floor if no M2 beats RW
       F6  SV beta stability: clip betas ±500 after OLS
"""

from pathlib import Path
import warnings, itertools, os, json
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use(os.environ["MPLBACKEND"], force=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# §1  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def normalize_month_index(idx) -> pd.DatetimeIndex:
    return pd.to_datetime(idx).to_period("M").to_timestamp("M")


def to_num_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(
            out[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return out


def choose_d_by_adf(series: pd.Series, alpha: float = 0.05) -> int:
    s = pd.Series(series).dropna()
    if len(s) < 12:
        return 1
    try:
        return 0 if adfuller(s)[1] < alpha else 1
    except Exception:
        return 1


# ── Nelson-Siegel ──────────────────────────────────────────────────────────────

def ns_loadings(tau: float, lam: float) -> Tuple[float, float]:
    x = tau / lam
    if np.isclose(x, 0.0):
        return 1.0, 0.0
    l1 = (1 - np.exp(-x)) / x
    return l1, l1 - np.exp(-x)


def nelson_siegel_yield(tau, b0, b1, b2, lam):
    l1, l2 = ns_loadings(tau, lam)
    return b0 + b1 * l1 + b2 * l2


def reconstruct_yc_from_ns_betas(
    df_betas: pd.DataFrame,
    maturity_map: Dict[str, float],
    lam: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {lbl: [nelson_siegel_yield(tau, r["beta0"], r["beta1"], r["beta2"], lam)
               for _, r in df_betas.iterrows()]
         for lbl, tau in maturity_map.items()},
        index=df_betas.index,
    )


# ── Svensson (1994) ────────────────────────────────────────────────────────────

def sv_loadings(tau: float, lam1: float, lam2: float) -> Tuple[float, float, float]:
    """
    Svensson (1994) factor loadings.
    Reference: Otsenko & Seleznev, RJMF 84(2), eqs. (5)-(6).

    L1 = (1 − e^{−τ/λ₁}) / (τ/λ₁)              (level/slope)
    L2 = L1 − e^{−τ/λ₁}                          (first curvature)
    L3 = (1 − e^{−τ/λ₂}) / (τ/λ₂) − e^{−τ/λ₂}  (second curvature)

    NOTE for short curves (max 2Y): λ₂ > λ₁ + 0.5 is required to ensure
    L2 and L3 are statistically distinguishable. Violation → near-singular OLS.
    """
    def _ns_pair(tau, lam):
        x = tau / lam
        if np.isclose(x, 0.0):
            return 1.0, 0.0
        l1 = (1 - np.exp(-x)) / x
        return l1, l1 - np.exp(-x)

    l1, l2 = _ns_pair(tau, lam1)
    _,  l3 = _ns_pair(tau, lam2)
    return l1, l2, l3


def sv_yield(tau, b0, b1, b2, b3, lam1, lam2):
    l1, l2, l3 = sv_loadings(tau, lam1, lam2)
    return b0 + b1*l1 + b2*l2 + b3*l3


def fit_sv_betas_ols(
    df_yc: pd.DataFrame,
    yc_cols: List[str],
    maturity_map: Dict[str, float],
    lam1: float,
    lam2: float,
) -> pd.DataFrame:
    """
    Per-date OLS fit of Svensson β₀…β₃ (λ₁, λ₂ fixed).

    [F6] Betas are clipped to ±500 after OLS to prevent numerical explosions
    from near-singular design matrices (which occur when λ₁ is too small).
    """
    mats = np.array([maturity_map[c] for c in yc_cols], dtype=float)
    L    = np.array([sv_loadings(t, lam1, lam2) for t in mats])   # (n_tenors, 3)
    X    = np.column_stack([np.ones(len(mats)), L[:, 0], L[:, 1], L[:, 2]])

    # [F6] Condition number check — warn if near-singular
    cond = np.linalg.cond(X)
    if cond > 1e8:
        pass  # will still clip betas below

    rows = []
    for _, row in df_yc[yc_cols].iterrows():
        y    = row.to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 4:
            rows.append([np.nan] * 4)
        else:
            coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            # [F6] Clip to prevent downstream ARIMA explosions
            coef = np.clip(coef, -500.0, 500.0)
            rows.append(coef.tolist())

    return pd.DataFrame(
        rows, index=df_yc.index,
        columns=["sv_b0", "sv_b1", "sv_b2", "sv_b3"],
    )


def reconstruct_yc_from_sv_betas(
    df_betas: pd.DataFrame,
    yc_cols: List[str],
    maturity_map: Dict[str, float],
    lam1: float,
    lam2: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {col: [sv_yield(maturity_map[col],
                        r["sv_b0"], r["sv_b1"], r["sv_b2"], r["sv_b3"],
                        lam1, lam2)
               for _, r in df_betas.iterrows()]
         for col in yc_cols},
        index=df_betas.index,
    )


def grid_search_sv_lambdas(
    df_yc: pd.DataFrame,
    yc_cols: List[str],
    maturity_map: Dict[str, float],
    lam1_grid: np.ndarray,
    lam2_grid: np.ndarray,
    w_on: float = 0.4,
    w_rest: float = 0.6,
) -> Tuple[float, float, float]:
    """
    [F1] Find (λ₁*, λ₂*) minimising weighted-RMSE on training window.

    Constraints enforced:
    ─────────────────────
    • λ₂ > λ₁ + 0.5   (ensures L2, L3 are statistically distinct)
    • λ₁ > 0, λ₂ > 0
    • Condition number of design matrix < 1e7  (near-singular → skip)

    For an ON–2Y curve, economically reasonable ranges are:
      λ₁ ∈ [0.3, 2.5]  (slope hump peaks at τ* = λ₁, within 2Y)
      λ₂ ∈ [2.0, 7.0]  (second hump further out — weakly identified but kept)
    """
    best_lam1, best_lam2, best_rmse = lam1_grid[0], lam2_grid[0], np.inf

    mats = np.array([maturity_map[c] for c in yc_cols], dtype=float)

    for l1, l2 in itertools.product(lam1_grid, lam2_grid):
        # [F1] Enforce separation constraint
        if l2 <= l1 + 0.5 or l1 <= 0 or l2 <= 0:
            continue

        # [F1] Skip near-singular design matrices
        L  = np.array([sv_loadings(t, l1, l2) for t in mats])
        X  = np.column_stack([np.ones(len(mats)), L[:, 0], L[:, 1], L[:, 2]])
        if np.linalg.cond(X) > 1e7:
            continue

        try:
            df_b   = fit_sv_betas_ols(df_yc, yc_cols, maturity_map, l1, l2)
            yc_rec = reconstruct_yc_from_sv_betas(df_b, yc_cols, maturity_map, l1, l2)
            rmse   = compute_weighted_rmse_curve(df_yc, yc_rec, yc_cols, w_on, w_rest)
            if rmse < best_rmse:
                best_rmse, best_lam1, best_lam2 = rmse, l1, l2
        except Exception:
            continue

    return best_lam1, best_lam2, best_rmse


# ── Metric ─────────────────────────────────────────────────────────────────────

def compute_weighted_rmse_curve(
    df_actual: pd.DataFrame,
    df_fc: pd.DataFrame,
    cols: List[str],
    w_on: float = 0.4,
    w_rest_total: float = 0.6,
) -> float:
    cols = [c for c in cols if c in df_actual.columns and c in df_fc.columns]
    if len(cols) < 2 or "ON" not in cols:
        return np.nan
    idx = df_actual.index.intersection(df_fc.index)
    if not len(idx):
        return np.nan
    actual  = df_actual.loc[idx, cols].astype(float)
    pred    = df_fc.loc[idx, cols].astype(float)
    rest    = [c for c in cols if c != "ON"]
    w_each  = w_rest_total / len(rest)
    weights = {"ON": w_on, **{c: w_each for c in rest}}
    wmse = sum(
        weights[c] * float(np.nanmean((actual[c] - pred[c]).values ** 2))
        for c in cols
    )
    return float(np.sqrt(max(wmse, 0.0)))


# ══════════════════════════════════════════════════════════════════════════════
# §2  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT    = Path(__file__).resolve().parent
DATA_DIR        = PROJECT_ROOT / "data"
BETAS_PATH      = DATA_DIR / "ns_results" / "betas_0_7308.csv"
YC_PATH         = DATA_DIR / "inputs"     / "yield_curve.xlsx"
MACRO_PATH      = DATA_DIR / "inputs"     / "macro_updated.xlsx"
IV_PATH         = PROJECT_ROOT            / "Problem_1_IV_train.xlsx"
OUTPUT_DIR      = Path(os.getenv("CODE4_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
SUBMISSION_PATH = Path(
    os.getenv("CODE4_SUBMISSION_PATH",
              str(PROJECT_ROOT / "Problem_1_yield_curve_predict.xlsx"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NS_LAMBDA        = 0.7308
BETA_COLS        = ["beta0", "beta1", "beta2"]
SV_BETA_COLS     = ["sv_b0", "sv_b1", "sv_b2", "sv_b3"]
START_DATE_BETAS = "2019-03-01"
FREQ             = "MS"

YIELD_COLS = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]
MATURITY_MAP = {
    "ON": 1/365, "1W": 7/365,  "2W": 14/365,
    "1M": 1/12,  "2M": 2/12,   "3M": 3/12,
    "6M": 6/12,  "1Y": 1.0,    "2Y": 2.0,
}

MACRO_COLS_CANDIDATE = [
    "cbr", 
    "inf", 
    "observed_inf", 
    "expected_inf",
    "usd", 
    "moex", 
    "brent", 
    "vix",
    # "GDP index", 
    "urals", 
    # "GPR",
]
MACRO_PCA_COMPONENTS = 3

IV_TENOR_MAP = {
    "1M": 1/12, "2M": 2/12, "3M": 3/12, "6M": 6/12,
    "9M": 9/12, "1Y": 1.0,  "2Y": 2.0,  "3Y": 3.0,
    "4Y": 4.0,  "5Y": 5.0,  "6Y": 6.0,  "7Y": 7.0,
    "8Y": 8.0,  "9Y": 9.0,  "10Y": 10.0,
}
IV_PCA_COMPONENTS = 3

# [F1] Economically constrained Svensson grids for ON–2Y curve
# λ₁: slope hump peaks at τ* = λ₁ years (must be within [0.3, 2.5] for 2Y curve)
# λ₂: second hump always at λ₂ >> λ₁ (must exceed λ₁ + 0.5)
SV_LAM1_GRID = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5])
SV_LAM2_GRID = np.array([2.0, 3.0, 4.0, 5.0, 7.0])

TRUE_FUTURE_INDEX = normalize_month_index(
    pd.date_range("2025-10-01", periods=6, freq="MS")
)

TEST_SIZE = 6
VAL_SIZE  = 6

# Optional overrides for cross-validation experiments (used by code5.py)
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

TEST_SIZE = _env_int("CODE4_TEST_SIZE", TEST_SIZE)
VAL_SIZE  = _env_int("CODE4_VAL_SIZE", VAL_SIZE)

# Optional run-mode flags
CODE4_SKIP_OUTPUT = _env_bool("CODE4_SKIP_OUTPUT", False)
CODE4_SKIP_PLOTS  = _env_bool("CODE4_SKIP_PLOTS", False)
CODE4_REPORT_JSON = _env_bool("CODE4_REPORT_JSON", False)
CODE4_FORCE_M1_TAG = os.getenv("CODE4_FORCE_M1_TAG", "").strip()
CODE4_FORCE_M2_TAG = os.getenv("CODE4_FORCE_M2_TAG", "").strip()
P_GRID    = [0, 1, 2, 3]
Q_GRID    = [0, 1, 2]
CLIP_SIGMA = 2.5

# [F2] Hard cap: prevents VAR from becoming underdetermined
# Rule of thumb: n_obs / n_vars >= 10  →  67 / 7 = 9.6 (borderline but ok)
MAX_VAR_TOTAL_VARS = 7
MAX_VAR_LAGS       = 4

W_ARIMA = 0.5
W_VAR   = 0.5

# [F4] Only pca-macro VAR specs (raw macro removed — causes collinearity)
# [F5] iv_n=1 for all IV variants (only iv_pc1 as endogenous, not all 3 IV PCs)
VAR_SPECS: List[Dict] = [
    # ── M1 candidates (no IV) ───────────────────────────────────────────────
    {"tag": "VAR_NS_only",          "beta": "ns", "macro": "none", "iv": False, "iv_n": 0},
    {"tag": "VAR_NS_macro_pca",     "beta": "ns", "macro": "pca",  "iv": False, "iv_n": 0},
    {"tag": "VAR_SV_only",          "beta": "sv", "macro": "none", "iv": False, "iv_n": 0},
    {"tag": "VAR_SV_macro_pca",     "beta": "sv", "macro": "pca",  "iv": False, "iv_n": 0},
    # ── M2 candidates (single IV PC to prevent overparameterisation) ────────
    {"tag": "VAR_NS_macro_pca_iv1", "beta": "ns", "macro": "pca",  "iv": True,  "iv_n": 1},
    {"tag": "VAR_NS_only_iv1",      "beta": "ns", "macro": "none", "iv": True,  "iv_n": 1},
    {"tag": "VAR_SV_macro_pca_iv1", "beta": "sv", "macro": "pca",  "iv": True,  "iv_n": 1},
    {"tag": "VAR_SV_only_iv1",      "beta": "sv", "macro": "none", "iv": True,  "iv_n": 1},
]


# ══════════════════════════════════════════════════════════════════════════════
# §3  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_betas() -> pd.DataFrame:
    df = pd.read_csv(BETAS_PATH)
    df["date"] = pd.date_range(start=START_DATE_BETAS, periods=len(df), freq=FREQ)
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return to_num_df(df)


def load_yield_curve() -> pd.DataFrame:
    df = pd.read_excel(YC_PATH)
    if "Month" not in df.columns:
        raise ValueError("yield_curve.xlsx: 'Month' column missing")
    df["date"] = pd.to_datetime(df["Month"])
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return to_num_df(df)


def load_macro() -> pd.DataFrame:
    df = pd.read_excel(MACRO_PATH)
    date_col = next(
        (c for c in df.columns if str(c).lower() in
         ["date", "dt", "month", "period", "дата"]),
        None,
    )
    if date_col is None:
        raise ValueError("macro_updated.xlsx: no date column")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return to_num_df(df)


def load_iv_raw() -> pd.DataFrame:
    df = pd.read_excel(IV_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Maturity (year fraction)", "Strike", "Volatility"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volatility"])


# ══════════════════════════════════════════════════════════════════════════════
# §4  IV FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _smile_stats(grp: pd.DataFrame) -> Dict[str, float]:
    vols = grp.sort_values("Strike")["Volatility"].values
    if len(vols) < 3:
        return {"rr": np.nan, "bf": np.nan, "smile_slope": np.nan}
    lo, hi  = vols[0], vols[-1]
    mid     = float(np.median(vols))
    strikes = grp.sort_values("Strike")["Strike"].values
    slope   = (0.0 if strikes.std() < 1e-10
               else float(np.polyfit(strikes, vols, 1)[0]))
    return {"rr": hi - lo, "bf": (hi + lo) / 2 - mid, "smile_slope": slope}


def extract_iv_features(df_iv_raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for date, grp_date in df_iv_raw.groupby("Date"):
        row: Dict[str, float] = {}

        atm: Dict[float, float] = {}
        for tau, g in grp_date.groupby("Maturity (year fraction)"):
            k_atm = g["Strike"].median()
            atm[float(tau)] = float(
                g.loc[(g["Strike"] - k_atm).abs().idxmin(), "Volatility"]
            )

        taus_s, vols_s = sorted(atm), [atm[t] for t in sorted(atm)]
        for lbl, tau in IV_TENOR_MAP.items():
            if tau in atm:
                row[f"atm_iv_{lbl}"] = atm[tau]
            elif len(taus_s) >= 2 and taus_s[0] <= tau <= taus_s[-1]:
                row[f"atm_iv_{lbl}"] = float(np.interp(tau, taus_s, vols_s))
            else:
                row[f"atm_iv_{lbl}"] = np.nan

        def _g(lbl): return row.get(f"atm_iv_{lbl}", np.nan)

        atm_vals = [v for k, v in row.items()
                    if k.startswith("atm_iv_") and pd.notna(v)]
        row["ts_level_mean"]  = float(np.nanmean(atm_vals)) if atm_vals else np.nan
        row["ts_slope_short"] = _g("3M") - _g("1M")
        row["ts_slope_long"]  = _g("2Y") - _g("6M")
        row["ts_curvature"]   = 2 * _g("1Y") - _g("3M") - _g("2Y")

        rr_list, bf_list, sl_list = [], [], []
        for _, g in grp_date.groupby("Maturity (year fraction)"):
            sm = _smile_stats(g)
            if pd.notna(sm["rr"]):
                rr_list.append(sm["rr"])
                bf_list.append(sm["bf"])
                sl_list.append(sm["smile_slope"])

        row["mean_rr"]          = float(np.mean(rr_list)) if rr_list else np.nan
        row["mean_bf"]          = float(np.mean(bf_list)) if bf_list else np.nan
        row["mean_smile_slope"] = float(np.mean(sl_list)) if sl_list else np.nan

        vols_all = grp_date["Volatility"].dropna().values
        if len(vols_all):
            row["iv_mean"]   = float(np.mean(vols_all))
            row["iv_median"] = float(np.median(vols_all))
            row["iv_std"]    = float(np.std(vols_all))
            row["iv_p10"]    = float(np.percentile(vols_all, 10))
            row["iv_p90"]    = float(np.percentile(vols_all, 90))
        else:
            row.update(dict.fromkeys(
                ["iv_mean","iv_median","iv_std","iv_p10","iv_p90"], np.nan
            ))

        records.append({"date": date, **row})

    df = pd.DataFrame(records).set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return df


def pca_reduce_iv(
    df_iv_feat: pd.DataFrame,
    n_components: int = IV_PCA_COMPONENTS,
    cols_to_reduce: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, PCA, StandardScaler, List[str]]:
    if cols_to_reduce is None:
        cols_to_reduce = [c for c in df_iv_feat.columns if c.startswith("atm_iv_")]
    available  = [c for c in cols_to_reduce if c in df_iv_feat.columns]
    other_cols = [c for c in df_iv_feat.columns if c not in available]

    X      = df_iv_feat[available].fillna(df_iv_feat[available].median())
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    n_comp = min(n_components, X_sc.shape[1], X_sc.shape[0])
    pca    = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_sc)

    pc_cols = [f"iv_pc{i+1}" for i in range(n_comp)]
    return (
        pd.concat([pd.DataFrame(scores, index=df_iv_feat.index, columns=pc_cols),
                   df_iv_feat[other_cols]], axis=1),
        pca, scaler, available,
    )


def apply_pca_transform(
    df_iv_feat: pd.DataFrame,
    pca: PCA,
    scaler: StandardScaler,
    reduced_cols: List[str],
) -> pd.DataFrame:
    other_cols = [c for c in df_iv_feat.columns if c not in reduced_cols]
    available  = [c for c in reduced_cols if c in df_iv_feat.columns]
    X          = df_iv_feat[available].fillna(df_iv_feat[available].median())
    scores     = pca.transform(scaler.transform(X))
    pc_cols    = [f"iv_pc{i+1}" for i in range(scores.shape[1])]
    return pd.concat([
        pd.DataFrame(scores, index=df_iv_feat.index, columns=pc_cols),
        df_iv_feat[other_cols],
    ], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# §5  MACRO PCA REDUCER
# ══════════════════════════════════════════════════════════════════════════════

class MacroPCAReducer:
    """
    Reduces macro to orthogonal PCs, preventing collinearity in VAR.
    Reference: Otsenko & Seleznev (RJMF 2022), §3.2.
    """

    def __init__(self, n_components: int = MACRO_PCA_COMPONENTS,
                 prefix: str = "macro_pc"):
        self.n_components = n_components
        self.prefix       = prefix
        self.pca_:    Optional[PCA]            = None
        self.scaler_: Optional[StandardScaler] = None
        self.cols_:   Optional[List[str]]       = None

    def fit(self, df: pd.DataFrame) -> "MacroPCAReducer":
        df = df.copy().replace([np.inf,-np.inf], np.nan).ffill().bfill().fillna(0.0)
        self.cols_   = list(df.columns)
        self.scaler_ = StandardScaler()
        X_sc         = self.scaler_.fit_transform(df)
        n            = min(self.n_components, X_sc.shape[1], X_sc.shape[0])
        self.pca_    = PCA(n_components=n)
        self.pca_.fit(X_sc)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        avail = [c for c in self.cols_ if c in df.columns]
        x     = df[avail].copy().replace([np.inf,-np.inf], np.nan).ffill().bfill().fillna(0.0)
        scores = self.pca_.transform(self.scaler_.transform(x))
        return pd.DataFrame(
            scores, index=df.index,
            columns=[f"{self.prefix}{i+1}" for i in range(scores.shape[1])],
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.pca_.explained_variance_ratio_ if self.pca_ else np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# §6  MACRO PROJECTION  (AR-1 forward fill)
# ══════════════════════════════════════════════════════════════════════════════

def project_series_ar1(series: pd.Series, steps: int) -> pd.Series:
    s = series.dropna()
    if len(s) < 4:
        return pd.Series([float(s.iloc[-1]) if len(s) else 0.0] * steps)
    try:
        d   = choose_d_by_adf(s)
        res = ARIMA(s, order=(1, d, 0)).fit()
        return pd.Series(np.asarray(res.forecast(steps=steps)))
    except Exception:
        return pd.Series([float(s.iloc[-1])] * steps)


def project_df_forward(df: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    steps = len(future_index)
    return pd.DataFrame(
        {col: project_series_ar1(df[col], steps).values for col in df.columns},
        index=future_index,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §7  ARIMA / ARIMAX  (NS betas)
# ══════════════════════════════════════════════════════════════════════════════

def _val_mse(y_train, y_val, order):
    try:
        fc = ARIMA(y_train, order=order).fit().forecast(steps=len(y_val))
        return float(np.mean((y_val.values - np.asarray(fc)) ** 2))
    except Exception:
        return np.inf


def grid_search_arima_order(y_train, y_val, p_grid=P_GRID, q_grid=Q_GRID):
    d = choose_d_by_adf(y_train)
    best_order, best_mse = (1, d, 0), np.inf
    for p, q in itertools.product(p_grid, q_grid):
        if p == 0 and q == 0:
            continue
        mse = _val_mse(y_train, y_val, (p, d, q))
        if mse < best_mse:
            best_mse, best_order = mse, (p, d, q)
    return best_order, best_mse


def find_best_arima_orders(df_b_train, df_b_val, beta_cols=BETA_COLS):
    best: Dict[str, tuple] = {}
    print("\n── ARIMA grid-search (NS betas) ──")
    for b in beta_cols:
        y_tr = df_b_train[b].dropna()
        y_va = df_b_val[b].dropna()
        if len(y_tr) < 10 or len(y_va) == 0:
            best[b] = (2, choose_d_by_adf(y_tr), 0); continue
        order, mse = grid_search_arima_order(y_tr, y_va)
        print(f"  {b}: {order}  val_MSE={mse:.5f}")
        best[b] = order
    return best


def clip_forecast(fc: pd.Series, train: pd.Series,
                  n_sigma: float = CLIP_SIGMA) -> pd.Series:
    mu, std = float(train.mean()), float(train.std())
    return fc if std < 1e-8 else fc.clip(mu - n_sigma*std, mu + n_sigma*std)


class ARIMAModel:
    def __init__(self, beta_cols, best_orders=None):
        self.beta_cols   = beta_cols
        self.best_orders = best_orders or {}
        self._m: Dict = {}
        self._tr: Dict = {}

    def fit(self, df_b):
        for b in self.beta_cols:
            y = df_b[b].dropna()
            self._tr[b] = y.copy()
            if len(y) < 10:
                self._m[b] = None; continue
            order = self.best_orders.get(b, (2, choose_d_by_adf(y), 0))
            try:
                self._m[b] = ARIMA(y, order=order).fit()
            except Exception:
                self._m[b] = None
        return self

    def forecast(self, steps, fi):
        out = pd.DataFrame(index=fi, columns=self.beta_cols, dtype=float)
        for b in self.beta_cols:
            tr = self._tr.get(b, pd.Series(dtype=float))
            m  = self._m.get(b)
            lv = float(tr.iloc[-1]) if len(tr) else np.nan
            if m is None:
                out[b] = lv; continue
            try:
                fc = pd.Series(np.asarray(m.forecast(steps)), index=fi)
                out[b] = clip_forecast(fc, tr)
            except Exception:
                out[b] = lv
        return out


class ARIMAXModel:
    def __init__(self, beta_cols, best_orders=None):
        self.beta_cols   = beta_cols
        self.best_orders = best_orders or {}
        self._m:    Dict = {}
        self._tr:   Dict = {}
        self._tX:   Dict = {}
        self._ecols: Dict = {}

    @staticmethod
    def _clean(X):
        X = X.copy().replace([np.inf,-np.inf], np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all").ffill().bfill()
        X = X.loc[:, X.std(ddof=0) > 1e-12]
        return X.fillna(0.0)

    def fit(self, df_b, df_exog):
        X_all = self._clean(df_exog)
        for b in self.beta_cols:
            y      = df_b[b].dropna()
            common = y.index.intersection(X_all.index)
            if len(common) < 10:
                self._m[b] = None; self._tr[b] = y; continue
            y1 = y.loc[common].astype(float)
            X1 = X_all.loc[common].astype(float).dropna(axis=1, how="all")
            X1 = None if X1.shape[1] == 0 else X1
            order = self.best_orders.get(b, (2, choose_d_by_adf(y1), 0))
            try:
                res = (ARIMA(y1, order=order).fit() if X1 is None
                       else ARIMA(y1, exog=X1, order=order).fit())
                self._m[b]     = res
                self._tr[b]    = y1
                self._tX[b]    = X1
                self._ecols[b] = list(X1.columns) if X1 is not None else []
            except Exception:
                self._m[b] = None; self._tr[b] = y
        return self

    def forecast(self, steps, fi, df_exog_future):
        Xf_all = self._clean(df_exog_future)
        out    = pd.DataFrame(index=fi, columns=self.beta_cols, dtype=float)
        for b in self.beta_cols:
            tr  = self._tr.get(b, pd.Series(dtype=float))
            lv  = float(tr.iloc[-1]) if len(tr) else np.nan
            m   = self._m.get(b)
            if m is None:
                out[b] = lv; continue
            try:
                ecols = self._ecols.get(b, [])
                if not ecols:
                    fc = pd.Series(np.asarray(m.forecast(steps)), index=fi)
                else:
                    Xf = Xf_all.reindex(columns=ecols)
                    for c in ecols:
                        if Xf[c].isna().all():
                            tX = self._tX.get(b)
                            Xf[c] = float(tX[c].iloc[-1]) if tX is not None else 0.0
                    Xf = Xf.reindex(fi).ffill().bfill().fillna(0.0)
                    fc = pd.Series(np.asarray(m.forecast(steps, exog=Xf)), index=fi)
                out[b] = clip_forecast(fc, tr)
            except Exception:
                out[b] = lv
        return out


# ══════════════════════════════════════════════════════════════════════════════
# §7b  RANDOM WALK BASELINE
# ══════════════════════════════════════════════════════════════════════════════

class RandomWalkModel:
    """Last observed yield curve repeated for all steps — quality gate."""

    def __init__(self, yc_cols: List[str] = YIELD_COLS):
        self.yc_cols = yc_cols
        self._last:  Optional[pd.Series] = None

    def fit(self, df_yc: pd.DataFrame) -> "RandomWalkModel":
        avail      = [c for c in self.yc_cols if c in df_yc.columns]
        self._last = df_yc[avail].iloc[-1].astype(float)
        return self

    def forecast(self, steps: int, fi: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(self._last.values, (steps, 1)),
            index=fi, columns=self._last.index.tolist(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# §7c  FLEXIBLE VAR MODEL  [F2, F4, F5]
# ══════════════════════════════════════════════════════════════════════════════

def _stationarize(df, prefer_growth=False):
    out, info = pd.DataFrame(index=df.index), {}
    for c in df.columns:
        s = (pd.to_numeric(df[c], errors="coerce")
             .replace([np.inf,-np.inf], np.nan).ffill().bfill())
        if s.isna().all():
            out[c], info[c] = s, "level"; continue
        if choose_d_by_adf(s) == 0:
            out[c], info[c] = s, "level"; continue
        if prefer_growth:
            g = s.pct_change().replace([np.inf,-np.inf], np.nan).ffill().bfill()
            if g.notna().sum() >= 15 and choose_d_by_adf(g) == 0:
                out[c], info[c] = g, "growth"; continue
        d = s.diff().replace([np.inf,-np.inf], np.nan).ffill().bfill()
        if d.notna().sum() >= 15 and choose_d_by_adf(d) == 0:
            out[c], info[c] = d, "diff"; continue
        out[c], info[c] = s, "level"
    return out.replace([np.inf,-np.inf], np.nan).ffill().bfill(), info


def _invert_forecast(last, fc_stat, info):
    out  = pd.DataFrame(index=fc_stat.index, columns=fc_stat.columns, dtype=float)
    prev = last.copy().astype(float)
    for dt in fc_stat.index:
        for c in fc_stat.columns:
            v = float(fc_stat.loc[dt, c])
            t = info.get(c, "level")
            if pd.isna(v):
                out.loc[dt, c] = np.nan; continue
            if t == "level":
                out.loc[dt, c] = v; prev[c] = v
            elif t == "diff":
                b = prev.get(c, np.nan)
                out.loc[dt, c] = (b + v) if pd.notna(b) else np.nan
                prev[c] = out.loc[dt, c]
            elif t == "growth":
                b = prev.get(c, np.nan)
                out.loc[dt, c] = (b * (1+v)) if pd.notna(b) else np.nan
                prev[c] = out.loc[dt, c]
            else:
                out.loc[dt, c] = v; prev[c] = v
    return out


def _fit_var_model(df_stat, maxlags=MAX_VAR_LAGS):
    df_stat = df_stat.replace([np.inf,-np.inf], np.nan).ffill().bfill().dropna()
    if len(df_stat) < 15 or df_stat.shape[1] < 2:
        return None
    # [F2] obs-per-param sanity: need at least 8 obs per variable per lag
    ml = min(maxlags, max(1, len(df_stat) // (8 * df_stat.shape[1])),
             max(1, len(df_stat) // 4))
    ml = max(1, ml)
    try:
        model = VAR(df_stat)
        sel   = model.select_order(maxlags=ml)
        p     = max(1, int(sel.selected_orders.get("aic", 1) or 1))
        return model.fit(p)
    except Exception:
        return None


class FlexVARModel:
    """
    [F2, F4, F5] Flexible VAR covering all variant specifications.

    Key fixes vs v2:
    ─────────────────
    F2  MAX_VAR_TOTAL_VARS cap enforced: excess columns trimmed in priority order
    (betas kept → IV PCs kept → macro PCs trimmed first when needed)
    F4  Raw macro removed from VAR_SPECS entirely
    F5  iv_n parameter: number of IV PCs to include (default 1, was 3)
        VAR_NS_macro_pca_iv1 = NS(3) + macro_pca(3) + iv_pc1(1) = 7 vars  ✓
        Old VAR_SV_macro_raw_iv = SV(4)+macro_raw(8)+IV_PCA(3) = 15 vars  ✗
    """

    def __init__(self, spec: Dict,
                 yc_cols: List[str]         = YIELD_COLS,
                 maturity_map: Dict         = MATURITY_MAP,
                 ns_lambda: float           = NS_LAMBDA):
        self.spec         = spec
        self.yc_cols      = yc_cols
        self.maturity_map = maturity_map
        self.ns_lambda    = ns_lambda
        self.tag          = spec["tag"]
        self._res         = None
        self._stat_df:   pd.DataFrame    = pd.DataFrame()
        self._stat_info: Dict[str, str]  = {}
        self._last:      pd.Series       = pd.Series(dtype=float)
        self._beta_cols: List[str]       = []
        self._macro_cols: List[str]      = []
        self._iv_cols:    List[str]      = []
        self._sv_lam1:   float           = 0.7
        self._sv_lam2:   float           = 3.0

    def build(
        self,
        df_ns_betas:  pd.DataFrame,
        df_sv_betas:  pd.DataFrame,
        df_macro_pca: pd.DataFrame,
        df_iv_pca:    pd.DataFrame,
        sv_lam1:      float,
        sv_lam2:      float,
    ) -> "FlexVARModel":
        self._sv_lam1 = sv_lam1
        self._sv_lam2 = sv_lam2
        spec  = self.spec
        iv_n  = int(spec.get("iv_n", 1))  # [F5]

        # ── Beta block ────────────────────────────────────────────────────────
        if spec["beta"] == "sv":
            df_b = df_sv_betas[SV_BETA_COLS].copy()
            self._beta_cols = SV_BETA_COLS
        else:
            df_b = df_ns_betas[BETA_COLS].copy()
            self._beta_cols = BETA_COLS

        # ── Macro block ───────────────────────────────────────────────────────
        # [F4] Only PCA macro is available; "none" skips macro entirely
        blocks = [df_b]
        macro_cols: List[str] = []
        if spec["macro"] == "pca":
            macro_cols = list(df_macro_pca.columns)
            blocks.append(df_macro_pca)
        # (raw macro option removed — see F4 rationale)

        # ── IV block [F5] ─────────────────────────────────────────────────────
        iv_cols: List[str] = []
        if spec["iv"] and iv_n > 0:
            iv_cols = [f"iv_pc{i+1}" for i in range(
                min(iv_n, df_iv_pca.shape[1])
            )]
            iv_cols = [c for c in iv_cols if c in df_iv_pca.columns]
            if iv_cols:
                blocks.append(df_iv_pca[iv_cols])
        self._macro_cols = macro_cols
        self._iv_cols    = iv_cols

        full = pd.concat(blocks, axis=1).replace([np.inf,-np.inf], np.nan).ffill().bfill()

        # ── [F2] Hard variable cap ────────────────────────────────────────────
        if full.shape[1] > MAX_VAR_TOTAL_VARS:
            beta_c  = self._beta_cols
            macro_c = [c for c in self._macro_cols if c in full.columns]
            iv_c    = [c for c in self._iv_cols if c in full.columns]
            other_c = [c for c in full.columns if c not in beta_c + macro_c + iv_c]

            keep_other_budget = MAX_VAR_TOTAL_VARS - len(beta_c)
            keep_others: List[str] = []
            # Preserve IV cols first; trim macro if budget is not enough.
            if keep_other_budget > 0:
                keep_others.extend(iv_c[:keep_other_budget])
            if len(keep_others) < keep_other_budget:
                keep_others.extend(macro_c[:max(0, keep_other_budget - len(keep_others))])
            if len(keep_others) < keep_other_budget:
                keep_others.extend(other_c[:max(0, keep_other_budget - len(keep_others))])

            full    = full[beta_c + keep_others]
            print(f"  [F2] {self.tag}: trimmed to {full.shape[1]} vars "
                  f"(cap={MAX_VAR_TOTAL_VARS})")

        self._last = full.iloc[-1].copy()

        # ── Stationarise ──────────────────────────────────────────────────────
        stat, info = _stationarize(full, prefer_growth=False)
        self._stat_df   = stat.replace([np.inf,-np.inf], np.nan).ffill().bfill()
        self._stat_info = {k: v for k, v in info.items()
                           if k in self._stat_df.columns}
        self._res = _fit_var_model(self._stat_df)

        if self._res is None:
            print(f"  [WARN] {self.tag}: VAR fit failed — "
                  f"falling back to last-value forecast")
        return self

    def forecast(self, steps: int, fi: pd.DatetimeIndex) -> pd.DataFrame:
        """Forecast betas → reconstruct yield curve."""
        fallback = pd.DataFrame(
            np.tile(self._last[self._beta_cols].values, (steps, 1)),
            index=fi, columns=self._beta_cols,
        )

        if self._res is not None:
            try:
                init    = self._stat_df.values[-self._res.k_ar:]
                fc_stat = pd.DataFrame(
                    self._res.forecast(init, steps=steps),
                    index=fi, columns=self._stat_df.columns,
                )
                fc_lv   = _invert_forecast(self._last, fc_stat, self._stat_info)
                beta_fc = fc_lv[self._beta_cols].ffill().bfill()
                for b in self._beta_cols:
                    if beta_fc[b].isna().any():
                        beta_fc[b] = beta_fc[b].fillna(
                            float(self._last.get(b, 0.0))
                        )
            except Exception:
                beta_fc = fallback
        else:
            beta_fc = fallback

        # ── Reconstruct YC ────────────────────────────────────────────────────
        if self.spec["beta"] == "sv":
            return reconstruct_yc_from_sv_betas(
                beta_fc, self.yc_cols, self.maturity_map,
                self._sv_lam1, self._sv_lam2,
            )
        else:
            return reconstruct_yc_from_ns_betas(
                beta_fc,
                {k: self.maturity_map[k] for k in self.yc_cols},
                lam=self.ns_lambda,
            )


# ══════════════════════════════════════════════════════════════════════════════
# §7d  SVENSSON ARIMA FORECASTER  [F1, F3, F6]
# ══════════════════════════════════════════════════════════════════════════════

class SvenssonARIMAForecaster:
    """
    Dynamic Svensson model (Otsenko & Seleznev RJMF 2022, §3–4).

    Fixes vs v2:
    ─────────────
    F1  Lambda grid now constrained: λ₁∈[0.3,2.5], λ₂∈[2.0,7.0], λ₂>λ₁+0.5
        Eliminates near-singular OLS matrices (old λ₁=0.1 caused explosion).
    F3  Two-phase fit to prevent leakage:
        Phase 1 .fit(train, val) → find λ* and ARIMA orders
        Phase 2 .refit_arima_only(trainval) → refit ARIMA betas on full data
    F6  OLS betas clipped to ±500 inside fit_sv_betas_ols (see §1).
    """

    def __init__(
        self,
        yc_cols:     List[str]           = YIELD_COLS,
        maturity_map: Dict[str, float]   = MATURITY_MAP,
        lam1_grid:   np.ndarray          = SV_LAM1_GRID,
        lam2_grid:   np.ndarray          = SV_LAM2_GRID,
        best_orders: Optional[Dict]      = None,
    ):
        self.yc_cols      = yc_cols
        self.maturity_map = maturity_map
        self.lam1_grid    = lam1_grid
        self.lam2_grid    = lam2_grid
        self.best_orders  = dict(best_orders) if best_orders else {}

        self.lam1_opt: float         = lam1_grid[2]   # default 0.7
        self.lam2_opt: float         = lam2_grid[1]   # default 3.0
        self._betas:   pd.DataFrame  = pd.DataFrame()
        self._arima:   Dict          = {}
        self._trains:  Dict          = {}

    # ── Phase 1: find λ* and ARIMA orders ────────────────────────────────────

    def fit(
        self,
        df_yc_train: pd.DataFrame,
        df_yc_val:   Optional[pd.DataFrame] = None,
        verbose:     bool = True,
    ) -> "SvenssonARIMAForecaster":
        """
        [F3] TRAIN PHASE ONLY.
        Pass df_yc_train = train slice, df_yc_val = val slice.
        Do NOT pass trainval as df_yc_train when val is the evaluation set.
        """
        # 1. Lambda grid search on TRAIN only
        self.lam1_opt, self.lam2_opt, in_rmse = grid_search_sv_lambdas(
            df_yc_train, self.yc_cols, self.maturity_map,
            self.lam1_grid, self.lam2_grid,
        )
        if verbose:
            print(f"  [SV] λ₁*={self.lam1_opt:.2f}  λ₂*={self.lam2_opt:.2f}"
                  f"  train-RMSE={in_rmse:.5f}")

        # 2. OLS betas on TRAIN only
        self._betas = fit_sv_betas_ols(
            df_yc_train, self.yc_cols, self.maturity_map,
            self.lam1_opt, self.lam2_opt,
        )

        # 3. ARIMA order search (train→val, no leakage) [F3]
        if df_yc_val is not None:
            sv_betas_val = fit_sv_betas_ols(
                df_yc_val, self.yc_cols, self.maturity_map,
                self.lam1_opt, self.lam2_opt,
            )
            if verbose:
                print("  [SV] ARIMA order grid-search …")
            for b in SV_BETA_COLS:
                y_tr = self._betas[b].dropna()
                y_va = sv_betas_val[b].dropna()
                if len(y_tr) < 10 or len(y_va) == 0:
                    self.best_orders[b] = (1, choose_d_by_adf(y_tr), 0)
                    continue
                order, mse = grid_search_arima_order(y_tr, y_va)
                if verbose:
                    print(f"    {b}: {order}  val_MSE={mse:.5f}")
                self.best_orders[b] = order

        # 4. Fit ARIMA on TRAIN betas
        self._fit_arima_on_current_betas()
        return self

    # ── Phase 2: refit ARIMA on larger dataset (same λ*, same orders) ────────

    def refit_arima_only(self, df_yc_new: pd.DataFrame) -> "SvenssonARIMAForecaster":
        """
        [F3] Re-estimate OLS betas on df_yc_new (e.g. trainval) then refit
        ARIMA using the orders found in Phase 1.  Lambda is NOT re-estimated.
        """
        self._betas = fit_sv_betas_ols(
            df_yc_new, self.yc_cols, self.maturity_map,
            self.lam1_opt, self.lam2_opt,
        )
        self._fit_arima_on_current_betas()
        return self

    def _fit_arima_on_current_betas(self):
        for b in SV_BETA_COLS:
            y = self._betas[b].dropna()
            self._trains[b] = y.copy()
            if len(y) < 10:
                self._arima[b] = None; continue
            order = self.best_orders.get(b, (1, choose_d_by_adf(y), 0))
            try:
                self._arima[b] = ARIMA(y, order=order).fit()
            except Exception:
                self._arima[b] = None

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast(self, steps: int, fi: pd.DatetimeIndex) -> pd.DataFrame:
        fc_b = pd.DataFrame(index=fi, columns=SV_BETA_COLS, dtype=float)
        for b in SV_BETA_COLS:
            m  = self._arima.get(b)
            tr = self._trains.get(b, pd.Series(dtype=float))
            lv = float(tr.iloc[-1]) if len(tr) else np.nan
            if m is None:
                fc_b[b] = lv; continue
            try:
                fc = pd.Series(np.asarray(m.forecast(steps)), index=fi)
                fc_b[b] = clip_forecast(fc, tr)
            except Exception:
                fc_b[b] = lv

        return reconstruct_yc_from_sv_betas(
            fc_b, self.yc_cols, self.maturity_map,
            self.lam1_opt, self.lam2_opt,
        )

    def insample_rmse(self, df_yc: pd.DataFrame) -> float:
        yc_rec = reconstruct_yc_from_sv_betas(
            self._betas, self.yc_cols, self.maturity_map,
            self.lam1_opt, self.lam2_opt,
        )
        return compute_weighted_rmse_curve(df_yc, yc_rec, self.yc_cols)


# ══════════════════════════════════════════════════════════════════════════════
# §8  YIELD-CURVE RECONSTRUCTION  (NS wrapper)
# ══════════════════════════════════════════════════════════════════════════════

def betas_to_yc(df_betas: pd.DataFrame) -> pd.DataFrame:
    return reconstruct_yc_from_ns_betas(
        df_betas,
        {k: MATURITY_MAP[k] for k in YIELD_COLS},
        lam=NS_LAMBDA,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §9  ENSEMBLE  (ARIMA + VAR on NS betas)
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_betas(fc_arima, fc_var, w_arima=W_ARIMA, w_var=W_VAR):
    out = pd.DataFrame(index=fc_arima.index, columns=BETA_COLS, dtype=float)
    for b in BETA_COLS:
        a, v   = fc_arima[b].astype(float), fc_var[b].astype(float)
        a_nan, v_nan = a.isna(), v.isna()
        both   = ~a_nan & ~v_nan
        out.loc[both, b]           = w_arima * a[both] + w_var * v[both]
        out.loc[~both & ~a_nan, b] = a[~both & ~a_nan]
        out.loc[~both & ~v_nan, b] = v[~both & ~v_nan]
        out[b] = out[b].fillna(0.0)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# §10  NaN SAFETY + SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

def nan_safe_yc(df_fc, df_hist, cols=YIELD_COLS):
    out = df_fc[cols].copy().ffill().bfill()
    for c in cols:
        if out[c].isna().any() and c in df_hist.columns:
            out[c] = out[c].fillna(df_hist[c].dropna().iloc[-1])
        out[c] = out[c].fillna(0.0)
    assert not out.isna().any().any()
    return out


def write_submission(df_m1, df_m2, path=SUBMISSION_PATH, cols=YIELD_COLS):
    def _prep(df):
        o = df[cols].copy().reset_index()
        o.rename(columns={"index": "Date"}, inplace=True)
        o["Date"] = o["Date"].dt.strftime("%Y-%m")
        return o
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        _prep(df_m1).to_excel(w, sheet_name="M1", index=False)
        _prep(df_m2).to_excel(w, sheet_name="M2", index=False)
    print(f"\n✓ Submission → {path}")
    print(f"  NaN M1={int(df_m1.isna().sum().sum())}  "
          f"NaN M2={int(df_m2.isna().sum().sum())}")


# ══════════════════════════════════════════════════════════════════════════════
# §11  LOAD & ALIGN DATA
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("Loading data …")

df_betas  = load_betas()
df_yc     = load_yield_curve()
df_macro  = load_macro()
df_iv_raw = load_iv_raw()

beta_cols  = [c for c in BETA_COLS            if c in df_betas.columns]
macro_cols = [c for c in MACRO_COLS_CANDIDATE if c in df_macro.columns]
yc_cols    = [c for c in YIELD_COLS           if c in df_yc.columns]

print("Extracting IV features …")
df_iv_feat = extract_iv_features(df_iv_raw)

common_idx = (
    df_betas.index .intersection(df_yc.index)
    .intersection(df_macro.index) .intersection(df_iv_feat.index)
    .sort_values()
)

df_betas   = df_betas.loc[common_idx, beta_cols].replace([np.inf,-np.inf], np.nan).dropna()
df_yc      = df_yc.loc[common_idx, yc_cols].replace([np.inf,-np.inf], np.nan)
df_macro   = df_macro.loc[common_idx, macro_cols].replace([np.inf,-np.inf], np.nan).ffill().bfill()
df_iv_feat = df_iv_feat.loc[common_idx].replace([np.inf,-np.inf], np.nan).ffill().bfill()
common_idx = df_betas.index

analysis_start = os.getenv("CODE4_ANALYSIS_START")
analysis_end = os.getenv("CODE4_ANALYSIS_END")
if analysis_start:
    start_dt = normalize_month_index(pd.to_datetime(analysis_start))
    common_idx = common_idx[common_idx >= start_dt]
if analysis_end:
    end_dt = normalize_month_index(pd.to_datetime(analysis_end))
    common_idx = common_idx[common_idx <= end_dt]

if len(common_idx) < (VAL_SIZE + TEST_SIZE + 1):
    raise ValueError(
        f"Not enough data for splits after analysis window filter: "
        f"{len(common_idx)} < {VAL_SIZE + TEST_SIZE + 1}"
    )

print(f"  Aligned: {common_idx[0]:%Y-%m} → {common_idx[-1]:%Y-%m}"
      f"  ({len(common_idx)} months)")


# ══════════════════════════════════════════════════════════════════════════════
# §12  SPLITS
# ══════════════════════════════════════════════════════════════════════════════

train_idx    = common_idx[:-(VAL_SIZE + TEST_SIZE)]
val_idx      = common_idx[-(VAL_SIZE + TEST_SIZE):-TEST_SIZE]
trainval_idx = common_idx[:-TEST_SIZE]
test_idx     = common_idx[-TEST_SIZE:]

print(f"\n  Splits → train:{len(train_idx)}  val:{len(val_idx)}"
      f"  trainval:{len(trainval_idx)}  test:{len(test_idx)}")


# ══════════════════════════════════════════════════════════════════════════════
# §13  PCA  —  IV (train only) + MACRO (train only)
# ══════════════════════════════════════════════════════════════════════════════

atm_cols = [c for c in df_iv_feat.columns if c.startswith("atm_iv_")]

(df_iv_tr_r, iv_pca, iv_sc, iv_pca_cols) = pca_reduce_iv(
    df_iv_feat.loc[train_idx], IV_PCA_COMPONENTS, atm_cols
)
_riv = lambda idx: apply_pca_transform(
    df_iv_feat.loc[idx], iv_pca, iv_sc, iv_pca_cols
)
df_iv_val_r  = _riv(val_idx)
df_iv_test_r = _riv(test_idx)
df_iv_tv_r   = _riv(trainval_idx)
df_iv_all_r  = _riv(common_idx)

macro_reducer = MacroPCAReducer(MACRO_PCA_COMPONENTS)
df_macro_tr_pca   = macro_reducer.fit_transform(df_macro.loc[train_idx])
df_macro_tv_pca   = macro_reducer.transform(df_macro.loc[trainval_idx])
df_macro_test_pca = macro_reducer.transform(df_macro.loc[test_idx])
df_macro_all_pca  = macro_reducer.transform(df_macro.loc[common_idx])

print(f"\n  IV PCA  : {IV_PCA_COMPONENTS} PCs "
      f"[{', '.join(f'{v:.1%}' for v in iv_pca.explained_variance_ratio_)}]")
print(f"  Macro PCA: {MACRO_PCA_COMPONENTS} PCs "
      f"[{', '.join(f'{v:.1%}' for v in macro_reducer.explained_variance_ratio_)}]")


# ══════════════════════════════════════════════════════════════════════════════
# §14  NS ARIMA ORDERS
# ══════════════════════════════════════════════════════════════════════════════

best_orders_ns = find_best_arima_orders(
    df_betas.loc[train_idx], df_betas.loc[val_idx], BETA_COLS
)


# ══════════════════════════════════════════════════════════════════════════════
# §15  SVENSSON LAMBDAS  [F1, F3]
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Svensson lambda grid-search ──")

# TRAIN-only lambdas (for back-test phase 1 and SV_ARIMA)
sv_lam1_tr, sv_lam2_tr, sv_rmse_tr = grid_search_sv_lambdas(
    df_yc.loc[train_idx], yc_cols, MATURITY_MAP,
    SV_LAM1_GRID, SV_LAM2_GRID,
)
print(f"  Train    λ₁*={sv_lam1_tr:.2f}  λ₂*={sv_lam2_tr:.2f}"
      f"  RMSE={sv_rmse_tr:.5f}")

# TRAINVAL lambdas (for final forecast)
sv_lam1_tv, sv_lam2_tv, sv_rmse_tv = grid_search_sv_lambdas(
    df_yc.loc[trainval_idx], yc_cols, MATURITY_MAP,
    SV_LAM1_GRID, SV_LAM2_GRID,
)
print(f"  Trainval λ₁*={sv_lam1_tv:.2f}  λ₂*={sv_lam2_tv:.2f}"
      f"  RMSE={sv_rmse_tv:.5f}")

# Svensson beta series per slice (used by FlexVARModel)
sv_betas_trainval = fit_sv_betas_ols(
    df_yc.loc[trainval_idx], yc_cols, MATURITY_MAP, sv_lam1_tv, sv_lam2_tv
)
sv_betas_all = fit_sv_betas_ols(
    df_yc.loc[common_idx], yc_cols, MATURITY_MAP, sv_lam1_tv, sv_lam2_tv
)


# ══════════════════════════════════════════════════════════════════════════════
# §16  RANDOM WALK BASELINE
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Random Walk baseline ──")
rw_model  = RandomWalkModel(yc_cols).fit(df_yc.loc[trainval_idx])
yc_rw_bt  = rw_model.forecast(TEST_SIZE, test_idx)
rmse_rw   = compute_weighted_rmse_curve(df_yc.loc[test_idx], yc_rw_bt, yc_cols)
print(f"  RW RMSE = {rmse_rw:.5f}  ← must beat this")


def _vs_rw(r: float) -> str:
    if np.isnan(r):
        return "N/A"
    gain = (rmse_rw - r) / rmse_rw
    return ("✓" if r < rmse_rw else "✗") + f" ({gain:+.1%} vs RW)"


# ══════════════════════════════════════════════════════════════════════════════
# §17  TOURNAMENT BACK-TEST  (trainval → test)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§17  Tournament back-test …")

results_m1: Dict[str, float]       = {}
results_m2: Dict[str, float]       = {}
yc_forecasts: Dict[str, pd.DataFrame] = {}

yc_actual_bt = df_yc.loc[test_idx]

b_tv  = df_betas.loc[trainval_idx]
m_tv  = df_macro.loc[trainval_idx]
iv_tv = df_iv_tv_r

# Project future exog for ARIMAX
m_test_proj  = project_df_forward(m_tv, test_idx)
iv_test_proj = project_df_forward(iv_tv, test_idx)

m_tv_pca   = df_macro_tv_pca
m_test_pca = df_macro_test_pca


def _record(tag, yc_fc, is_m2=False):
    r = compute_weighted_rmse_curve(yc_actual_bt, yc_fc, yc_cols)
    yc_forecasts[tag] = yc_fc
    (results_m2 if is_m2 else results_m1)[tag] = r
    cat = "M2" if is_m2 else "M1"
    print(f"  {tag:<32}  {cat}: {r:.5f}  {_vs_rw(r)}")
    return r


# ── ARIMA M1 ──────────────────────────────────────────────────────────────────
arima_m1   = ARIMAModel(beta_cols, best_orders_ns).fit(b_tv)
yc_a_m1_bt = betas_to_yc(arima_m1.forecast(TEST_SIZE, test_idx))
_record("ARIMA_NS", yc_a_m1_bt, is_m2=False)

# ── ARIMAX M2 (macro_pca + iv_pc1) [F5] ──────────────────────────────────────
# [F5] Only iv_pc1 as exogenous — prevents overparameterisation in ARIMAX
exog_tv_m2_pca  = pd.concat(
    [m_tv_pca, iv_tv[["iv_pc1"]]], axis=1
) if "iv_pc1" in iv_tv.columns else m_tv_pca

exog_test_m2_pca = pd.concat([
    m_test_pca,
    project_df_forward(iv_tv[["iv_pc1"]], test_idx),
], axis=1) if "iv_pc1" in iv_tv.columns else m_test_pca

arimax_m2_pca   = ARIMAXModel(beta_cols, best_orders_ns).fit(b_tv, exog_tv_m2_pca)
yc_ax_m2_bt     = betas_to_yc(
    arimax_m2_pca.forecast(TEST_SIZE, test_idx, exog_test_m2_pca)
)
_record("ARIMAX_NS_macro_pca_iv1", yc_ax_m2_bt, is_m2=True)

# ── Svensson ARIMA  [F3] two-phase ───────────────────────────────────────────
print("\n  [F3] SV_ARIMA two-phase fit …")
# Phase 1: train-only lambda + ARIMA orders
sv_arima_p1 = SvenssonARIMAForecaster(
    yc_cols, MATURITY_MAP, SV_LAM1_GRID, SV_LAM2_GRID
)
sv_arima_p1.fit(
    df_yc_train=df_yc.loc[train_idx],
    df_yc_val=df_yc.loc[val_idx],
    verbose=True,
)
# Phase 2: refit ARIMA on trainval, same λ* and orders [F3]
sv_arima_bt = SvenssonARIMAForecaster(
    yc_cols, MATURITY_MAP, SV_LAM1_GRID, SV_LAM2_GRID,
    best_orders=sv_arima_p1.best_orders,
)
sv_arima_bt.lam1_opt = sv_arima_p1.lam1_opt
sv_arima_bt.lam2_opt = sv_arima_p1.lam2_opt
sv_arima_bt.refit_arima_only(df_yc.loc[trainval_idx])

yc_sv_arima_bt = sv_arima_bt.forecast(TEST_SIZE, test_idx)
_record("SV_ARIMA", yc_sv_arima_bt, is_m2=False)

# ── All VAR variants ──────────────────────────────────────────────────────────
print("\n  VAR variants:")
for spec in VAR_SPECS:
    tag    = spec["tag"]
    is_m2  = bool(spec.get("iv", False))
    try:
        vm = FlexVARModel(spec, yc_cols, MATURITY_MAP, NS_LAMBDA)
        vm.build(
            df_ns_betas  = b_tv,
            df_sv_betas  = sv_betas_trainval,
            df_macro_pca = m_tv_pca,
            df_iv_pca    = iv_tv,
            sv_lam1      = sv_lam1_tv,
            sv_lam2      = sv_lam2_tv,
        )
        yc_fc = vm.forecast(TEST_SIZE, test_idx)
    except Exception as e:
        print(f"    {tag}: ERROR {e}")
        yc_fc = yc_rw_bt.copy()
    _record(tag, yc_fc, is_m2=is_m2)


# ══════════════════════════════════════════════════════════════════════════════
# §18  BEST MODEL SELECTION  [F8]
# ══════════════════════════════════════════════════════════════════════════════

def select_best(
    results: Dict[str, float],
    rmse_rw: float,
    fallback_tag: str,
    force_tag: str | None = None,
    cat: str = "M1",
) -> Tuple[str, float]:
    """
    [F8] Select best model among those beating RW.
    If NO model beats RW, use the fallback (best M1 for M2, or RW itself).
    """
    if force_tag:
        if force_tag == "RW":
            return force_tag, rmse_rw
        if force_tag in results and not np.isnan(results[force_tag]):
            return force_tag, results[force_tag]
        print(
            f"  [F8] Forced {cat} model '{force_tag}' unavailable or NaN "
            f"→ fallback '{fallback_tag}'"
        )
    beats = {t: r for t, r in results.items()
             if not np.isnan(r) and r < rmse_rw}
    if beats:
        tag  = min(beats, key=beats.get)
        return tag, beats[tag]
    else:
        # [F8] No model beats RW — use fallback
        fb_r = results.get(fallback_tag, rmse_rw)
        print(f"  [F8] No model beats RW → using fallback '{fallback_tag}'"
              f"  RMSE={fb_r:.5f}")
        return fallback_tag, fb_r


# M1 fallback = random walk (if even the best M1 can't beat RW)
best_m1_tag, rmse_m1_bt = select_best(
    results_m1,
    rmse_rw,
    "RW",
    force_tag=CODE4_FORCE_M1_TAG or None,
    cat="M1",
)
# M2 fallback = best M1 model (if no M2 beats RW, submit best M1 for both)
best_m2_tag, rmse_m2_bt = select_best(
    results_m2,
    rmse_rw,
    best_m1_tag,
    force_tag=CODE4_FORCE_M2_TAG or None,
    cat="M2",
)

# Store RW forecast so it can be used as fallback
yc_forecasts["RW"] = yc_rw_bt

rmse_tot_bt = 0.5 * rmse_m1_bt + 0.5 * rmse_m2_bt

print("\n" + "─" * 70)
print(f"  Best M1: {best_m1_tag:<32}  RMSE={rmse_m1_bt:.5f}  {_vs_rw(rmse_m1_bt)}")
print(f"  Best M2: {best_m2_tag:<32}  RMSE={rmse_m2_bt:.5f}  {_vs_rw(rmse_m2_bt)}")
print(f"  RMSEtotal (back-test): {rmse_tot_bt:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
# §19  FINAL FORECAST  2025-10 → 2026-03
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("§19  Final forecast: 2025-10 → 2026-03")

# Re-fit PCA on all data (maximum information)
(df_iv_all_rf, iv_pca_f, iv_sc_f, iv_pca_cols_f) = pca_reduce_iv(
    df_iv_feat.loc[common_idx], IV_PCA_COMPONENTS, atm_cols
)
macro_reducer_f  = MacroPCAReducer(MACRO_PCA_COMPONENTS)
df_macro_all_pca_f = macro_reducer_f.fit_transform(df_macro.loc[common_idx])

# Project macro + IV to true future
m_fut      = project_df_forward(df_macro, TRUE_FUTURE_INDEX)
m_fut_pca  = macro_reducer_f.transform(m_fut)
iv_fut     = project_df_forward(df_iv_all_rf, TRUE_FUTURE_INDEX)


def _build_final_var(tag: str, spec: Dict) -> pd.DataFrame:
    vm = FlexVARModel(spec, yc_cols, MATURITY_MAP, NS_LAMBDA)
    vm.build(
        df_ns_betas  = df_betas,
        df_sv_betas  = sv_betas_all,
        df_macro_pca = df_macro_all_pca_f,
        df_iv_pca    = df_iv_all_rf,
        sv_lam1      = sv_lam1_tv,
        sv_lam2      = sv_lam2_tv,
    )
    return vm.forecast(6, TRUE_FUTURE_INDEX)


def _final_yc(tag: str) -> pd.DataFrame:
    if tag == "RW":
        return RandomWalkModel(yc_cols).fit(df_yc).forecast(6, TRUE_FUTURE_INDEX)

    if tag == "ARIMA_NS":
        m = ARIMAModel(beta_cols, best_orders_ns).fit(df_betas)
        return betas_to_yc(m.forecast(6, TRUE_FUTURE_INDEX))

    if tag == "ARIMAX_NS_macro_pca_iv1":
        exog_all = pd.concat(
            [df_macro_all_pca_f,
             df_iv_all_rf[["iv_pc1"]]], axis=1
        ) if "iv_pc1" in df_iv_all_rf.columns else df_macro_all_pca_f
        exog_fut = pd.concat(
            [m_fut_pca, iv_fut[["iv_pc1"]]], axis=1
        ) if "iv_pc1" in iv_fut.columns else m_fut_pca
        m = ARIMAXModel(beta_cols, best_orders_ns).fit(df_betas, exog_all)
        return betas_to_yc(m.forecast(6, TRUE_FUTURE_INDEX, exog_fut))

    if tag == "SV_ARIMA":
        # [F3] Two-phase: train→orders then refit on all data
        sv_f_p1 = SvenssonARIMAForecaster(
            yc_cols, MATURITY_MAP, SV_LAM1_GRID, SV_LAM2_GRID
        )
        sv_f_p1.fit(df_yc_train=df_yc.loc[trainval_idx],
                    df_yc_val=None, verbose=False)
        sv_f_p1.refit_arima_only(df_yc.loc[common_idx])
        return sv_f_p1.forecast(6, TRUE_FUTURE_INDEX)

    # VAR variants
    spec = next((s for s in VAR_SPECS if s["tag"] == tag), None)
    if spec:
        return _build_final_var(tag, spec)

    raise ValueError(f"Unknown tag: {tag}")


yc_m1_final = nan_safe_yc(_final_yc(best_m1_tag), df_yc, yc_cols)
yc_m2_final = nan_safe_yc(_final_yc(best_m2_tag), df_yc, yc_cols)

print(f"\n  M1 [{best_m1_tag}]:")
print(yc_m1_final.to_string())
print(f"\n  M2 [{best_m2_tag}]:")
print(yc_m2_final.to_string())


# ══════════════════════════════════════════════════════════════════════════════
# §20  WRITE SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════

if not CODE4_SKIP_OUTPUT:
    write_submission(yc_m1_final, yc_m2_final)


# ══════════════════════════════════════════════════════════════════════════════
# §21  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Tournament bar chart ──────────────────────────────────────────────────────
if not CODE4_SKIP_PLOTS:
    all_res = {**results_m1, **results_m2, "RW_baseline": rmse_rw}
    tags  = sorted(all_res, key=all_res.get)
    vals  = [all_res[t] for t in tags]

    def _color(t):
        if t == "RW_baseline":          return "#e74c3c"
        if t in (best_m1_tag, best_m2_tag): return "#2ecc71"
        if t in results_m2:             return "#9b59b6"
        return "#3498db"

    fig, ax = plt.subplots(figsize=(16, max(6, len(tags) * 0.45)))
    bars = ax.barh(tags, vals, color=[_color(t) for t in tags],
                   edgecolor="white", linewidth=0.5)
    ax.axvline(rmse_rw, color="#e74c3c", ls="--", lw=1.2)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=7)
    ax.set_xlabel("Weighted RMSE (back-test)")
    ax.set_title("Tournament: all model RMSE  (trainval→test)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="RW baseline"),
        Patch(color="#2ecc71", label="Selected (best M1/M2)"),
        Patch(color="#3498db", label="M1 candidate"),
        Patch(color="#9b59b6", label="M2 candidate"),
    ], fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tournament_rmse.png", dpi=150)
    plt.show()

    # ── Svensson betas over time ──────────────────────────────────────────────────
    sv_betas_diag = fit_sv_betas_ols(
        df_yc.loc[common_idx], yc_cols, MATURITY_MAP, sv_lam1_tv, sv_lam2_tv
    )
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Svensson betas  λ₁*={sv_lam1_tv:.2f}  λ₂*={sv_lam2_tv:.2f}", fontsize=11)
    for ax, col in zip(axes, SV_BETA_COLS):
        ax.plot(sv_betas_diag.index, sv_betas_diag[col], "k-", lw=1.2)
        ax.set_ylabel(col, fontsize=8); ax.grid(True, alpha=0.2)
        ax.axhline(0, color="gray", ls=":", lw=0.7)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "svensson_betas.png", dpi=150)
    plt.show()

    # ── Back-test per tenor ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(
        f"Back-test {test_idx[0]:%Y-%m}–{test_idx[-1]:%Y-%m}\n"
        f"M1={best_m1_tag}({rmse_m1_bt:.4f})  "
        f"M2={best_m2_tag}({rmse_m2_bt:.4f})  RW({rmse_rw:.4f})",
        fontsize=9,
    )
    t  = list(range(TEST_SIZE))
    tl = [d.strftime("%y-%m") for d in test_idx]

    for ax, col in zip(axes.flat, yc_cols):
        def _s(df, c=col):
            return (df[c].values.astype(float) if c in df.columns
                    else np.full(TEST_SIZE, np.nan))
        ax.plot(t, _s(yc_actual_bt),               "k-o",  ms=4, lw=1.8, label="Actual")
        ax.plot(t, _s(yc_forecasts[best_m1_tag]),  "b--o", ms=3, lw=1.2, label="M1")
        ax.plot(t, _s(yc_forecasts[best_m2_tag]),  "r--o", ms=3, lw=1.2, label="M2")
        ax.plot(t, _s(yc_rw_bt),                   "g:s",  ms=3, lw=1.0, label="RW")
        ax.set_title(col, fontsize=9)
        ax.set_xticks(t); ax.set_xticklabels(tl, rotation=45, fontsize=7)
        ax.grid(True, alpha=0.2)
        if col == "ON":
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_per_tenor.png", dpi=150)
    plt.show()

    # ── Forecast YC shape per month ───────────────────────────────────────────────
    x = np.arange(len(yc_cols))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Forecast YC  2025-10→2026-03  M1 vs M2", fontsize=12)
    for ax, dt in zip(axes.flat, TRUE_FUTURE_INDEX):
        ax.plot(x, yc_m1_final.loc[dt, yc_cols].astype(float),
                "b-o", ms=4, lw=1.5, label="M1")
        ax.plot(x, yc_m2_final.loc[dt, yc_cols].astype(float),
                "r-o", ms=4, lw=1.5, label="M2")
        ax.set_xticks(x); ax.set_xticklabels(yc_cols, rotation=45, fontsize=8)
        ax.set_title(dt.strftime("%Y-%m"), fontsize=9)
        ax.grid(True, alpha=0.2)
        if dt == TRUE_FUTURE_INDEX[0]:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_yc_forecasts.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# §22  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL SUMMARY  (v3 — all bugs fixed)")
print("=" * 70)
print(f"  Sample  : {common_idx[0]:%Y-%m} → {common_idx[-1]:%Y-%m}"
      f"  ({len(common_idx)} months)")
print(f"  Svensson: λ₁*={sv_lam1_tv:.2f}  λ₂*={sv_lam2_tv:.2f}"
      f"  (train: λ₁*={sv_lam1_tr:.2f}  λ₂*={sv_lam2_tr:.2f})")
print(f"  IV PCA  : [{', '.join(f'{v:.1%}' for v in iv_pca_f.explained_variance_ratio_)}]")
print(f"  MacroPCA: [{', '.join(f'{v:.1%}' for v in macro_reducer_f.explained_variance_ratio_)}]")
print()
print(f"  Back-test ({test_idx[0]:%Y-%m}→{test_idx[-1]:%Y-%m}):")
print(f"    Random Walk  = {rmse_rw:.5f}")
print()
print("  M1 results (sorted):")
for t, r in sorted(results_m1.items(), key=lambda x: x[1]):
    star = " ◀ SELECTED" if t == best_m1_tag else ""
    print(f"    {t:<34} = {r:.5f}  {_vs_rw(r)}{star}")
print()
print("  M2 results (sorted):")
for t, r in sorted(results_m2.items(), key=lambda x: x[1]):
    star = " ◀ SELECTED" if t == best_m2_tag else ""
    print(f"    {t:<34} = {r:.5f}  {_vs_rw(r)}{star}")
print()
print(f"  Selected M1 : {best_m1_tag}  RMSE={rmse_m1_bt:.5f}")
print(f"  Selected M2 : {best_m2_tag}  RMSE={rmse_m2_bt:.5f}")
print(f"  RMSEtotal   : {rmse_tot_bt:.5f}")
print()
print(f"  NaN M1={int(yc_m1_final.isna().sum().sum())}"
      f"  NaN M2={int(yc_m2_final.isna().sum().sum())}")
print(f"  Submission  : {SUBMISSION_PATH}")
print("=" * 70)

if CODE4_REPORT_JSON:
    def _safe_float(v):
        try:
            fv = float(v)
            if np.isfinite(fv):
                return fv
            return None
        except Exception:
            return None

    print(
        "CODE4_CV_SUMMARY="
        + json.dumps(
            {
                "analysis_start": str(common_idx[0].date()),
                "analysis_end": str(common_idx[-1].date()),
                "test_start": str(test_idx[0].date()),
                "test_end": str(test_idx[-1].date()),
                "len_train": len(train_idx),
                "len_val": len(val_idx),
                "len_test": len(test_idx),
                "rmse_rw": _safe_float(rmse_rw),
                "rmse_m1_bt": _safe_float(rmse_m1_bt),
                "rmse_m2_bt": _safe_float(rmse_m2_bt),
                "rmse_total_bt": _safe_float(rmse_tot_bt),
                "best_m1_tag": best_m1_tag,
                "best_m2_tag": best_m2_tag,
                "forced_m1_tag": CODE4_FORCE_M1_TAG or None,
                "forced_m2_tag": CODE4_FORCE_M2_TAG or None,
                "selected_m1_tag": best_m1_tag,
                "selected_m2_tag": best_m2_tag,
                "results_m1": {k: _safe_float(v) for k, v in results_m1.items()},
                "results_m2": {k: _safe_float(v) for k, v in results_m2.items()},
            },
            ensure_ascii=False,
        )
    )