from pathlib import Path
import warnings
import itertools
import os
import json
import sys
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВЫВОДА
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def suppress_warnings():
    """Подавление всех warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr


def print_header(text, char="═", width=80):
    """Печать главного заголовка"""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_section(text, char="─", width=80):
    """Печать секции"""
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_metric(label, value, width=35):
    """Печать метрики"""
    if isinstance(value, float):
        print(f"  {label:<{width}} {value:>12.6f}")
    else:
        print(f"  {label:<{width}} {str(value):>12}")


def print_progress(step, total, text=""):
    """Печать прогресса"""
    bar_len = 40
    filled = int(bar_len * step / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * step / total
    print(f"\r  [{bar}] {pct:>5.1f}% {text}", end="", flush=True)
    if step == total:
        print()


# ══════════════════════════════════════════════════════════════════════════════
#  БАЗОВЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def normalize_month_index(idx):
    return pd.to_datetime(idx).to_period("M").to_timestamp("M")


def to_num_df(df):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return out


def choose_d_by_adf(series, alpha=0.05):
    s = pd.Series(series).dropna()
    if len(s) < 12:
        return 1
    try:
        return 0 if adfuller(s)[1] < alpha else 1
    except Exception:
        return 1


def choose_pca_n_components(df, max_components=8, var_threshold=0.9):
    x = df.copy().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    if x.shape[0] < 2 or x.shape[1] < 1:
        return 1
    x_sc = StandardScaler().fit_transform(x)
    nmax = max(1, min(max_components, x_sc.shape[0], x_sc.shape[1]))
    pca = PCA(n_components=nmax).fit(x_sc)
    csum = np.cumsum(pca.explained_variance_ratio_)
    return max(1, min(int(np.searchsorted(csum, var_threshold) + 1), nmax))


# ══════════════════════════════════════════════════════════════════════════════
#  КОНСТАНТЫ
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
YC_PATH = DATA_DIR / "inputs" / "yield_curve.xlsx"
MACRO_PATH = DATA_DIR / "inputs" / "macro_updated.xlsx"
IV_PATH = DATA_DIR / "inputs" / "Problem_1_IV_train.xlsx"
OUTPUT_DIR = Path(os.getenv("CODE4_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
SUBMISSION_PATH = Path(os.getenv("CODE4_SUBMISSION_PATH",
                                 str(PROJECT_ROOT / "outputs" / "outputs" / "Problem_1_yield_curve_predict.xlsx")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

NS_LAMBDA = 0.7308
BETA_COLS = ["beta0", "beta1", "beta2"]
SV_BETA_COLS = ["sv_b0", "sv_b1", "sv_b2", "sv_b3"]
YIELD_COLS = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]
MATURITY_MAP = {
    "ON": 1 / 365, "1W": 7 / 365, "2W": 14 / 365,
    "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12,
    "6M": 6 / 12, "1Y": 1.0, "2Y": 2.0,
}
IV_TENOR_MAP = {
    "1M": 1 / 12, "2M": 2 / 12, "3M": 3 / 12, "6M": 6 / 12, "9M": 9 / 12,
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "4Y": 4.0, "5Y": 5.0,
    "6Y": 6.0, "7Y": 7.0, "8Y": 8.0, "9Y": 9.0, "10Y": 10.0,
}
MACRO_COLS_CANDIDATE = ["cbr", "inf", "observed_inf", "expected_inf", "usd", "moex", "urals"]
SV_LAM1_GRID = np.array([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5])
SV_LAM2_GRID = np.array([2.0, 3.0, 4.0, 5.0, 7.0])
NS_LAM_GRID = np.linspace(0.20, 1.50, 40)
TEST_SIZE = int(os.getenv("CODE4_TEST_SIZE", "6"))
VAL_SIZE = int(os.getenv("CODE4_VAL_SIZE", "6"))
MAX_VAR_TOTAL_VARS = 8
MAX_VAR_LAGS = 4
CLIP_SIGMA = 2.5


# ══════════════════════════════════════════════════════════════════════════════
#  NELSON-SIEGEL
# ══════════════════════════════════════════════════════════════════════════════

def ns_loadings(tau, lam):
    x = tau / lam
    if np.isclose(x, 0.0):
        return 1.0, 0.0
    l1 = (1 - np.exp(-x)) / x
    return l1, l1 - np.exp(-x)


def ns_loadings_vec(tau_arr, lam):
    tau_arr = np.asarray(tau_arr, dtype=float)
    x = np.where(np.abs(tau_arr / lam) < 1e-10, 1e-10, tau_arr / lam)
    l1 = (1.0 - np.exp(-x)) / x
    return l1, l1 - np.exp(-x)


def nelson_siegel_yield(tau, b0, b1, b2, lam):
    l1, l2 = ns_loadings(tau, lam)
    return b0 + b1 * l1 + b2 * l2


def fit_ns_betas_ols_row(tau_arr, y, lam, min_points=4):
    tau_arr = np.asarray(tau_arr, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < min_points:
        return None
    l1, l2 = ns_loadings_vec(tau_arr[mask], lam)
    X = np.column_stack([np.ones(mask.sum()), l1, l2])
    coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
    rmse = float(np.sqrt(np.mean((y[mask] - X @ coef) ** 2)))
    return {"beta0": float(coef[0]), "beta1": float(coef[1]), "beta2": float(coef[2]), "rmse": rmse}


def fit_ns_betas_frame(df_yc, yc_cols, maturity_map, lam):
    tau_arr = np.array([maturity_map[c] for c in yc_cols], dtype=float)
    rows = []
    for _, row in df_yc[yc_cols].iterrows():
        fit = fit_ns_betas_ols_row(tau_arr, row.to_numpy(dtype=float), lam)
        rows.append([np.nan] * 3 if fit is None else [fit["beta0"], fit["beta1"], fit["beta2"]])
    return pd.DataFrame(rows, index=df_yc.index, columns=BETA_COLS)


def reconstruct_yc_from_ns_betas(df_betas, maturity_map, lam):
    return pd.DataFrame(
        {lbl: [nelson_siegel_yield(tau, r["beta0"], r["beta1"], r["beta2"], lam)
               for _, r in df_betas.iterrows()]
         for lbl, tau in maturity_map.items()},
        index=df_betas.index,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SVENSSON
# ══════════════════════════════════════════════════════════════════════════════

def sv_loadings(tau, lam1, lam2):
    def _ns_pair(tau0, lam0):
        x = tau0 / lam0
        if np.isclose(x, 0.0):
            return 1.0, 0.0
        l1 = (1 - np.exp(-x)) / x
        return l1, l1 - np.exp(-x)

    l1, l2 = _ns_pair(tau, lam1)
    _, l3 = _ns_pair(tau, lam2)
    return l1, l2, l3


def sv_yield(tau, b0, b1, b2, b3, lam1, lam2):
    l1, l2, l3 = sv_loadings(tau, lam1, lam2)
    return b0 + b1 * l1 + b2 * l2 + b3 * l3


def fit_sv_betas_ols(df_yc, yc_cols, maturity_map, lam1, lam2):
    mats = np.array([maturity_map[c] for c in yc_cols], dtype=float)
    L = np.array([sv_loadings(t, lam1, lam2) for t in mats])
    X = np.column_stack([np.ones(len(mats)), L[:, 0], L[:, 1], L[:, 2]])
    rows = []
    for _, row in df_yc[yc_cols].iterrows():
        y = row.to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 4:
            rows.append([np.nan] * 4)
        else:
            coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            rows.append(np.clip(coef, -500.0, 500.0).tolist())
    return pd.DataFrame(rows, index=df_yc.index, columns=SV_BETA_COLS)


def reconstruct_yc_from_sv_betas(df_betas, yc_cols, maturity_map, lam1, lam2):
    return pd.DataFrame(
        {col: [sv_yield(maturity_map[col], r["sv_b0"], r["sv_b1"], r["sv_b2"], r["sv_b3"], lam1, lam2)
               for _, r in df_betas.iterrows()]
         for col in yc_cols},
        index=df_betas.index,
    )


def compute_weighted_rmse_curve(df_actual, df_fc, cols, w_on=0.4, w_rest_total=0.6):
    cols = [c for c in cols if c in df_actual.columns and c in df_fc.columns]
    if len(cols) < 2 or "ON" not in cols:
        return np.nan
    idx = df_actual.index.intersection(df_fc.index)
    if not len(idx):
        return np.nan
    actual = df_actual.loc[idx, cols].astype(float)
    pred = df_fc.loc[idx, cols].astype(float)
    rest = [c for c in cols if c != "ON"]
    w_each = w_rest_total / len(rest)
    weights = {"ON": w_on, **{c: w_each for c in rest}}
    wmse = sum(weights[c] * float(np.nanmean((actual[c] - pred[c]).values ** 2)) for c in cols)
    return float(np.sqrt(max(wmse, 0.0)))


def grid_search_sv_lambdas(df_yc, yc_cols, maturity_map, lam1_grid, lam2_grid):
    best_lam1, best_lam2, best_rmse = lam1_grid[0], lam2_grid[0], np.inf
    mats = np.array([maturity_map[c] for c in yc_cols], dtype=float)
    total = len(list(itertools.product(lam1_grid, lam2_grid)))
    step = 0
    for l1, l2 in itertools.product(lam1_grid, lam2_grid):
        step += 1
        print_progress(step, total, "Grid search λ")
        if l2 <= l1 + 0.5 or l1 <= 0 or l2 <= 0:
            continue
        L = np.array([sv_loadings(t, l1, l2) for t in mats])
        X = np.column_stack([np.ones(len(mats)), L[:, 0], L[:, 1], L[:, 2]])
        if np.linalg.cond(X) > 1e7:
            continue
        try:
            df_b = fit_sv_betas_ols(df_yc, yc_cols, maturity_map, l1, l2)
            yc_rec = reconstruct_yc_from_sv_betas(df_b, yc_cols, maturity_map, l1, l2)
            rmse = compute_weighted_rmse_curve(df_yc, yc_rec, yc_cols)
            if rmse < best_rmse:
                best_rmse, best_lam1, best_lam2 = rmse, l1, l2
        except Exception:
            continue
    return best_lam1, best_lam2, best_rmse


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_yield_curve():
    df = pd.read_excel(YC_PATH)
    if "Month" not in df.columns:
        raise ValueError("yield_curve.xlsx: 'Month' column missing")
    df["date"] = pd.to_datetime(df["Month"])
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return to_num_df(df)


def load_macro():
    df = pd.read_excel(MACRO_PATH)
    date_col = next((c for c in df.columns if str(c).lower() in
                     ["date", "dt", "month", "period", "\u0434\u0430\u0442\u0430"]), None)
    if date_col is None:
        raise ValueError("macro_updated.xlsx: no date column")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return to_num_df(df)


def load_iv_raw():
    df = pd.read_excel(IV_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Maturity (year fraction)", "Strike", "Volatility"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volatility"])


def _smile_stats(grp):
    vols = grp.sort_values("Strike")["Volatility"].values
    if len(vols) < 3:
        return {"rr": np.nan, "bf": np.nan, "smile_slope": np.nan}
    lo, hi = vols[0], vols[-1]
    mid = float(np.median(vols))
    strikes = grp.sort_values("Strike")["Strike"].values
    slope = 0.0 if strikes.std() < 1e-10 else float(np.polyfit(strikes, vols, 1)[0])
    return {"rr": hi - lo, "bf": (hi + lo) / 2 - mid, "smile_slope": slope}


def extract_iv_features(df_iv_raw):
    records = []
    for date, grp_date in df_iv_raw.groupby("Date"):
        row = {}
        atm = {}
        for tau, g in grp_date.groupby("Maturity (year fraction)"):
            k_atm = g["Strike"].median()
            atm[float(tau)] = float(g.loc[(g["Strike"] - k_atm).abs().idxmin(), "Volatility"])
        taus_s, vols_s = sorted(atm), [atm[t] for t in sorted(atm)]
        for lbl, tau in IV_TENOR_MAP.items():
            if tau in atm:
                row[f"atm_iv_{lbl}"] = atm[tau]
            elif len(taus_s) >= 2 and taus_s[0] <= tau <= taus_s[-1]:
                row[f"atm_iv_{lbl}"] = float(np.interp(tau, taus_s, vols_s))
            else:
                row[f"atm_iv_{lbl}"] = np.nan

        def _g(lbl):
            return row.get(f"atm_iv_{lbl}", np.nan)

        atm_vals = [v for k, v in row.items() if k.startswith("atm_iv_") and pd.notna(v)]
        row["ts_level_mean"] = float(np.nanmean(atm_vals)) if atm_vals else np.nan
        row["ts_slope_short"] = _g("3M") - _g("1M")
        row["ts_slope_long"] = _g("2Y") - _g("6M")
        row["ts_curvature"] = 2 * _g("1Y") - _g("3M") - _g("2Y")
        rr_list, bf_list, sl_list = [], [], []
        for _, g in grp_date.groupby("Maturity (year fraction)"):
            sm = _smile_stats(g)
            if pd.notna(sm["rr"]):
                rr_list.append(sm["rr"]);
                bf_list.append(sm["bf"]);
                sl_list.append(sm["smile_slope"])
        row["mean_rr"] = float(np.mean(rr_list)) if rr_list else np.nan
        row["mean_bf"] = float(np.mean(bf_list)) if bf_list else np.nan
        row["mean_smile_slope"] = float(np.mean(sl_list)) if sl_list else np.nan
        vols_all = grp_date["Volatility"].dropna().values
        if len(vols_all):
            row["iv_mean"] = float(np.mean(vols_all))
            row["iv_median"] = float(np.median(vols_all))
            row["iv_std"] = float(np.std(vols_all))
            row["iv_p10"] = float(np.percentile(vols_all, 10))
            row["iv_p90"] = float(np.percentile(vols_all, 90))
        else:
            for c in ["iv_mean", "iv_median", "iv_std", "iv_p10", "iv_p90"]:
                row[c] = np.nan
        records.append({"date": date, **row})
    df = pd.DataFrame(records).set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  PCA
# ══════════════════════════════════════════════════════════════════════════════

class _PCAReducer:
    def __init__(self, n_components=None, prefix="pc"):
        self.n_components = n_components
        self.prefix = prefix
        self.pca_ = self.scaler_ = self.cols_ = None

    def fit(self, df):
        x = df.copy().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        self.cols_ = list(x.columns)
        self.scaler_ = StandardScaler()
        x_sc = self.scaler_.fit_transform(x)
        n = self.n_components or choose_pca_n_components(x)
        n = min(n, x_sc.shape[0], x_sc.shape[1])
        self.pca_ = PCA(n_components=n).fit(x_sc)
        return self

    def transform(self, df):
        x = df.reindex(columns=self.cols_).copy().replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        scores = self.pca_.transform(self.scaler_.transform(x))
        return pd.DataFrame(scores, index=df.index,
                            columns=[f"{self.prefix}{i + 1}" for i in range(scores.shape[1])])

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    @property
    def explained_variance_ratio_(self):
        return self.pca_.explained_variance_ratio_ if self.pca_ else np.array([])


class MacroPCAReducer(_PCAReducer):
    def __init__(self, n_components=None):
        super().__init__(n_components, prefix="macro_pc")


class JointPCAReducer(_PCAReducer):
    def __init__(self, n_components=None):
        super().__init__(n_components, prefix="m2_pc")


# ══════════════════════════════════════════════════════════════════════════════
#  TIME-SERIES HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def project_series_ar1(series, steps):
    s = series.dropna()
    if len(s) < 4:
        return pd.Series([float(s.iloc[-1]) if len(s) else 0.0] * steps)
    try:
        d = choose_d_by_adf(s)
        res = ARIMA(s, order=(1, d, 0)).fit()
        return pd.Series(np.asarray(res.forecast(steps=steps)))
    except Exception:
        return pd.Series([float(s.iloc[-1])] * steps)


def project_df_forward(df, future_index):
    steps = len(future_index)
    return pd.DataFrame({col: project_series_ar1(df[col], steps).values
                         for col in df.columns}, index=future_index)


def clip_forecast(fc, train, n_sigma=CLIP_SIGMA):
    mu, std = float(train.mean()), float(train.std())
    return fc if std < 1e-8 else fc.clip(mu - n_sigma * std, mu + n_sigma * std)


def _val_mse(y_train, y_val, order):
    try:
        fc = ARIMA(y_train, order=order).fit().forecast(steps=len(y_val))
        return float(np.mean((y_val.values - np.asarray(fc)) ** 2))
    except Exception:
        return np.inf


def grid_search_arima_order(y_train, y_val, p_grid=(0, 1, 2, 3), q_grid=(0, 1, 2)):
    d = choose_d_by_adf(y_train)
    best_order, best_mse = (1, d, 0), np.inf
    for p, q in itertools.product(p_grid, q_grid):
        if p == 0 and q == 0:
            continue
        mse = _val_mse(y_train, y_val, (p, d, q))
        if mse < best_mse:
            best_mse, best_order = mse, (p, d, q)
    return best_order, best_mse


# ══════════════════════════════════════════════════════════════════════════════
#  ARIMA / ARIMAX МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════════

class ARIMAModel:
    def __init__(self, beta_cols, best_orders=None):
        self.beta_cols = beta_cols
        self.best_orders = best_orders or {}
        self._m = {}
        self._tr = {}

    def fit(self, df_b):
        for b in self.beta_cols:
            y = df_b[b].dropna()
            self._tr[b] = y.copy()
            if len(y) < 10:
                self._m[b] = None
                continue
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
            m = self._m.get(b)
            lv = float(tr.iloc[-1]) if len(tr) else np.nan
            if m is None:
                out[b] = lv
                continue
            try:
                fc = pd.Series(np.asarray(m.forecast(steps)), index=fi)
                out[b] = clip_forecast(fc, tr)
            except Exception:
                out[b] = lv
        return out


class ARIMAXModel:
    def __init__(self, beta_cols, best_orders=None):
        self.beta_cols = beta_cols
        self.best_orders = best_orders or {}
        self._m = {}
        self._tr = {}
        self._tx = {}
        self._ecols = {}

    @staticmethod
    def _clean(X):
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all").ffill().bfill()
        X = X.loc[:, X.std(ddof=0) > 1e-12]
        return X.fillna(0.0)

    def fit(self, df_b, df_exog):
        X_all = self._clean(df_exog)
        for b in self.beta_cols:
            y = df_b[b].dropna()
            common = y.index.intersection(X_all.index)
            if len(common) < 10:
                self._m[b] = None
                self._tr[b] = y
                continue
            y1 = y.loc[common].astype(float)
            X1 = X_all.loc[common].astype(float).dropna(axis=1, how="all")
            X1 = None if X1.shape[1] == 0 else X1
            order = self.best_orders.get(b, (2, choose_d_by_adf(y1), 0))
            try:
                res = (ARIMA(y1, order=order).fit() if X1 is None
                       else ARIMA(y1, exog=X1, order=order).fit())
                self._m[b] = res
                self._tr[b] = y1
                self._tx[b] = X1
                self._ecols[b] = list(X1.columns) if X1 is not None else []
            except Exception:
                self._m[b] = None
                self._tr[b] = y
        return self

    def forecast(self, steps, fi, df_exog_future):
        Xf_all = self._clean(df_exog_future)
        out = pd.DataFrame(index=fi, columns=self.beta_cols, dtype=float)
        for b in self.beta_cols:
            tr = self._tr.get(b, pd.Series(dtype=float))
            lv = float(tr.iloc[-1]) if len(tr) else np.nan
            m = self._m.get(b)
            if m is None:
                out[b] = lv
                continue
            try:
                ecols = self._ecols.get(b, [])
                if not ecols:
                    fc = pd.Series(np.asarray(m.forecast(steps)), index=fi)
                else:
                    Xf = Xf_all.reindex(columns=ecols)
                    for c in ecols:
                        if Xf[c].isna().all():
                            tX = self._tx.get(b)
                            Xf[c] = float(tX[c].iloc[-1]) if tX is not None else 0.0
                    Xf = Xf.reindex(fi).ffill().bfill().fillna(0.0)
                    fc = pd.Series(np.asarray(m.forecast(steps, exog=Xf)), index=fi)
                out[b] = clip_forecast(fc, tr)
            except Exception:
                out[b] = lv
        return out


class RandomWalkModel:
    def __init__(self, yc_cols=YIELD_COLS):
        self.yc_cols = yc_cols
        self._last = None

    def fit(self, df_yc):
        avail = [c for c in self.yc_cols if c in df_yc.columns]
        self._last = df_yc[avail].iloc[-1].astype(float)
        return self

    def forecast(self, steps, fi):
        return pd.DataFrame(np.tile(self._last.values, (steps, 1)),
                            index=fi, columns=self._last.index.tolist())


# ══════════════════════════════════════════════════════════════════════════════
#  VAR STABILITY DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def _check_var_stability(res) -> dict:
    out = {
        "is_stable": False,
        "max_modulus": np.nan,
        "n_explosive": 0,
        "n_near_unit": 0,
        "lag_order": getattr(res, "k_ar", 0),
        "n_vars": getattr(res, "neqs", 0),
        "n_obs": getattr(res, "nobs", 0),
        "eigenvalues": [],
        "verdict": "unknown",
    }
    try:
        k = res.neqs
        p = res.k_ar
        A = res.coefs
        C = np.zeros((k * p, k * p))
        for i in range(p):
            C[:k, i * k:(i + 1) * k] = A[i]
        if p > 1:
            C[k:, :k * (p - 1)] = np.eye(k * (p - 1))
        mods = np.sort(np.abs(np.linalg.eigvals(C)))[::-1]
        max_mod = float(mods[0])
        n_explosive = int(np.sum(mods >= 1.0))
        n_near_unit = int(np.sum((mods >= 0.90) & (mods < 1.0)))
        is_stable = n_explosive == 0
        out.update({
            "is_stable": is_stable,
            "max_modulus": max_mod,
            "n_explosive": n_explosive,
            "n_near_unit": n_near_unit,
            "eigenvalues": mods.tolist(),
        })
        if not is_stable:
            out["verdict"] = f"EXPLOSIVE (max|λ|={max_mod:.4f})"
        elif n_near_unit > 0:
            out["verdict"] = f"NEAR-UNIT (max|λ|={max_mod:.4f})"
        else:
            out["verdict"] = f"STABLE (max|λ|={max_mod:.4f})"
    except Exception as exc:
        out["verdict"] = f"Check failed: {exc}"
    return out


def _adf_on_stationarized(df_stat: pd.DataFrame, info: dict) -> dict:
    out = {}
    for col in df_stat.columns:
        s = df_stat[col].dropna()
        transform = info.get(col, "level")
        n = len(s)
        if n < 8:
            out[col] = {"transform": transform, "adf_pvalue": np.nan,
                        "is_stationary": None, "n_obs": n}
            continue
        try:
            pval = float(adfuller(s, autolag="AIC")[1])
            out[col] = {"transform": transform, "adf_pvalue": pval,
                        "is_stationary": pval < 0.10, "n_obs": n}
        except Exception:
            out[col] = {"transform": transform, "adf_pvalue": np.nan,
                        "is_stationary": None, "n_obs": n}
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  VAR MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _stationarize(df, prefer_growth=False):
    out, info = pd.DataFrame(index=df.index), {}
    for c in df.columns:
        s = (pd.to_numeric(df[c], errors="coerce")
             .replace([np.inf, -np.inf], np.nan).ffill().bfill())
        if s.isna().all():
            out[c], info[c] = s, "level";
            continue
        if choose_d_by_adf(s) == 0:
            out[c], info[c] = s, "level";
            continue
        if prefer_growth:
            g = s.pct_change().replace([np.inf, -np.inf], np.nan).ffill().bfill()
            if g.notna().sum() >= 15 and choose_d_by_adf(g) == 0:
                out[c], info[c] = g, "growth";
                continue
        d = s.diff().replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if d.notna().sum() >= 15 and choose_d_by_adf(d) == 0:
            out[c], info[c] = d, "diff";
            continue
        out[c], info[c] = s, "level"
    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill(), info


def _invert_forecast(last, fc_stat, info):
    out = pd.DataFrame(index=fc_stat.index, columns=fc_stat.columns, dtype=float)
    prev = last.copy().astype(float)
    for dt in fc_stat.index:
        for c in fc_stat.columns:
            v = float(fc_stat.loc[dt, c])
            t = info.get(c, "level")
            if pd.isna(v):
                out.loc[dt, c] = np.nan;
                continue
            if t == "level":
                out.loc[dt, c] = v;
                prev[c] = v
            elif t == "diff":
                b = prev.get(c, np.nan)
                out.loc[dt, c] = (b + v) if pd.notna(b) else np.nan
                prev[c] = out.loc[dt, c]
            elif t == "growth":
                b = prev.get(c, np.nan)
                out.loc[dt, c] = (b * (1 + v)) if pd.notna(b) else np.nan
                prev[c] = out.loc[dt, c]
            else:
                out.loc[dt, c] = v;
                prev[c] = v
    return out


def _fit_var_model(df_stat, maxlags=MAX_VAR_LAGS):
    df_stat = df_stat.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    if len(df_stat) < 15 or df_stat.shape[1] < 2:
        return None, "insufficient_data"
    ml = min(maxlags, max(1, len(df_stat) // (8 * df_stat.shape[1])),
             max(1, len(df_stat) // 4))
    ml = max(1, ml)
    try:
        model = VAR(df_stat)
        sel = model.select_order(maxlags=ml)
        p = max(1, int(sel.selected_orders.get("aic", 1) or 1))
        return model.fit(p), "ok"
    except Exception as e:
        return None, str(e)


class FlexVARModel:
    def __init__(self, spec, yc_cols=YIELD_COLS, maturity_map=MATURITY_MAP,
                 ns_lambda=NS_LAMBDA):
        self.spec = spec
        self.yc_cols = yc_cols
        self.maturity_map = maturity_map
        self.ns_lambda = ns_lambda
        self.tag = spec["tag"]
        self._res = None
        self._fit_status = "not_fitted"
        self._stat_df = pd.DataFrame()
        self._stat_info = {}
        self._last = pd.Series(dtype=float)
        self._beta_cols = []
        self._joint_cols = []
        self._sv_lam1 = 0.7
        self._sv_lam2 = 3.0
        self.stability: dict = {}
        self.adf_report: dict = {}

    def build(self, df_ns_betas, df_sv_betas, df_macro_pca, df_joint_pca,
              sv_lam1, sv_lam2):
        self._sv_lam1 = sv_lam1
        self._sv_lam2 = sv_lam2
        spec = self.spec
        if spec["beta"] == "sv":
            df_b = df_sv_betas[SV_BETA_COLS].copy()
            self._beta_cols = SV_BETA_COLS
        else:
            df_b = df_ns_betas[BETA_COLS].copy()
            self._beta_cols = BETA_COLS
        blocks = [df_b]
        if spec.get("joint"):
            blocks.append(df_joint_pca)
            self._joint_cols = list(df_joint_pca.columns)
        elif spec.get("macro") == "macro_pca":
            blocks.append(df_macro_pca)
        full = pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if full.shape[1] > MAX_VAR_TOTAL_VARS:
            beta_c = self._beta_cols
            other_c = [c for c in full.columns if c not in beta_c]
            full = full[beta_c + other_c[:max(0, MAX_VAR_TOTAL_VARS - len(beta_c))]]
        self._last = full.iloc[-1].copy()
        stat, info = _stationarize(full, prefer_growth=False)
        self._stat_df = stat.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        self._stat_info = {k: v for k, v in info.items() if k in self._stat_df.columns}
        self.adf_report = _adf_on_stationarized(self._stat_df, self._stat_info)
        self._res, self._fit_status = _fit_var_model(self._stat_df)
        if self._res is not None:
            self.stability = _check_var_stability(self._res)
        else:
            self.stability = {
                "is_stable": False, "max_modulus": np.nan,
                "n_explosive": 0, "n_near_unit": 0,
                "lag_order": 0, "n_vars": 0, "n_obs": 0,
                "eigenvalues": [],
                "verdict": f"VAR fit failed: {self._fit_status}",
            }
        return self

    def forecast(self, steps, fi):
        fallback = pd.DataFrame(
            np.tile(self._last[self._beta_cols].values, (steps, 1)),
            index=fi, columns=self._beta_cols)
        if self._res is not None:
            try:
                init = self._stat_df.values[-self._res.k_ar:]
                fc_stat = pd.DataFrame(
                    self._res.forecast(init, steps=steps),
                    index=fi, columns=self._stat_df.columns)
                fc_lv = _invert_forecast(self._last, fc_stat, self._stat_info)
                beta_fc = fc_lv[self._beta_cols].ffill().bfill()
                for b in self._beta_cols:
                    if beta_fc[b].isna().any():
                        beta_fc[b] = beta_fc[b].fillna(float(self._last.get(b, 0.0)))
            except Exception:
                beta_fc = fallback
        else:
            beta_fc = fallback
        if self.spec["beta"] == "sv":
            return reconstruct_yc_from_sv_betas(
                beta_fc, self.yc_cols, self.maturity_map, self._sv_lam1, self._sv_lam2)
        return reconstruct_yc_from_ns_betas(
            beta_fc, {k: self.maturity_map[k] for k in self.yc_cols}, lam=self.ns_lambda)


# ══════════════════════════════════════════════════════════════════════════════
#  VAR STABILITY REPORT (КРАТКИЙ)
# ══════════════════════════════════════════════════════════════════════════════

def print_var_stability_report(var_models, forecasts, test_idx, df_yc,
                               yc_cols, rmse_rw):
    if not var_models:
        return

    print_section("VAR СТАБИЛЬНОСТЬ", "═")

    print(f"\n  {'Модель':<28} {'k':>3} {'p':>3} {'RMSE':>8} {'max|λ|':>7} Статус")
    print(f"  {'─' * 72}")

    n_stable = n_near = n_expl = 0

    for m in var_models:
        st = m.stability
        rmse = compute_weighted_rmse_curve(
            df_yc.loc[test_idx],
            forecasts.get(m.tag, pd.DataFrame()),
            yc_cols,
        ) if m.tag in forecasts else np.nan

        if not st.get("is_stable") or st.get("n_explosive", 0) > 0:
            status = "⚠ НЕУСТ";
            n_expl += 1
        elif st.get("n_near_unit", 0) > 0:
            status = "⚡ ПОГР";
            n_near += 1
        elif not np.isnan(rmse) and rmse < rmse_rw:
            status = "✓ ОК";
            n_stable += 1
        else:
            status = "○ СЛАБ";
            n_stable += 1

        mm = f"{st.get('max_modulus', np.nan):.4f}" if not np.isnan(st.get('max_modulus', np.nan)) else " N/A"
        rr = f"{rmse:.5f}" if not np.isnan(rmse) else "  N/A "

        print(f"  {m.tag:<28} {st.get('n_vars', 0):>3} {st.get('lag_order', 0):>3} "
              f"{rr:>8} {mm:>7}  {status}")

    print(f"  {'─' * 72}")
    print(f"  Стабильных: {n_stable} | Пограничных: {n_near} | Нестабильных: {n_expl}")
    print(f"  Random Walk RMSE: {rmse_rw:.5f}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SPECS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_symmetric_specs():
    base = [
        {"tag": "VAR_NS_only", "beta": "ns", "macro": "none", "joint": False, "group": "M1"},
        {"tag": "VAR_NS_macro_pca", "beta": "ns", "macro": "macro_pca", "joint": False, "group": "M1"},
        {"tag": "VAR_SV_only", "beta": "sv", "macro": "none", "joint": False, "group": "M1"},
        {"tag": "VAR_SV_macro_pca", "beta": "sv", "macro": "macro_pca", "joint": False, "group": "M1"},
        {"tag": "VAR_NS_joint_pca", "beta": "ns", "macro": "none", "joint": True, "group": "M2"},
        {"tag": "VAR_SV_joint_pca", "beta": "sv", "macro": "none", "joint": True, "group": "M2"},
    ]
    seen, out = set(), []
    for spec in base:
        key = tuple(sorted(spec.items()))
        if key not in seen:
            out.append(spec);
            seen.add(key)
        if spec["beta"] == "sv":
            ns = {**spec, "beta": "ns", "tag": spec["tag"].replace("SV", "NS")}
            key = tuple(sorted(ns.items()))
            if key not in seen:
                out.append(ns);
                seen.add(key)
    return out


VAR_SPECS = build_symmetric_specs()
MODEL_PAIRS = [
    ("VAR_NS_macro_pca", "VAR_NS_joint_pca"),
    ("VAR_SV_macro_pca", "VAR_SV_joint_pca"),
    ("ARIMA_NS", "ARIMAX_NS_joint_pca"),
    ("ARIMA_SV", "ARIMAX_SV_joint_pca"),
]


def save_betas_plot(df_betas, cols, path, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in cols:
        if c in df_betas.columns:
            ax.plot(df_betas.index, df_betas[c], label=c)
    ax.set_title(title);
    ax.legend(loc="best");
    ax.grid(True, alpha=0.3)
    fig.tight_layout();
    fig.savefig(path, dpi=180);
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_header("ПРОГНОЗИРОВАНИЕ КРИВОЙ ДОХОДНОСТИ", "═")

    # ═══ Загрузка данных ═══
    print_section("1. ЗАГРУЗКА ДАННЫХ")
    with suppress_warnings():
        df_yc = load_yield_curve()
        df_macro = load_macro()
        df_iv_raw = load_iv_raw()
        df_iv_feat = extract_iv_features(df_iv_raw)

    yc_cols = [c for c in YIELD_COLS if c in df_yc.columns]
    macro_cols = [c for c in MACRO_COLS_CANDIDATE if c in df_macro.columns]
    common_idx = (df_yc.index.intersection(df_macro.index)
                  .intersection(df_iv_feat.index).sort_values())

    df_yc = df_yc.loc[common_idx, yc_cols].replace([np.inf, -np.inf], np.nan)
    df_macro = (df_macro.loc[common_idx, macro_cols]
                .replace([np.inf, -np.inf], np.nan).ffill().bfill())
    df_iv_feat = df_iv_feat.loc[common_idx].replace([np.inf, -np.inf], np.nan).ffill().bfill()

    print_metric("YC наблюдений", len(df_yc))
    print_metric("Теноров", len(yc_cols))
    print_metric("Макропеременных", len(macro_cols))

    if len(common_idx) <= VAL_SIZE + TEST_SIZE + 6:
        raise ValueError("Недостаточно данных")

    train_idx = common_idx[:-(VAL_SIZE + TEST_SIZE)]
    val_idx = common_idx[-(VAL_SIZE + TEST_SIZE):-TEST_SIZE]
    trainval_idx = common_idx[:-TEST_SIZE]
    test_idx = common_idx[-TEST_SIZE:]
    future_idx = test_idx

    print_metric("Train размер", len(train_idx))
    print_metric("Val размер", len(val_idx))
    print_metric("Test размер", len(test_idx))

    # ═══ Nelson-Siegel ═══
    print_section("2. NELSON-SIEGEL")
    with suppress_warnings():
        df_ns_betas_all = fit_ns_betas_frame(df_yc, yc_cols, MATURITY_MAP, NS_LAMBDA)
        ns_in_rmse = compute_weighted_rmse_curve(
            df_yc,
            reconstruct_yc_from_ns_betas(df_ns_betas_all, {k: MATURITY_MAP[k] for k in yc_cols}, NS_LAMBDA),
            yc_cols,
        )
    print_metric("NS λ", NS_LAMBDA)
    print_metric("NS in-sample RMSE", ns_in_rmse)

    # ═══ Svensson ═══
    print_section("3. SVENSSON")
    with suppress_warnings():
        sv_lam1, sv_lam2, _ = grid_search_sv_lambdas(
            df_yc.loc[train_idx], yc_cols, MATURITY_MAP, SV_LAM1_GRID, SV_LAM2_GRID)
        df_sv_betas_all = fit_sv_betas_ols(df_yc, yc_cols, MATURITY_MAP, sv_lam1, sv_lam2)
        sv_in_rmse = compute_weighted_rmse_curve(
            df_yc,
            reconstruct_yc_from_sv_betas(df_sv_betas_all, yc_cols, MATURITY_MAP, sv_lam1, sv_lam2),
            yc_cols,
        )
    print_metric("SV λ1", sv_lam1)
    print_metric("SV λ2", sv_lam2)
    print_metric("SV in-sample RMSE", sv_in_rmse)

    save_betas_plot(df_ns_betas_all, BETA_COLS,
                    OUTPUT_DIR / "nelson_betas.png",
                    f"Nelson (λ={NS_LAMBDA:.4f})")
    save_betas_plot(df_sv_betas_all, SV_BETA_COLS,
                    OUTPUT_DIR / "svensson_betas.png",
                    f"Svensson (λ1={sv_lam1:.2f}, λ2={sv_lam2:.2f})")

    # ═══ PCA ═══
    print_section("4. PCA")
    with suppress_warnings():
        macro_reducer = MacroPCAReducer()
        df_macro_train_pca = macro_reducer.fit_transform(df_macro.loc[train_idx])
        df_macro_trainval_pca = macro_reducer.transform(df_macro.loc[trainval_idx])
        df_macro_test_pca = macro_reducer.transform(df_macro.loc[test_idx])
        df_macro_all_pca = macro_reducer.transform(df_macro.loc[common_idx])

        joint_reducer = JointPCAReducer()
        joint_train = pd.concat([df_macro.loc[train_idx], df_iv_feat.loc[train_idx]], axis=1)
        joint_trainval = pd.concat([df_macro.loc[trainval_idx], df_iv_feat.loc[trainval_idx]], axis=1)
        joint_test = pd.concat([df_macro.loc[test_idx], df_iv_feat.loc[test_idx]], axis=1)
        joint_all = pd.concat([df_macro.loc[common_idx], df_iv_feat.loc[common_idx]], axis=1)
        df_joint_train_pca = joint_reducer.fit_transform(joint_train)
        df_joint_trainval_pca = joint_reducer.transform(joint_trainval)
        df_joint_test_pca = joint_reducer.transform(joint_test)
        df_joint_all_pca = joint_reducer.transform(joint_all)

    print_metric("Macro PC", df_macro_train_pca.shape[1])
    print_metric("Joint PC", df_joint_train_pca.shape[1])

    # ═══ Tournament ═══
    print_section("5. ТУРНИР МОДЕЛЕЙ")
    results: list = []
    forecasts: dict = {}
    fitted_var_models: List[FlexVARModel] = []

    def record(tag, group, beta_family, yc_fc):
        rmse = compute_weighted_rmse_curve(df_yc.loc[test_idx], yc_fc, yc_cols)
        results.append({"tag": tag, "group": group, "beta_family": beta_family, "rmse": rmse})
        forecasts[tag] = yc_fc.copy()
        status = "✓" if rmse < 0.5 else "○"
        print(f"  {status} {tag:32s} RMSE={rmse:.6f}")

    # Random Walk
    with suppress_warnings():
        rw = RandomWalkModel(yc_cols).fit(df_yc.loc[trainval_idx])
        yc_rw = rw.forecast(TEST_SIZE, future_idx)
    record("RandomWalk", "BASE", "none", yc_rw)
    rmse_rw = compute_weighted_rmse_curve(df_yc.loc[test_idx], yc_rw, yc_cols)

    def _best_order(betas_all, b):
        tr = betas_all.loc[train_idx, b].dropna()
        va = betas_all.loc[val_idx, b].dropna()
        if len(tr) >= 10 and len(va) > 0:
            with suppress_warnings():
                return grid_search_arima_order(tr, va)[0]
        return (1, choose_d_by_adf(tr), 0)

    ns_orders = {b: _best_order(df_ns_betas_all, b) for b in BETA_COLS}
    sv_orders = {b: _best_order(df_sv_betas_all, b) for b in SV_BETA_COLS}

    # ARIMA / ARIMAX
    with suppress_warnings():
        ns_arima = ARIMAModel(BETA_COLS, ns_orders).fit(df_ns_betas_all.loc[trainval_idx])
        record("ARIMA_NS", "M1", "ns",
               reconstruct_yc_from_ns_betas(
                   ns_arima.forecast(TEST_SIZE, future_idx),
                   {k: MATURITY_MAP[k] for k in yc_cols}, NS_LAMBDA))

        sv_arima = ARIMAModel(SV_BETA_COLS, sv_orders).fit(df_sv_betas_all.loc[trainval_idx])
        record("ARIMA_SV", "M1", "sv",
               reconstruct_yc_from_sv_betas(
                   sv_arima.forecast(TEST_SIZE, future_idx),
                   yc_cols, MATURITY_MAP, sv_lam1, sv_lam2))

        ns_arimax = ARIMAXModel(BETA_COLS, ns_orders).fit(
            df_ns_betas_all.loc[trainval_idx], df_joint_trainval_pca)
        record("ARIMAX_NS_joint_pca", "M2", "ns",
               reconstruct_yc_from_ns_betas(
                   ns_arimax.forecast(TEST_SIZE, future_idx, df_joint_test_pca),
                   {k: MATURITY_MAP[k] for k in yc_cols}, NS_LAMBDA))

        sv_arimax = ARIMAXModel(SV_BETA_COLS, sv_orders).fit(
            df_sv_betas_all.loc[trainval_idx], df_joint_trainval_pca)
        record("ARIMAX_SV_joint_pca", "M2", "sv",
               reconstruct_yc_from_sv_betas(
                   sv_arimax.forecast(TEST_SIZE, future_idx, df_joint_test_pca),
                   yc_cols, MATURITY_MAP, sv_lam1, sv_lam2))

    # VAR models
    print("\n  VAR модели:")
    for i, spec in enumerate(VAR_SPECS, 1):
        print_progress(i, len(VAR_SPECS), spec["tag"])
        with suppress_warnings():
            model = FlexVARModel(spec, yc_cols, MATURITY_MAP, NS_LAMBDA).build(
                df_ns_betas=df_ns_betas_all.loc[trainval_idx],
                df_sv_betas=df_sv_betas_all.loc[trainval_idx],
                df_macro_pca=df_macro_trainval_pca,
                df_joint_pca=df_joint_trainval_pca,
                sv_lam1=sv_lam1, sv_lam2=sv_lam2,
            )
            yc_fc = model.forecast(TEST_SIZE, future_idx)
        record(spec["tag"], spec["group"], spec["beta"], yc_fc)
        fitted_var_models.append(model)

    # ═══ Стабильность VAR ═══
    print_var_stability_report(
        var_models=fitted_var_models,
        forecasts=forecasts,
        test_idx=test_idx,
        df_yc=df_yc,
        yc_cols=yc_cols,
        rmse_rw=rmse_rw,
    )

    # ═══ Выбор пары ═══
    print_section("6. ВЫБОР ЛУЧШЕЙ ПАРЫ")
    results_df = (pd.DataFrame(results)
                  .sort_values(["group", "rmse", "tag"])
                  .reset_index(drop=True))
    best_m1 = results_df[results_df["group"] == "M1"].sort_values("rmse").iloc[0]
    best_m2 = results_df[results_df["group"] == "M2"].sort_values("rmse").iloc[0]
    best_m1_tag = str(best_m1["tag"])
    best_m2_tag = str(best_m2["tag"])

    if MODEL_PAIRS:
        pair_rows = []
        for m1t, m2t in MODEL_PAIRS:
            r1 = float(results_df.loc[results_df["tag"] == m1t, "rmse"].iloc[0]) \
                if (results_df["tag"] == m1t).any() else np.inf
            r2 = float(results_df.loc[results_df["tag"] == m2t, "rmse"].iloc[0]) \
                if (results_df["tag"] == m2t).any() else np.inf
            pair_rows.append((r1 + r2, m1t, m2t, r1, r2))
        valid = [r for r in pair_rows if np.isfinite(r[0])]
        if valid:
            _, best_m1_tag, best_m2_tag, _, _ = min(valid, key=lambda x: x[0])
            best_m1 = results_df.loc[results_df["tag"] == best_m1_tag].iloc[0]
            best_m2 = results_df.loc[results_df["tag"] == best_m2_tag].iloc[0]

    print_metric("Лучшая M1", best_m1_tag)
    print_metric("M1 RMSE", best_m1["rmse"])
    print_metric("Лучшая M2", best_m2_tag)
    print_metric("M2 RMSE", best_m2["rmse"])

    # ═══ Графики ═══
    print_section("7. СОХРАНЕНИЕ ГРАФИКОВ")
    allres = {r["tag"]: r["rmse"] for r in results if r["group"] in ("M1", "M2", "BASE")}
    tags = sorted(allres, key=allres.get)
    vals = [allres[t] for t in tags]
    sel1, sel2 = best_m1_tag, best_m2_tag

    def _color_t(t):
        if t == "RandomWalk":           return "#e74c3c"
        if t in (sel1, sel2):           return "#2ecc71"
        grp = results_df.loc[results_df["tag"] == t, "group"]
        return "#9b59b6" if (len(grp) and grp.iloc[0] == "M2") else "#3498db"

    fig, ax = plt.subplots(figsize=(16, max(6, len(tags) * 0.45)))
    bars = ax.barh(tags, vals, color=[_color_t(t) for t in tags],
                   edgecolor="white", linewidth=0.5)
    ax.axvline(rmse_rw, color="#e74c3c", ls="--", lw=1.2)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=7)
    ax.set_xlabel("RMSE")
    ax.set_title("Турнир моделей", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="RW"),
        Patch(color="#2ecc71", label="Выбрано"),
        Patch(color="#3498db", label="M1"),
        Patch(color="#9b59b6", label="M2"),
    ], fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tournament_rmse.png", dpi=150)
    plt.close(fig)
    print("  ✓ tournament_rmse.png")

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(
        f"Back-test | M1={best_m1_tag[:20]} | M2={best_m2_tag[:20]}",
        fontsize=9,
    )
    t_ = list(range(TEST_SIZE))
    tl_ = [d.strftime("%y-%m") for d in test_idx]
    for ax, col in zip(axes.flat, yc_cols):
        def _s(df, c=col):
            return (df[c].values.astype(float)
                    if c in df.columns else np.full(TEST_SIZE, np.nan))

        ax.plot(t_, _s(df_yc.loc[test_idx]), "k-o", ms=4, lw=1.8, label="Факт")
        ax.plot(t_, _s(forecasts[best_m1_tag]), "b--o", ms=3, lw=1.2, label="M1")
        ax.plot(t_, _s(forecasts[best_m2_tag]), "r--o", ms=3, lw=1.2, label="M2")
        ax.plot(t_, _s(yc_rw), "g--s", ms=3, lw=1.0, label="RW")
        ax.set_title(col, fontsize=9)
        ax.set_xticks(t_);
        ax.set_xticklabels(tl_, rotation=45, fontsize=7)
        ax.grid(True, alpha=0.2)
        if col == "ON":
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "backtest_per_tenor.png", dpi=150)
    plt.close(fig)
    print("  ✓ backtest_per_tenor.png")

    # ═══ Финальный прогноз ═══
    print_section("8. ФИНАЛЬНЫЙ ПРОГНОЗ")
    full_ns = fit_ns_betas_frame(df_yc.loc[trainval_idx], yc_cols, MATURITY_MAP, NS_LAMBDA)
    full_sv = fit_sv_betas_ols(df_yc.loc[trainval_idx], yc_cols, MATURITY_MAP, sv_lam1, sv_lam2)
    future_joint = project_df_forward(df_joint_all_pca.loc[trainval_idx], test_idx)

    best_tag = best_m2_tag if best_m2["rmse"] <= best_m1["rmse"] else best_m1_tag

    with suppress_warnings():
        if best_tag == "ARIMA_NS":
            final_fc = reconstruct_yc_from_ns_betas(
                ARIMAModel(BETA_COLS, ns_orders).fit(full_ns).forecast(TEST_SIZE, test_idx),
                {k: MATURITY_MAP[k] for k in yc_cols}, NS_LAMBDA)
        elif best_tag == "ARIMA_SV":
            final_fc = reconstruct_yc_from_sv_betas(
                ARIMAModel(SV_BETA_COLS, sv_orders).fit(full_sv).forecast(TEST_SIZE, test_idx),
                yc_cols, MATURITY_MAP, sv_lam1, sv_lam2)
        elif best_tag == "ARIMAX_NS_joint_pca":
            final_fc = reconstruct_yc_from_ns_betas(
                ARIMAXModel(BETA_COLS, ns_orders).fit(full_ns, df_joint_all_pca.loc[trainval_idx])
                .forecast(TEST_SIZE, test_idx, future_joint),
                {k: MATURITY_MAP[k] for k in yc_cols}, NS_LAMBDA)
        elif best_tag == "ARIMAX_SV_joint_pca":
            final_fc = reconstruct_yc_from_sv_betas(
                ARIMAXModel(SV_BETA_COLS, sv_orders).fit(full_sv, df_joint_all_pca.loc[trainval_idx])
                .forecast(TEST_SIZE, test_idx, future_joint),
                yc_cols, MATURITY_MAP, sv_lam1, sv_lam2)
        elif best_tag.startswith("VAR_"):
            chosen_spec = next(s for s in VAR_SPECS if s["tag"] == best_tag)
            final_fc = (FlexVARModel(chosen_spec, yc_cols, MATURITY_MAP, NS_LAMBDA)
                        .build(full_ns, full_sv,
                               df_macro_all_pca.loc[trainval_idx],
                               df_joint_all_pca.loc[trainval_idx],
                               sv_lam1, sv_lam2)
                        .forecast(TEST_SIZE, test_idx))
        else:
            final_fc = rw.forecast(TEST_SIZE, test_idx)

    final_fc = final_fc.reindex(columns=yc_cols).ffill().bfill()
    print_metric("Финальная модель", best_tag)

    x_ = np.arange(len(yc_cols))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Прогноз кривой доходности", fontsize=12)
    m1f = forecasts[best_m1_tag].reindex(test_idx).copy()
    m2f = forecasts[best_m2_tag].reindex(test_idx).copy()
    for ax, dt in zip(axes.flat, test_idx):
        ax.plot(x_, m1f.loc[dt, yc_cols].astype(float), "b-o", ms=4, lw=1.5, label="M1")
        ax.plot(x_, m2f.loc[dt, yc_cols].astype(float), "r-o", ms=4, lw=1.5, label="M2")
        ax.set_xticks(x_);
        ax.set_xticklabels(yc_cols, rotation=45, fontsize=8)
        ax.set_title(dt.strftime("%Y-%m"), fontsize=9);
        ax.grid(True, alpha=0.2)
        if dt == test_idx[0]:
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_yc_forecasts.png", dpi=150)
    plt.close(fig)
    print("  ✓ final_yc_forecasts.png")

    # ═══ Сохранение ═══
    print_section("9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    submission = final_fc.copy()
    submission.index.name = "date"
    with pd.ExcelWriter(SUBMISSION_PATH, engine="openpyxl") as writer:
        submission.to_excel(writer, sheet_name="forecast")
        results_df.to_excel(writer, sheet_name="tournament", index=False)
        pd.DataFrame({
            "metric": ["ns_in_sample_rmse", "sv_in_sample_rmse", "best_m1", "best_m2", "best_final"],
            "value": [ns_in_rmse, sv_in_rmse, best_m1["tag"], best_m2["tag"], best_tag],
        }).to_excel(writer, sheet_name="summary", index=False)
    print(f"  ✓ {SUBMISSION_PATH.name}")

    report = {
        "ns_lambda": NS_LAMBDA,
        "sv_lambda1": float(sv_lam1),
        "sv_lambda2": float(sv_lam2),
        "macro_pca_n": int(df_macro_train_pca.shape[1]),
        "joint_pca_n": int(df_joint_train_pca.shape[1]),
        "best_m1": str(best_m1["tag"]),
        "best_m2": str(best_m2["tag"]),
        "best_final": str(best_tag),
        "var_stability": {
            m.tag: {
                "is_stable": m.stability.get("is_stable"),
                "max_modulus": m.stability.get("max_modulus"),
                "verdict": m.stability.get("verdict"),
            }
            for m in fitted_var_models
        },
    }
    (OUTPUT_DIR / "code4_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  ✓ code4_report.json")

    print_header("ЗАВЕРШЕНО", "═")
    print_metric("Best M1", best_m1_tag)
    print_metric("Best M2", best_m2_tag)
    print_metric("Final", best_tag)
    print()


if __name__ == "__main__":
    main()