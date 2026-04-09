import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# =========================================================
# 2.1 ПАРАМЕТРЫ
# =========================================================

BETAS_PATH      = "data/ns_results/betas_0_7308.csv"
YC_PATH         = "data/inputs/yield_curve.xlsx"
MACRO_PATH      = "data/inputs/macro_updated.xlsx"
IV_PATH         = "Problem_1_IV_train.xlsx"

START_DATE_BETAS = "2019-03-01"
FREQ             = "MS"

BETA_COLS  = ["beta0", "beta1", "beta2"]
MACRO_COLS_CANDIDATE = [
    "cbr", "inf", "observed_inf", "expected_inf",
    "usd", "moex", "brent", "vix"
]

YIELD_COLS = ["ON", "1W", "2W", "1M", "2M", "3M", "6M", "1Y", "2Y"]

MATURITY_MAP = {
    "ON": 1/365,
    "1W": 7/365,
    "2W": 14/365,
    "1M": 1/12,
    "2M": 2/12,
    "3M": 3/12,
    "6M": 6/12,
    "1Y": 1.0,
    "2Y": 2.0
}

NS_LAMBDA = 0.7308

# --- НОВЫЕ ПАРАМЕТРЫ: разделение на train / val / test ---
TEST_SIZE = 6          # последние 6 месяцев — тест
VAL_SIZE  = 6          # ещё 6 до теста — валидация для подбора порядков ARIMA

# Решётка для grid-search (p, q) — ищем по MSE на VAL-выборке
P_GRID = [0, 1, 2, 3]
Q_GRID = [0, 1, 2]

# Ограничение прогноза (mean-reversion clip): не уходить дальше N sigma от
# скользящего среднего тренировочного ряда
CLIP_SIGMA = 2.5


# =========================================================
# 2.2 ВСПОМОГАТЕЛЬНЫЕ
# =========================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def choose_d_by_adf(series):
    """ADF-тест: если p < 0.05 — ряд стационарен (d=0), иначе d=1."""
    try:
        pval = adfuller(series.dropna())[1]
        return 0 if pval < 0.05 else 1
    except Exception:
        return 1


def normalize_month_index(idx):
    idx = pd.to_datetime(idx)
    return idx.to_period("M").to_timestamp("M")


# =========================================================
# 2.3 ЗАГРУЗКА ДАННЫХ
# =========================================================

df_betas = pd.read_csv(BETAS_PATH)
df_betas["date"] = pd.date_range(
    start=START_DATE_BETAS,
    periods=len(df_betas),
    freq=FREQ
)
df_betas = df_betas.set_index("date").sort_index()
df_betas.index = normalize_month_index(df_betas.index)

for col in df_betas.columns:
    df_betas[col] = pd.to_numeric(df_betas[col], errors="coerce")


df_yc = pd.read_excel(YC_PATH)
df_yc["date"] = pd.to_datetime(df_yc["Month"])
df_yc = df_yc.set_index("date").sort_index()
df_yc.index = normalize_month_index(df_yc.index)

for col in df_yc.columns:
    df_yc[col] = (
        df_yc[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df_yc[col] = pd.to_numeric(df_yc[col], errors="coerce")


df_macro = pd.read_excel(MACRO_PATH)
date_col = None
for col in df_macro.columns:
    if col.lower() in ["date", "dt", "month", "period", "дата"]:
        date_col = col
        break
if date_col is None:
    raise ValueError("Нет колонки Date в macro")
if date_col != "date":
    df_macro = df_macro.rename(columns={date_col: "date"})
df_macro["date"] = pd.to_datetime(df_macro["date"])
df_macro = df_macro.set_index("date").sort_index()
df_macro.index = normalize_month_index(df_macro.index)

for col in df_macro.columns:
    df_macro[col] = pd.to_numeric(df_macro[col], errors="coerce")


df_iv = pd.read_excel(IV_PATH)
df_iv.columns = [c.strip() for c in df_iv.columns]

if "Date" not in df_iv.columns:
    raise ValueError("Нет колонки Date в IV")

df_iv["Date"] = pd.to_datetime(df_iv["Date"])
df_iv["Maturity (year fraction)"] = pd.to_numeric(
    df_iv["Maturity (year fraction)"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)
df_iv["Volatility"] = pd.to_numeric(
    df_iv["Volatility"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce"
)

df_iv = df_iv.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volatility"])

df_iv = df_iv.groupby("Date").agg(
    iv_mean=("Volatility", "mean"),
    iv_median=("Volatility", "median"),
    iv_count=("Volatility", "count")
).reset_index()

df_iv["date"] = pd.to_datetime(df_iv["Date"])
df_iv = df_iv[["date", "iv_mean", "iv_median", "iv_count"]].dropna()
df_iv = df_iv.set_index("date").sort_index()
df_iv.index = normalize_month_index(df_iv.index)


# =========================================================
# 2.4 НЕЛЬСОН-ЗИГЕЛЬ
# =========================================================

def ns_loadings(tau, lam):
    x = tau / lam
    if np.isclose(x, 0.0):
        l1, l2 = 1.0, 0.0
    else:
        l1 = (1 - np.exp(-x)) / x
        l2 = l1 - np.exp(-x)
    return l1, l2


def nelson_siegel_yield(tau, beta0, beta1, beta2, lam=NS_LAMBDA):
    l1, l2 = ns_loadings(tau, lam)
    return beta0 + beta1*l1 + beta2*l2


def reconcile_yc_from_betas(df_betas, maturity_map, lam=NS_LAMBDA):
    out = pd.DataFrame(index=df_betas.index)
    for label, tau in maturity_map.items():
        out[label] = [
            nelson_siegel_yield(
                tau=tau,
                beta0=row["beta0"],
                beta1=row["beta1"],
                beta2=row["beta2"],
                lam=lam
            )
            for _, row in df_betas.iterrows()
        ]
    return out


# =========================================================
# 2.5 СПЛИТЫ: train / val / test
# =========================================================

beta_cols  = [c for c in BETA_COLS if c in df_betas.columns]
yc_cols    = [c for c in YIELD_COLS if c in df_yc.columns]
macro_cols = [c for c in MACRO_COLS_CANDIDATE if c in df_macro.columns]

common_idx = (
    df_betas.index
    .intersection(df_yc.index)
    .intersection(df_macro.index)
    .sort_values()
)

df_betas = df_betas.loc[common_idx, beta_cols].replace([np.inf, -np.inf], np.nan).dropna()
df_yc    = df_yc.loc[common_idx, yc_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
df_macro = df_macro.loc[common_idx, macro_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

df_all = df_macro.join(df_iv[["iv_mean"]], how="left")
df_all["iv_mean"] = df_all["iv_mean"].ffill()

macro_cols_with_iv = macro_cols + ["iv_mean"]

# --- три сплита ---
train_idx     = common_idx[:-(VAL_SIZE + TEST_SIZE)]
val_idx       = common_idx[-(VAL_SIZE + TEST_SIZE):-TEST_SIZE]
trainval_idx  = common_idx[:-TEST_SIZE]      # train+val для финального обучения
test_idx      = common_idx[-TEST_SIZE:]

df_b_train    = df_betas.loc[train_idx]
df_b_val      = df_betas.loc[val_idx]
df_b_trainval = df_betas.loc[trainval_idx]
df_b_test     = df_betas.loc[test_idx]

df_m_train    = df_all.loc[train_idx,    macro_cols].copy()
df_m_val      = df_all.loc[val_idx,      macro_cols].copy()
df_m_trainval = df_all.loc[trainval_idx, macro_cols].copy()
df_m_test     = df_all.loc[test_idx,     macro_cols].copy()

df_iv_train    = df_all.loc[train_idx,    ["iv_mean"]].copy()
df_iv_val      = df_all.loc[val_idx,      ["iv_mean"]].copy()
df_iv_trainval = df_all.loc[trainval_idx, ["iv_mean"]].copy()
df_iv_test     = df_all.loc[test_idx,     ["iv_mean"]].copy()


# =========================================================
# 2.6 МЕТРИКА (точная формула хакатона)
# =========================================================

def compute_weighted_rmse_curve(
    df_yc_actual: pd.DataFrame,
    df_yc_forecast: pd.DataFrame,
    cols=YIELD_COLS,
    w_on=0.4,
    w_rest_total=0.6
):
    """
    sqrt( sum_c w_c * mean_t((y_c,t - yhat_c,t)^2) )
    ON получает вес 0.4; каждый из остальных 8 теноров — 0.075.
    """
    if not set(cols).issubset(df_yc_actual.columns) or \
       not set(cols).issubset(df_yc_forecast.columns):
        return np.nan

    n_rest = len(cols) - 1
    w_rest = w_rest_total / n_rest

    weight = {c: (w_on if c == "ON" else w_rest) for c in cols}

    sq_err_total = 0.0
    for col in cols:
        actual   = df_yc_actual[col].dropna()
        forecast = df_yc_forecast[col].reindex(actual.index).dropna()
        idx = actual.index.intersection(forecast.index)
        if len(idx) == 0:
            continue
        mse_c = np.mean((actual.loc[idx].values - forecast.loc[idx].values) ** 2)
        sq_err_total += weight[col] * mse_c

    return float(np.sqrt(sq_err_total))


# =========================================================
# 2.7 GRID-SEARCH ПОРЯДКОВ ARIMA ПО VAL-ВЫБОРКЕ
# =========================================================

def _val_mse(y_train, y_val, order):
    try:
        fit = ARIMA(y_train, order=order).fit()
        fc  = fit.forecast(steps=len(y_val))
        return float(np.mean((y_val.values - np.asarray(fc)) ** 2))
    except Exception:
        return np.inf


def grid_search_arima_order(y_train, y_val, p_grid=P_GRID, q_grid=Q_GRID):
    """Перебирает (p, d, q) и выбирает порядок с минимальным val-MSE."""
    d = choose_d_by_adf(y_train)
    best_order, best_mse = (1, d, 0), np.inf

    for p, q in itertools.product(p_grid, q_grid):
        if p == 0 and q == 0:
            continue
        mse = _val_mse(y_train, y_val, (p, d, q))
        if mse < best_mse:
            best_mse, best_order = mse, (p, d, q)

    return best_order, best_mse


def find_best_orders(df_b_train, df_b_val):
    """Возвращает {beta_col: best_order} после grid-search."""
    best_orders = {}
    print("\n--- Grid-search ARIMA-порядков по VAL-выборке ---")
    for beta in BETA_COLS:
        y_tr = df_b_train[beta].dropna()
        y_va = df_b_val[beta].dropna()
        if len(y_tr) < 10 or len(y_va) == 0:
            best_orders[beta] = (2, choose_d_by_adf(y_tr), 0)
            continue
        order, val_mse = grid_search_arima_order(y_tr, y_va)
        print(f"  {beta}: best order={order}, val_MSE={val_mse:.4f}")
        best_orders[beta] = order
    return best_orders


# =========================================================
# 2.8 MEAN-REVERSION CLIP
# =========================================================

def clip_forecast_to_history(forecast_series, train_series, n_sigma=CLIP_SIGMA):
    """
    Ограничивает прогноз диапазоном [mu - n_sigma*std, mu + n_sigma*std].
    Предотвращает уход ARIMA на концах кривой.
    """
    mu  = float(train_series.mean())
    std = float(train_series.std())
    if std < 1e-8:
        return forecast_series
    return forecast_series.clip(mu - n_sigma * std, mu + n_sigma * std)


# =========================================================
# 2.9 КЛАССЫ МОДЕЛЕЙ
# =========================================================

class ARIMA_Model:
    def __init__(self, beta_cols, best_orders=None, default_order=(2, 0, 0)):
        self.beta_cols     = beta_cols
        self.best_orders   = best_orders or {}
        self.default_order = default_order
        self.models        = {}
        self.used_orders   = {}
        self.train_series  = {}

    def fit(self, df_b):
        for beta in self.beta_cols:
            y = df_b[beta].dropna()
            self.train_series[beta] = y.copy()
            if len(y) < 10:
                self.models[beta] = None; self.used_orders[beta] = None; continue
            order = self.best_orders.get(beta) or (
                self.default_order[0], choose_d_by_adf(y), self.default_order[2]
            )
            try:
                self.models[beta]      = ARIMA(y, order=order).fit()
                self.used_orders[beta] = order
            except Exception:
                self.models[beta] = None; self.used_orders[beta] = None
        return self

    def forecast(self, steps, future_index, apply_clip=True):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)
        for beta in self.beta_cols:
            mod = self.models.get(beta)
            if mod is None: out[beta] = np.nan; continue
            try:
                fc = pd.Series(np.asarray(mod.forecast(steps=steps)), index=future_index)
                if apply_clip and beta in self.train_series:
                    fc = clip_forecast_to_history(fc, self.train_series[beta])
                out[beta] = fc
            except Exception:
                out[beta] = np.nan
        return out


class ARIMAX_Model:
    def __init__(self, beta_cols, macro_cols, best_orders=None, default_order=(2, 0, 0)):
        self.beta_cols     = beta_cols
        self.macro_cols    = macro_cols
        self.best_orders   = best_orders or {}
        self.default_order = default_order
        self.models        = {}
        self.used_orders   = {}
        self.train_series  = {}
        self.train_X       = {}

    def _clean_exog(self, X):
        if X is None or len(X) == 0:
            return pd.DataFrame()
        X = X.copy().replace([np.inf, -np.inf], np.nan)
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.dropna(axis=1, how="all")
        if X.shape[1] == 0: return X
        X = X.ffill().bfill()
        std = X.std(ddof=0)
        X   = X[std[(std > 1e-12) | std.isna()].index.tolist()]
        return X.fillna(0.0)

    def _store_none(self, beta):
        self.models[beta] = self.used_orders[beta] = self.train_series[beta] = self.train_X[beta] = None

    def fit(self, df_b, df_exog):
        for beta in self.beta_cols:
            try:
                y = pd.to_numeric(df_b[beta].copy(), errors="coerce")
                X = self._clean_exog(df_exog.copy())
                common = y.index.intersection(X.index).sort_values()
                y, X = y.loc[common], X.loc[common]
                tmp  = pd.concat([y.rename(beta), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                if tmp.shape[0] < 10: self._store_none(beta); continue
                y1   = tmp[beta].astype(float)
                X1   = tmp.drop(columns=[beta]).astype(float)
                X1   = X1 if X1.shape[1] > 0 else None
                order = self.best_orders.get(beta) or (
                    self.default_order[0], choose_d_by_adf(y1), self.default_order[2]
                )
                if X1 is None:
                    res = ARIMA(y1, order=order).fit(); self.train_X[beta] = None
                else:
                    res = ARIMA(y1, exog=X1, order=order).fit(); self.train_X[beta] = X1.copy()
                self.models[beta]       = res
                self.used_orders[beta]  = order
                self.train_series[beta] = y1.copy()
            except Exception:
                self._store_none(beta)
        return self

    def forecast(self, steps, future_index, df_exog_future, apply_clip=True):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)
        for beta in self.beta_cols:
            out[beta] = np.nan
            res = self.models.get(beta)
            if res is None: continue
            try:
                Xtrain = self.train_X.get(beta)
                if Xtrain is None:
                    fc = pd.Series(np.asarray(res.forecast(steps=steps)), index=future_index)
                else:
                    Xf = self._clean_exog(df_exog_future.copy().loc[future_index])
                    Xf = Xf.reindex(columns=Xtrain.columns)
                    for c in Xf.columns:
                        if Xf[c].isna().all():
                            Xf[c] = Xtrain[c].iloc[-1] if c in Xtrain.columns else 0.0
                    Xf = Xf.ffill().bfill().fillna(0.0)
                    if Xf.shape[0] != steps:
                        Xf = Xf.reindex(future_index).ffill().bfill().fillna(0.0)
                    fc = pd.Series(np.asarray(res.forecast(steps=steps, exog=Xf)), index=future_index)
                if apply_clip and self.train_series.get(beta) is not None:
                    fc = clip_forecast_to_history(fc, self.train_series[beta])
                out[beta] = fc
            except Exception:
                out[beta] = np.nan
        return out


# =========================================================
# 2.10 GRID-SEARCH → ФИНАЛЬНОЕ ОБУЧЕНИЕ → ПРОГНОЗ
# =========================================================

# 1) Ищем лучшие порядки на train/val
best_orders = find_best_orders(df_b_train, df_b_val)

# 2) Финальное обучение на train+val, прогноз на test

# ARIMA без IV
model_arima_no_iv = ARIMA_Model(beta_cols=BETA_COLS, best_orders=best_orders)
model_arima_no_iv.fit(df_b_trainval)
fc_betas_arima_no_iv = model_arima_no_iv.forecast(steps=TEST_SIZE, future_index=test_idx)

# ARIMAX: только IV
model_arimax_iv_only = ARIMAX_Model(beta_cols=BETA_COLS, macro_cols=["iv_mean"], best_orders=best_orders)
model_arimax_iv_only.fit(df_b_trainval, df_iv_trainval)
fc_betas_arimax_iv_only = model_arimax_iv_only.forecast(
    steps=TEST_SIZE, future_index=test_idx, df_exog_future=df_iv_test
)

# ARIMAX: только макро
model_arimax_no_iv = ARIMAX_Model(beta_cols=BETA_COLS, macro_cols=macro_cols, best_orders=best_orders)
model_arimax_no_iv.fit(df_b_trainval, df_m_trainval)
fc_betas_arimax_no_iv = model_arimax_no_iv.forecast(
    steps=TEST_SIZE, future_index=test_idx, df_exog_future=df_m_test
)

# ARIMAX: макро + IV
X_trainval_with_iv = pd.concat([df_m_trainval, df_iv_trainval], axis=1)
X_test_with_iv     = pd.concat([df_m_test,     df_iv_test],     axis=1)

model_arimax_iv = ARIMAX_Model(beta_cols=BETA_COLS, macro_cols=macro_cols_with_iv, best_orders=best_orders)
model_arimax_iv.fit(df_b_trainval, X_trainval_with_iv)
fc_betas_arimax_iv = model_arimax_iv.forecast(
    steps=TEST_SIZE, future_index=test_idx, df_exog_future=X_test_with_iv
)


# =========================================================
# 2.11 КОНВЕРТАЦИЯ БЕТА → КРИВАЯ + МЕТРИКИ
# =========================================================

yc_actual = reconcile_yc_from_betas(
    df_b_test, maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS}
)

def yc_rmse(fc_betas):
    yc_fc = reconcile_yc_from_betas(
        fc_betas, maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS}
    )
    return compute_weighted_rmse_curve(yc_actual, yc_fc), yc_fc


rmse_arima_no_iv,    yc_fc_arima_no_iv    = yc_rmse(fc_betas_arima_no_iv)
rmse_arimax_iv_only, yc_fc_arimax_iv_only = yc_rmse(fc_betas_arimax_iv_only)
rmse_arimax_no_iv,   yc_fc_arimax_no_iv   = yc_rmse(fc_betas_arimax_no_iv)
rmse_arimax_iv,      yc_fc_arimax_iv      = yc_rmse(fc_betas_arimax_iv)


# =========================================================
# 2.12 ВЫВОД РЕЗУЛЬТАТОВ
# =========================================================

print("\n" + "="*60)
print("RMSE по кривой (ON-2Y, тест 6 точек)")
print("="*60)
results = {
    "ARIMA without IV":    rmse_arima_no_iv,
    "ARIMAX with IV only": rmse_arimax_iv_only,
    "ARIMAX without IV":   rmse_arimax_no_iv,
    "ARIMAX with IV":      rmse_arimax_iv,
}
for name, val in results.items():
    tag = f"{val:.4f}" if not np.isnan(val) else "N/A"
    print(f"  {name:<25}: {tag}")

delta = 0.5
rmse_total_arima  = delta * rmse_arima_no_iv  + (1 - delta) * rmse_arimax_iv_only
rmse_total_arimax = delta * rmse_arimax_no_iv + (1 - delta) * rmse_arimax_iv

print("\n" + "="*60)
print("RMSEtotal (хакатонная формула: 0.5*M1 + 0.5*M2)")
print("="*60)
print(f"  ARIMA-style  (no-IV + IV-only): {rmse_total_arima:.4f}")
print(f"  ARIMAX-style (macro  + macro+IV): {rmse_total_arimax:.4f}")
print(f"\n  >>> Лучший RMSEtotal: {min(rmse_total_arima, rmse_total_arimax):.4f}")

print("\n" + "="*60)
print("Подобранные порядки после grid-search")
print("="*60)
for beta in BETA_COLS:
    print(f"  {beta}: {best_orders.get(beta)}")


# =========================================================
# 2.13 ВИЗУАЛИЗАЦИЯ
# =========================================================

fig, axes = plt.subplots(len(BETA_COLS), 1, figsize=(13, 9), sharex=True)
fig.suptitle("Прогноз бет: ARIMA (grid-search + clip) vs Факт", fontsize=13)

for ax, beta in zip(axes, BETA_COLS):
    hist  = df_b_trainval[beta]
    fact  = df_b_test[beta]
    fc_no = fc_betas_arima_no_iv[beta]
    fc_iv = fc_betas_arimax_iv_only[beta]

    ax.plot(hist.index, hist.values, "k-",   lw=1,   label="История")
    ax.plot(fact.index, fact.values, "ko",   ms=5,   label="Факт (тест)")
    ax.plot(fc_no.index, fc_no.values, "b--", lw=1.5, label="ARIMA no-IV")
    ax.plot(fc_iv.index, fc_iv.values, "r--", lw=1.5, label="ARIMAX IV-only")

    mu  = hist.mean()
    std = hist.std()
    ax.axhline(mu + CLIP_SIGMA*std, color="gray", ls=":", lw=0.8, alpha=0.7, label=f"±{CLIP_SIGMA}σ clip")
    ax.axhline(mu - CLIP_SIGMA*std, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_ylabel(beta)
    ax.legend(fontsize=7, ncol=5)

axes[-1].set_xlabel("Дата")
plt.tight_layout()
plt.savefig("data/ns_results/arima_forecast_betas.png", dpi=150)
plt.show()
print("\nГрафик сохранён: data/ns_results/arima_forecast_betas.png")
