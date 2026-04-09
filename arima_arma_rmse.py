import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# =========================================================
# 2.1 ПАРАМЕТРЫ
# =========================================================

BETAS_PATH = "data/ns_results/betas_0_7308.csv"
YC_PATH = "data/inputs/yield_curve.xlsx"
MACRO_PATH = "data/inputs/macro_updated.xlsx"
IV_PATH = "Problem_1_IV_train.xlsx"

START_DATE_BETAS = "2019-03-01"
FREQ = "MS"

BETA_COLS = ["beta0", "beta1", "beta2"]
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

TEST_SIZE = 6
INITIAL_TRAIN_SIZE = 48

ARIMA_ORDER = (2, 0, 0)
ARIMAX_ORDER = (2, 0, 0)


# =========================================================
# 2.2 ВСПОМОГАТЕЛЬНЫЕ
# =========================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def choose_d_by_adf(series):
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

# агрегация IV по дате
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
# 2.4 НЕЛЬСОН–ЗИГЕЛЬ
# =========================================================

def ns_loadings(tau, lam):
    x = tau / lam
    if np.isclose(x, 0.0):
        l1 = 1.0
        l2 = 0.0
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
# 2.5 СОБИРАЕМ ALL DATAFRAME
# =========================================================

beta_cols = [c for c in BETA_COLS if c in df_betas.columns]
yc_cols = [c for c in YIELD_COLS if c in df_yc.columns]
macro_cols = [c for c in MACRO_COLS_CANDIDATE if c in df_macro.columns]

common_idx = (
    df_betas.index
    .intersection(df_yc.index)
    .intersection(df_macro.index)
    .sort_values()
)

df_betas = df_betas.loc[common_idx, beta_cols].replace([np.inf, -np.inf], np.nan).dropna()
df_yc = df_yc.loc[common_idx, yc_cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
df_macro = df_macro.loc[common_idx, macro_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()

# объединяем с IV
df_all = df_macro.join(df_iv[["iv_mean"]], how="left")
df_all["iv_mean"] = df_all["iv_mean"].ffill()

macro_cols_with_iv = macro_cols + ["iv_mean"]

train_idx = common_idx[:-TEST_SIZE]
test_idx = common_idx[-TEST_SIZE:]

df_b_train = df_betas.loc[train_idx]
df_b_test = df_betas.loc[test_idx]

df_m_train = df_all.loc[train_idx, macro_cols].copy()
df_m_test = df_all.loc[test_idx, macro_cols].copy()
df_iv_train = df_all.loc[train_idx, ["iv_mean"]].copy()
df_iv_test = df_all.loc[test_idx, ["iv_mean"]].copy()

# =========================================================
# 2.6 ФУНКЦИИ МЕТРИКИ И СЧЕТА (на базе твоего описания)
# =========================================================

def compute_weighted_rmse_curve(
    df_yc_actual: pd.DataFrame,
    df_yc_forecast: pd.DataFrame,
    cols=YIELD_COLS,
    w_on=0.4,
    w_rest_total=0.6
):
    """
    RMSE по кривой доходности по 6 тестовым датам,
    где ON имеет вес 0.4, остальные теноры равномерно 0.6.
    """
    if not set(cols).issubset(df_yc_actual.columns) or \
       not set(cols).issubset(df_yc_forecast.columns):
        return np.nan

    n_tenors = len(cols)
    on_idx = cols[0] if "ON" in cols else None
    if on_idx is None:
        raise ValueError("ON должно быть первым тенором или в cols")

    if n_tenors <= 1:
        return np.nan

    weight = {on_idx: w_on}
    w_rest = w_rest_total / (n_tenors - 1)
    for col in cols:
        if col != on_idx:
            weight[col] = w_rest

    diffs = (df_yc_actual[cols] - df_yc_forecast[cols]).dropna(how="all")

    if diffs.shape[0] == 0:
        return np.nan

    # вычисляем взвешенную RMSE
    sq_err_weighted = 0.0
    total_weight = 0.0
    for col in cols:
        if col not in weight:
            continue
        w = weight[col]
        mask = diffs[col].notna()
        if mask.sum() == 0:
            continue
        sq_err = ((diffs.loc[mask, col]) ** 2).sum()
        sq_err_weighted += w * sq_err
        total_weight += w * mask.sum()

    if total_weight <= 0:
        return np.nan

    mse_weighted = sq_err_weighted / total_weight
    return np.sqrt(mse_weighted)


def compute_rmse_total_score(rmse_m1, rmse_m2, rmse_simple_bm, rmse_adv_bm, delta=0.5):
    """
    Вычисляет итоговый RMSEtotal и Score по формуле из задания.
    """
    if np.isnan(rmse_m1) or np.isnan(rmse_m2):
        return np.nan, np.nan

    rmse_m1 = np.maximum(rmse_m1, 1e-8)
    rmse_m2 = np.maximum(rmse_m2, 1e-8)

    rmse_total = delta * rmse_m1 + (1.0 - delta) * rmse_m2

    if rmse_simple_bm <= 0:
        return rmse_total, np.nan

    i_i = 1.0 - rmse_total / rmse_simple_bm

    if rmse_adv_bm > 0 and rmse_simple_bm > 0:
        alpha = max(
            1.0 - rmse_adv_bm / rmse_simple_bm,
            1.0 - rmse_total / rmse_simple_bm
        )
    else:
        alpha = 1.0

    score = 200.0 * min(1.0, max(0.0, i_i))
    if alpha > 0:
        score = min(score, 200.0 * (1.0 / alpha))

    return rmse_total, score


# =========================================================
# 2.7 МОДЕЛЬ: ARIMA (без IV)
# =========================================================

class ARIMA_Model:
    def __init__(self, beta_cols, arima_order):
        self.beta_cols = beta_cols
        self.arima_order = arima_order
        self.models = {}
        self.used_orders = {}
        self.train_series = {}

    def fit(self, df_b):
        for beta in self.beta_cols:
            y = df_b[beta].dropna()
            self.train_series[beta] = y.copy()

            if len(y) < 10:
                self.models[beta] = None
                self.used_orders[beta] = None
                continue

            d = choose_d_by_adf(y)
            order = (self.arima_order[0], d, self.arima_order[2])
            try:
                model = ARIMA(y, order=order)
                self.models[beta] = model.fit()
                self.used_orders[beta] = order
            except Exception:
                self.models[beta] = None
                self.used_orders[beta] = None

        return self

    def forecast(self, steps, future_index):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)
        for beta in self.beta_cols:
            mod = self.models.get(beta)
            if mod is None:
                out[beta] = np.nan
                continue
            try:
                out[beta] = np.asarray(mod.forecast(steps=steps))
            except Exception:
                out[beta] = np.nan
        return out


# ---------------------------------------------------------
# 2.7.1 ARIMA без IV
# ---------------------------------------------------------

model_arima_no_iv = ARIMA_Model(beta_cols=BETA_COLS, arima_order=ARIMA_ORDER)
model_arima_no_iv.fit(df_b_train)

fc_betas_arima_no_iv = model_arima_no_iv.forecast(
    steps=TEST_SIZE,
    future_index=test_idx
)

yc_actual = reconcile_yc_from_betas(df_b_test, maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS})

yc_forecast_arima_no_iv = reconcile_yc_from_betas(
    df_betas=fc_betas_arima_no_iv,
    maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS}
)

rmse_arima_no_iv = compute_weighted_rmse_curve(
    df_yc_actual=yc_actual,
    df_yc_forecast=yc_forecast_arima_no_iv
)


# ---------------------------------------------------------
# 2.7.2 ARIMA с IV (как exog)
# ---------------------------------------------------------
# Внимание: ARIMA允许 exog, но тут логичнее ARIMAX; поэтому ARIMA с IV — просто
# как демонстрация, реально лучше использовать 2.8 ниже.

model_arima_iv = ARIMA_Model(beta_cols=BETA_COLS, arima_order=ARIMA_ORDER)
model_arima_iv.fit(pd.concat([df_b_train, df_iv_train], axis=1))

fc_betas_arima_iv = model_arima_iv.forecast(
    steps=TEST_SIZE,
    future_index=test_idx
)

yc_forecast_arima_iv = reconcile_yc_from_betas(
    df_betas=fc_betas_arima_iv,
    maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS}
)

rmse_arima_iv = compute_weighted_rmse_curve(
    df_yc_actual=yc_actual,
    df_yc_forecast=yc_forecast_arima_iv
)


# =========================================================
# 2.8 МОДЕЛЬ: ARIMAX (без IV и с IV)
# =========================================================

class ARIMAX_Model:
    def __init__(self, beta_cols, macro_cols, arimax_order):
        self.beta_cols = beta_cols
        self.macro_cols = macro_cols
        self.arimax_order = arimax_order
        self.models = {}
        self.used_orders = {}
        self.train_series = {}
        self.train_X = {}

    def _clean_exog(self, X):
        if X is None or len(X) == 0:
            return pd.DataFrame()

        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        X = X.dropna(axis=1, how="all")

        if X.shape[1] == 0:
            return X

        X = X.ffill().bfill()

        std = X.std(ddof=0)
        keep_cols = std[(std > 1e-12) | std.isna()].index.tolist()
        X = X[keep_cols]

        X = X.fillna(0.0)
        return X

    def fit(self, df_b, df_exog):
        for beta in self.beta_cols:
            try:
                if beta not in df_b.columns:
                    self.models[beta] = None
                    self.used_orders[beta] = None
                    self.train_series[beta] = None
                    self.train_X[beta] = None
                    continue

                y = df_b[beta].copy()
                X = df_exog.copy()

                common_idx = y.index.intersection(X.index).sort_values()
                y = y.loc[common_idx]
                X = X.loc[common_idx]

                y = pd.to_numeric(y, errors="coerce")
                X = self._clean_exog(X)

                tmp = pd.concat([y.rename(beta), X], axis=1)
                tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()

                if tmp.shape[0] < 10:
                    self.models[beta] = None
                    self.used_orders[beta] = None
                    self.train_series[beta] = None
                    self.train_X[beta] = None
                    continue

                y1 = tmp[beta].astype(float)
                X1 = tmp.drop(columns=[beta]).astype(float)

                if X1.shape[1] == 0:
                    X1 = None

                d = choose_d_by_adf(y1)
                order = (self.arimax_order[0], d, self.arimax_order[2])

                if X1 is None:
                    model = ARIMA(y1, order=order)
                    res = model.fit()
                    self.train_X[beta] = None
                else:
                    model = ARIMA(y1, exog=X1, order=order)
                    res = model.fit()
                    self.train_X[beta] = X1.copy()

                self.models[beta] = res
                self.used_orders[beta] = order
                self.train_series[beta] = y1.copy()

            except Exception:
                self.models[beta] = None
                self.used_orders[beta] = None
                self.train_series[beta] = None
                self.train_X[beta] = None

        return self

    def forecast(self, steps, future_index, df_exog_future):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)

        for beta in self.beta_cols:
            out[beta] = np.nan

            res = self.models.get(beta)
            if res is None:
                continue

            try:
                Xtrain = self.train_X.get(beta)

                if Xtrain is None:
                    pred = res.forecast(steps=steps)
                    out[beta] = np.asarray(pred, dtype=float)
                    continue

                Xf = df_exog_future.copy()
                Xf = Xf.loc[future_index]
                Xf = self._clean_exog(Xf)

                Xf = Xf.reindex(columns=Xtrain.columns)

                for c in Xf.columns:
                    if Xf[c].isna().all():
                        Xf[c] = Xtrain[c].iloc[-1] if c in Xtrain.columns else 0.0

                Xf = Xf.ffill().bfill().fillna(0.0)

                if Xf.shape[0] != steps:
                    Xf = Xf.reindex(future_index)
                    Xf = Xf.ffill().bfill().fillna(0.0)

                pred = res.forecast(steps=steps, exog=Xf)
                out[beta] = np.asarray(pred, dtype=float)

            except Exception:
                out[beta] = np.nan

        return out

# ---------------------------------------------------------
# 2.8.1 ARIMAX без IV (только макро)
# ---------------------------------------------------------

model_arimax_no_iv = ARIMAX_Model(
    beta_cols=BETA_COLS,
    macro_cols=macro_cols,
    arimax_order=ARIMAX_ORDER
)
model_arimax_no_iv.fit(df_b_train, df_m_train)

fc_betas_arimax_no_iv = model_arimax_no_iv.forecast(
    steps=TEST_SIZE,
    future_index=test_idx,
    df_exog_future=df_m_test
)

yc_forecast_arimax_no_iv = reconcile_yc_from_betas(
    df_betas=fc_betas_arimax_no_iv,
    maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS}
)

rmse_arimax_no_iv = compute_weighted_rmse_curve(
    df_yc_actual=yc_actual,
    df_yc_forecast=yc_forecast_arimax_no_iv
)




# ---------------------------------------------------------
# 2.8.2 ARIMAX с IV (макро + IV)
# ---------------------------------------------------------

# собираем exog с IV для ARIMAX
X_train_with_iv = pd.concat([df_m_train, df_iv_train], axis=1)
X_test_with_iv = pd.concat([df_m_test, df_iv_test], axis=1)

model_arimax_iv = ARIMAX_Model(
    beta_cols=BETA_COLS,
    macro_cols=macro_cols_with_iv,
    arimax_order=ARIMAX_ORDER
)
model_arimax_iv.fit(df_b_train, X_train_with_iv)

fc_betas_arimax_iv = model_arimax_iv.forecast(
    steps=TEST_SIZE,
    future_index=test_idx,
    df_exog_future=X_test_with_iv
)

yc_forecast_arimax_iv = reconcile_yc_from_betas(
    df_betas=fc_betas_arimax_iv,
    maturity_map={k: MATURITY_MAP[k] for k in YIELD_COLS}
)

rmse_arimax_iv = compute_weighted_rmse_curve(
    df_yc_actual=yc_actual,
    df_yc_forecast=yc_forecast_arimax_no_iv
)

# =========================================================
# 2.9 ВЫВОД ОШИБОК ПО ВСЕМ ВАРИАНТАМ
# =========================================================

results = {
    "ARIMA without IV": rmse_arima_no_iv,
    "ARIMA with IV": rmse_arima_iv,
    "ARIMAX without IV": rmse_arimax_no_iv,
    "ARIMAX with IV": rmse_arimax_iv
}

print("\n" + "="*60)
print("RMSE по кривой (ON–2Y, вся выборка)")
print("="*60)

for name, val in results.items():
    if np.isnan(val):
        print(f"{name:<25}: N/A")
    else:
        print(f"{name:<25}: {val:.4f}")

# =========================================================
# 2.7c. ИТОГОВАЯ МЕТРИКА RMSEtotal (без Score, без бенчмарков)
# =========================================================

delta = 0.5
rmse_m1 = rmse_arimax_no_iv
rmse_m2 = rmse_arimax_iv

if np.isnan(rmse_m1) or np.isnan(rmse_m2):
    rmse_total = np.nan
else:
    rmse_total = delta * rmse_m1 + (1.0 - delta) * rmse_m2

print("\n" + "="*60)
print("RMSE по кривой (ON–2Y, вся выборка)")
print("="*60)
print(f"ARIMA without IV         : {rmse_arima_no_iv:.4f}")
print(f"ARIMA with IV            : {rmse_arima_iv:.4f}")
print(f"ARIMAX without IV        : {rmse_arimax_no_iv:.4f}")
print(f"ARIMAX with IV           : {rmse_arimax_iv:.4f}")
print("\n" + "-"*60)
print("Итоговая метрика (M1 = ARIMA, M2 = ARIMAX)")
print("-"*60)
print(f"RMSEtotal (delta=0.5)    : {rmse_total:.4f}")