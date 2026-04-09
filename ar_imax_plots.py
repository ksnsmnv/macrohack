import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


# =========================================================
# 0. ПАРАМЕТРЫ
# =========================================================

BETAS_PATH = "data/ns_results/betas_0_7308.csv"
MACRO_PATH = "data/inputs/macro_updated.xlsx"
YC_PATH = "data/inputs/yield_curve.xlsx"

START_DATE_BETAS = "2019-03-01"
FREQ = "MS"

FORECAST_HORIZON = 6
TEST_SIZE = 6

BETA_COLS = ["beta0", "beta1", "beta2"]
MACRO_COLS_CANDIDATE = [
    "cbr", "inf", "observed_inf", "expected_inf",
    "usd", "moex", "brent", "vix"
]
ARIMAX_ORDER_DEFAULT = (2, 0, 0)
CI_ALPHA = 0.05

# ТО, ЧТО У ВАС В ДАННЫХ: lambda = 0.7308
NS_LAMBDA = 0.7308

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


# =========================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def choose_d_by_adf(series):
    try:
        p_value = adfuller(series.dropna())[1]
        return 0 if p_value < 0.05 else 1
    except Exception:
        return 1


def normalize_month_index(idx):
    idx = pd.to_datetime(idx)
    return idx.to_period("M").to_timestamp("M")


def load_betas():
    df_betas = pd.read_csv(BETAS_PATH)
    df_betas["date"] = pd.date_range(
        start=START_DATE_BETAS,
        periods=len(df_betas),
        freq=FREQ
    )
    df_betas = df_betas.set_index("date").sort_index()

    for col in df_betas.columns:
        df_betas[col] = pd.to_numeric(df_betas[col], errors="coerce")

    df_betas.index = normalize_month_index(df_betas.index)
    return df_betas


def load_macro():
    df_macro = pd.read_excel(MACRO_PATH)

    date_col = None
    for col in df_macro.columns:
        if str(col).lower() in ["date", "dt", "month", "period", "дата"]:
            date_col = col
            break

    if date_col is None:
        raise ValueError("В macro_updated.xlsx не найден столбец даты.")

    if date_col != "date":
        df_macro = df_macro.rename(columns={date_col: "date"})

    df_macro["date"] = pd.to_datetime(df_macro["date"])
    df_macro = df_macro.set_index("date").sort_index()

    for col in df_macro.columns:
        df_macro[col] = pd.to_numeric(df_macro[col], errors="coerce")

    df_macro.index = normalize_month_index(df_macro.index)
    return df_macro


def load_yield_curve():
    df_yc = pd.read_excel(YC_PATH)

    if "Month" not in df_yc.columns:
        raise ValueError("В yield_curve.xlsx не найден столбец 'Month'.")

    df_yc = df_yc.rename(columns={"Month": "date"})
    df_yc["date"] = pd.to_datetime(df_yc["date"])
    df_yc = df_yc.set_index("date").sort_index()

    # Приводим строки вида '17,35' к float
    for col in df_yc.columns:
        df_yc[col] = (
            df_yc[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )
        df_yc[col] = pd.to_numeric(df_yc[col], errors="coerce")

    df_yc.index = normalize_month_index(df_yc.index)
    return df_yc


def forecast_macro_naive(df_macro_train, future_index, macro_cols):
    last_row = df_macro_train[macro_cols].iloc[-1]
    future_macro = pd.DataFrame(
        np.tile(last_row.values, (len(future_index), 1)),
        index=future_index,
        columns=macro_cols
    )
    return future_macro


# =========================================================
# 2. NELSON-SIEGEL С YOUR LAMBDA
# =========================================================

def ns_loadings(tau, lam):
    x = tau / lam
    if np.isclose(x, 0):
        l1 = 1.0
        l2 = 0.0
    else:
        l1 = (1 - np.exp(-x)) / x
        l2 = l1 - np.exp(-x)
    return l1, l2


def nelson_siegel_yield(tau, beta0, beta1, beta2, lam):
    l1, l2 = ns_loadings(tau, lam)
    return beta0 + beta1 * l1 + beta2 * l2


def reconstruct_yield_curve_from_betas(df_betas, maturity_map, lam):
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


def reconstruct_yield_curve_ci(df_beta_mean, df_beta_lower, df_beta_upper, maturity_map, lam):
    yc_mean = reconstruct_yield_curve_from_betas(df_beta_mean, maturity_map, lam)
    yc_lower = reconstruct_yield_curve_from_betas(df_beta_lower, maturity_map, lam)
    yc_upper = reconstruct_yield_curve_from_betas(df_beta_upper, maturity_map, lam)

    yc_lo = pd.DataFrame(index=yc_mean.index, columns=yc_mean.columns, dtype=float)
    yc_hi = pd.DataFrame(index=yc_mean.index, columns=yc_mean.columns, dtype=float)

    for c in yc_mean.columns:
        stacked = np.vstack([yc_lower[c].values, yc_upper[c].values])
        yc_lo[c] = stacked.min(axis=0)
        yc_hi[c] = stacked.max(axis=0)

    return yc_mean, yc_lo, yc_hi


# =========================================================
# 3. MODEL: ARIMAX НА BETA
# =========================================================

class Model_ARIMAX:
    def __init__(self, beta_cols, macro_cols, order_default=(2, 0, 0)):
        self.beta_cols = beta_cols
        self.macro_cols = macro_cols
        self.order_default = order_default

        self.models = {}
        self.train_series = {}
        self.train_exog = {}
        self.used_orders = {}

    def fit(self, df_betas, df_macro):
        for beta in self.beta_cols:
            y = df_betas[beta].copy()
            X = df_macro[self.macro_cols].copy()

            common_idx = y.dropna().index.intersection(X.dropna().index)
            y = y.loc[common_idx]
            X = X.loc[common_idx]

            self.train_series[beta] = y.copy()
            self.train_exog[beta] = X.copy()

            if len(y) < 12:
                self.models[beta] = None
                self.used_orders[beta] = None
                continue

            try:
                d = choose_d_by_adf(y)
                order = (self.order_default[0], d, self.order_default[2])
                model = ARIMA(y, exog=X, order=order)
                self.models[beta] = model.fit()
                self.used_orders[beta] = order
            except Exception:
                self.models[beta] = None
                self.used_orders[beta] = None

        return self

    def forecast_with_ci(self, future_index, future_macro, alpha=0.05):
        steps = len(future_index)

        mean_df = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)
        lower_df = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)
        upper_df = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)

        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                continue

            try:
                exog_future = future_macro.loc[future_index, self.macro_cols]
                pred_res = model.get_forecast(steps=steps, exog=exog_future)

                mean_df[beta] = np.asarray(pred_res.predicted_mean)

                ci = pred_res.conf_int(alpha=alpha)
                lower_df[beta] = np.asarray(ci.iloc[:, 0])
                upper_df[beta] = np.asarray(ci.iloc[:, 1])

            except Exception:
                mean_df[beta] = np.nan
                lower_df[beta] = np.nan
                upper_df[beta] = np.nan

        return mean_df, lower_df, upper_df

    def train_rmse_beta(self):
        rmse_train = {}
        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                rmse_train[beta] = np.inf
                continue
            try:
                y_true = self.train_series[beta].values
                y_pred = np.asarray(model.predict(exog=self.train_exog[beta]))
                n = min(len(y_true), len(y_pred))
                rmse_train[beta] = rmse(y_true[-n:], y_pred[-n:]) if n > 1 else np.inf
            except Exception:
                rmse_train[beta] = np.inf
        return rmse_train


# =========================================================
# 4. ЗАГРУЗКА И РАЗДЕЛЕНИЕ НА TRAIN/TEST
# =========================================================

df_betas = load_betas()
df_macro = load_macro()
df_yc = load_yield_curve()

beta_cols = [c for c in BETA_COLS if c in df_betas.columns]
macro_cols = [c for c in MACRO_COLS_CANDIDATE if c in df_macro.columns]
yc_cols = [c for c in MATURITY_MAP.keys() if c in df_yc.columns]

if len(beta_cols) != 3:
    raise ValueError(f"Ожидались beta0,beta1,beta2, найдено: {beta_cols}")
if len(macro_cols) == 0:
    raise ValueError("Не найдены макро-столбцы.")
if len(yc_cols) == 0:
    raise ValueError("Не найдены сроки доходностей в yield_curve.xlsx.")

common_idx = df_betas.index.intersection(df_macro.index).intersection(df_yc.index)
common_idx = common_idx.sort_values()

df_betas = df_betas.loc[common_idx, beta_cols].copy()
df_macro = df_macro.loc[common_idx, macro_cols].copy()
df_yc = df_yc.loc[common_idx, yc_cols].copy()

df_betas = df_betas.replace([np.inf, -np.inf], np.nan).dropna()
df_macro = df_macro.replace([np.inf, -np.inf], np.nan).ffill().bfill()
df_yc = df_yc.replace([np.inf, -np.inf], np.nan).dropna(how="all")

common_idx = df_betas.index.intersection(df_macro.index).intersection(df_yc.index)
common_idx = common_idx.sort_values()

df_betas = df_betas.loc[common_idx]
df_macro = df_macro.loc[common_idx]
df_yc = df_yc.loc[common_idx]

if len(common_idx) <= TEST_SIZE + 12:
    raise ValueError("Слишком мало наблюдений для hold-out оценки.")

train_idx = common_idx[:-TEST_SIZE]
test_idx = common_idx[-TEST_SIZE:]

df_b_train = df_betas.loc[train_idx]
df_m_train = df_macro.loc[train_idx]
df_yc_train = df_yc.loc[train_idx]

df_b_test = df_betas.loc[test_idx]
df_m_test = df_macro.loc[test_idx]
df_yc_test = df_yc.loc[test_idx]

print("=" * 80)
print("ARIMAX НА BETA + ОЦЕНКА ЧЕРЕЗ ИСХОДНЫЕ КРИВЫЕ ДОХОДНОСТИ")
print("Используем NS_LAMBDA =", NS_LAMBDA)
print("=" * 80)
print(f"Train: {train_idx[0].strftime('%Y-%m-%d')} - {train_idx[-1].strftime('%Y-%m-%d')} ({len(train_idx)} мес)")
print(f"Test : {test_idx[0].strftime('%Y-%m-%d')} - {test_idx[-1].strftime('%Y-%m-%d')} ({len(test_idx)} мес)")
print("beta_cols :", beta_cols)
print("macro_cols:", macro_cols)
print("yc_cols   :", yc_cols)


# =========================================================
# 5. ОБУЧЕНИЕ ARIMAX И ПРОГНОЗ BETA + CI
# =========================================================

future_macro = forecast_macro_naive(df_m_train, test_idx, macro_cols)

model = Model_ARIMAX(
    beta_cols=beta_cols,
    macro_cols=macro_cols,
    order_default=ARIMAX_ORDER_DEFAULT
)
model.fit(df_b_train, df_m_train)

df_beta_mean, df_beta_lower, df_beta_upper = model.forecast_with_ci(
    future_index=test_idx,
    future_macro=future_macro,
    alpha=CI_ALPHA
)

print("\nИспользованные порядки ARIMAX:")
for b, order in model.used_orders.items():
    print(f"{b}: {order}")


# =========================================================
# 6. RMSE ПО BETA
# =========================================================

rmse_beta_test = {}
for beta in beta_cols:
    rmse_beta_test[beta] = rmse(
        df_b_test[beta].values,
        df_beta_mean[beta].values
    )

avg_rmse_beta_test = float(np.mean(list(rmse_beta_test.values())))

print("\nRMSE по beta:")
for beta, val in rmse_beta_test.items():
    print(f"{beta:10s} | RMSE = {val:8.4f}")
print(f"{'AVG':10s} | RMSE = {avg_rmse_beta_test:8.4f}")


# =========================================================
# 7. ВОССТАНОВЛЕНИЕ YIELD CURVE И RMSE ПО ИСХОДНЫМ СТАВКАМ
# =========================================================

df_yc_pred_mean, df_yc_pred_lower, df_yc_pred_upper = reconstruct_yield_curve_ci(
    df_beta_mean,
    df_beta_lower,
    df_beta_upper,
    maturity_map={k: v for k, v in MATURITY_MAP.items() if k in yc_cols},
    lam=NS_LAMBDA
)

rmse_yc_by_maturity = {}
for col in yc_cols:
    mask = (~df_yc_test[col].isna()) & (~df_yc_pred_mean[col].isna())
    if mask.sum() > 0:
        rmse_yc_by_maturity[col] = rmse(
            df_yc_test.loc[mask, col].values,
            df_yc_pred_mean.loc[mask, col].values
        )
    else:
        rmse_yc_by_maturity[col] = np.nan

all_true = []
all_pred = []
for col in yc_cols:
    mask = (~df_yc_test[col].isna()) & (~df_yc_pred_mean[col].isna())
    all_true.extend(df_yc_test.loc[mask, col].values.tolist())
    all_pred.extend(df_yc_pred_mean.loc[mask, col].values.tolist())

overall_yc_rmse = rmse(np.array(all_true), np.array(all_pred))

print("\nRMSE по исходным доходностям curve:")
for col, val in rmse_yc_by_maturity.items():
    print(f"{col:10s} | RMSE = {val:8.4f}")
print(f"{'OVERALL':10s} | RMSE = {overall_yc_rmse:8.4f}")


# =========================================================
# 8. ГРАФИКИ BETA С ДОВЕРИТЕЛЬНЫМИ ИНТЕРВАЛАМИ
# =========================================================

for beta in beta_cols:
    plt.figure(figsize=(12, 5))

    x_train = df_b_train.index.to_pydatetime()
    x_test = df_b_test.index.to_pydatetime()
    x_fcst = df_beta_mean.index.to_pydatetime()

    plt.plot(
        x_train,
        df_b_train[beta].values,
        label="Train",
        color="steelblue",
        linewidth=2
    )

    plt.plot(
        x_test,
        df_b_test[beta].values,
        label="Actual test",
        color="black",
        linewidth=2
    )

    plt.plot(
        x_fcst,
        df_beta_mean[beta].values,
        label="Forecast",
        color="crimson",
        linewidth=2
    )

    plt.fill_between(
        x_fcst,
        df_beta_lower[beta].values.astype(float),
        df_beta_upper[beta].values.astype(float),
        color="crimson",
        alpha=0.18,
        label=f"{int((1 - CI_ALPHA) * 100)}% CI"
    )

    plt.axvline(
        x_test[0],
        color="gray",
        linestyle="--",
        linewidth=1
    )

    plt.title(f"ARIMAX forecast for {beta} with confidence interval")
    plt.xlabel("Date")
    plt.ylabel(beta)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot/forecast_{beta}_with_ci.png", dpi=160)
    plt.show()


# =========================================================
# 9. ГРАФИКИ КРИВЫХ ДОХОДНОСТИ НА TEST-ГОРИЗОНТЕ
# =========================================================

maturity_order = [c for c in MATURITY_MAP.keys() if c in yc_cols]

for dt in test_idx:
    plt.figure(figsize=(10, 5))

    x = np.arange(len(maturity_order))

    actual_vals = df_yc_test.loc[dt, maturity_order].values.astype(float)
    pred_vals = df_yc_pred_mean.loc[dt, maturity_order].values.astype(float)
    low_vals = df_yc_pred_lower.loc[dt, maturity_order].values.astype(float)
    up_vals = df_yc_pred_upper.loc[dt, maturity_order].values.astype(float)

    plt.plot(
        x,
        actual_vals,
        marker="o",
        color="black",
        linewidth=2,
        label="Actual YC"
    )

    plt.plot(
        x,
        pred_vals,
        marker="o",
        color="crimson",
        linewidth=2,
        label="Forecast YC"
    )

    plt.fill_between(
        x,
        low_vals,
        up_vals,
        color="crimson",
        alpha=0.15,
        label="Forecast band"
    )

    plt.xticks(x, maturity_order)
    plt.title(f"Yield curve: actual vs forecast | {dt.strftime('%Y-%m-%d')}")
    plt.xlabel("Maturity")
    plt.ylabel("Yield")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot/yield_curve_forecast_{dt.strftime('%Y_%m_%d')}.png", dpi=160)
    plt.show()


# =========================================================
# 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================================================

df_beta_mean.to_csv("forecast/arimax_beta_forecast_mean.csv")
# df_beta_lower.to_csv("forecast/arimax_beta_forecast_lower.csv")
# df_beta_upper.to_csv("forecast/arimax_beta_forecast_upper.csv")

df_yc_pred_mean.to_csv("forecast/arimax_yield_curve_forecast_mean.csv")
# df_yc_pred_lower.to_csv("forecast/arimax_yield_curve_forecast_lower.csv")
# df_yc_pred_upper.to_csv("forecast/arimax_yield_curve_forecast_upper.csv")

summary_beta = pd.DataFrame({
    "beta": list(rmse_beta_test.keys()) + ["AVG"],
    "rmse_beta": list(rmse_beta_test.values()) + [avg_rmse_beta_test]
})
summary_beta.to_csv("forecast/arimax_rmse_beta.csv", index=False)

summary_yc = pd.DataFrame({
    "maturity": list(rmse_yc_by_maturity.keys()) + ["OVERALL"],
    "rmse_yield_curve": list(rmse_yc_by_maturity.values()) + [overall_yc_rmse]
})
summary_yc.to_csv("forecast/arimax_rmse_yield_curve.csv", index=False)

print("\nФайлы сохранены:")
print("- arimax_beta_forecast_mean.csv")
print("- arimax_beta_forecast_lower.csv")
print("- arimax_beta_forecast_upper.csv")
print("- arimax_yield_curve_forecast_mean.csv")
print("- arimax_yield_curve_forecast_lower.csv")
print("- arimax_yield_curve_forecast_upper.csv")
print("- arimax_rmse_beta.csv")
print("- arimax_rmse_yield_curve.csv")
print("- PNG-графики по beta и по yield curves")

