import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller


# =========================================================
# 0. ПАРАМЕТРЫ
# =========================================================

BETAS_PATH = "data/ns_results/betas_0_7308.csv"
MACRO_PATH = "data/inputs/macro_updated.xlsx"

START_DATE_BETAS = "2019-03-01"
FREQ = "MS"

BETA_COLS = ["beta0", "beta1", "beta2"]
MACRO_COLS_CANDIDATE = ["cbr", "inf", "expected_inf", "observed_inf", "usd", "moex", "brent", "vix"]

FORECAST_HORIZON = 6

INITIAL_TRAIN_SIZE = 48
CV_HORIZON = 6
CV_STEP = 1

AR_LAG = 1
ARX_LAG = 1

ARIMA_ORDER_DEFAULT = (2, 0, 0)
ARIMAX_ORDER_DEFAULT = (2, 0, 0)


# =========================================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data():
    df_betas = pd.read_csv(BETAS_PATH)
    df_betas["date"] = pd.date_range(
        start=START_DATE_BETAS,
        periods=len(df_betas),
        freq=FREQ
    )
    df_betas = df_betas.set_index("date").sort_index()

    for col in df_betas.columns:
        df_betas[col] = pd.to_numeric(df_betas[col], errors="coerce")

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

    return df_betas, df_macro


def prepare_common_sample(df_betas, df_macro, beta_cols, macro_cols):
    beta_cols = [c for c in beta_cols if c in df_betas.columns]
    macro_cols = [c for c in macro_cols if c in df_macro.columns]

    if len(beta_cols) == 0:
        raise ValueError("Не найдены beta-столбцы.")
    if len(macro_cols) == 0:
        raise ValueError("Не найдены macro-столбцы.")

    df_b = df_betas[beta_cols].copy()
    df_m = df_macro[macro_cols].copy()

    common_idx = df_b.index.intersection(df_m.index)
    df_b = df_b.loc[common_idx].sort_index()
    df_m = df_m.loc[common_idx].sort_index()

    df_b = df_b.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df_m = df_m.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    valid_idx = df_b.dropna().index.intersection(df_m.dropna().index)
    df_b = df_b.loc[valid_idx]
    df_m = df_m.loc[valid_idx]

    return df_b, df_m, beta_cols, macro_cols


def expanding_window_splits(index, initial_train_size, horizon, step=1):
    n = len(index)
    splits = []

    train_end = initial_train_size
    while train_end + horizon <= n:
        train_idx = index[:train_end]
        test_idx = index[train_end:train_end + horizon]
        splits.append((train_idx, test_idx))
        train_end += step

    return splits


def forecast_macro_naive(df_macro_train, future_index, macro_cols):
    last_row = df_macro_train[macro_cols].iloc[-1]
    future_macro = pd.DataFrame(
        np.tile(last_row.values, (len(future_index), 1)),
        index=future_index,
        columns=macro_cols
    )
    return future_macro


def choose_d_by_adf(series):
    try:
        p_value = adfuller(series.dropna())[1]
        return 0 if p_value < 0.05 else 1
    except Exception:
        return 1


# =========================================================
# 2. MODEL 1: AR(1)
# =========================================================

class Model_AR1:
    def __init__(self, beta_cols, lag=1):
        self.beta_cols = beta_cols
        self.lag = lag
        self.models = {}
        self.train_series = {}

    def fit(self, df_betas):
        for beta in self.beta_cols:
            series = df_betas[beta].dropna()
            self.train_series[beta] = series.copy()

            if len(series) < (self.lag + 5):
                self.models[beta] = None
                continue

            try:
                model = AutoReg(series, lags=self.lag, trend="c", old_names=False)
                self.models[beta] = model.fit()
            except Exception:
                self.models[beta] = None
        return self

    def forecast(self, steps, future_index):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)

        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                out[beta] = np.nan
            else:
                try:
                    fc = model.predict(
                        start=len(self.train_series[beta]),
                        end=len(self.train_series[beta]) + steps - 1,
                        dynamic=False
                    )
                    out[beta] = np.asarray(fc)
                except Exception:
                    out[beta] = np.nan
        return out

    def train_rmse(self):
        rmse_train = {}
        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                rmse_train[beta] = np.inf
                continue
            try:
                y_true = self.train_series[beta].values[self.lag:]
                y_pred = np.asarray(model.fittedvalues)
                n = min(len(y_true), len(y_pred))
                rmse_train[beta] = rmse(y_true[-n:], y_pred[-n:]) if n > 1 else np.inf
            except Exception:
                rmse_train[beta] = np.inf
        return rmse_train


# =========================================================
# 3. MODEL 2: ARX
# =========================================================

class Model_ARX:
    def __init__(self, beta_cols, macro_cols, lag=1):
        self.beta_cols = beta_cols
        self.macro_cols = macro_cols
        self.lag = lag
        self.models = {}
        self.train_series = {}
        self.train_exog = {}

    def fit(self, df_betas, df_macro):
        for beta in self.beta_cols:
            y = df_betas[beta].copy()
            X = df_macro[self.macro_cols].copy()

            common_idx = y.dropna().index.intersection(X.dropna().index)
            y = y.loc[common_idx]
            X = X.loc[common_idx]

            self.train_series[beta] = y.copy()
            self.train_exog[beta] = X.copy()

            if len(y) < (self.lag + 8):
                self.models[beta] = None
                continue

            try:
                model = AutoReg(
                    endog=y,
                    lags=self.lag,
                    exog=X,
                    trend="c",
                    old_names=False
                )
                self.models[beta] = model.fit()
            except Exception:
                self.models[beta] = None
        return self

    def forecast(self, steps, future_index, future_macro):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)

        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                out[beta] = np.nan
                continue

            try:
                exog_oos = future_macro.loc[future_index, self.macro_cols]
                fc = model.predict(
                    start=len(self.train_series[beta]),
                    end=len(self.train_series[beta]) + steps - 1,
                    dynamic=False,
                    exog_oos=exog_oos
                )
                out[beta] = np.asarray(fc)
            except Exception:
                out[beta] = np.nan
        return out

    def train_rmse(self):
        rmse_train = {}
        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                rmse_train[beta] = np.inf
                continue
            try:
                y_true = self.train_series[beta].values[self.lag:]
                y_pred = np.asarray(model.fittedvalues)
                n = min(len(y_true), len(y_pred))
                rmse_train[beta] = rmse(y_true[-n:], y_pred[-n:]) if n > 1 else np.inf
            except Exception:
                rmse_train[beta] = np.inf
        return rmse_train


# =========================================================
# 4. MODEL 3: ARIMA
# =========================================================

class Model_ARIMA:
    def __init__(self, beta_cols, order_default=(2, 0, 0)):
        self.beta_cols = beta_cols
        self.order_default = order_default
        self.models = {}
        self.train_series = {}
        self.used_orders = {}

    def fit(self, df_betas):
        for beta in self.beta_cols:
            series = df_betas[beta].dropna()
            self.train_series[beta] = series.copy()

            if len(series) < 12:
                self.models[beta] = None
                self.used_orders[beta] = None
                continue

            try:
                d = choose_d_by_adf(series)
                order = (self.order_default[0], d, self.order_default[2])
                model = ARIMA(series, order=order)
                self.models[beta] = model.fit()
                self.used_orders[beta] = order
            except Exception:
                self.models[beta] = None
                self.used_orders[beta] = None
        return self

    def forecast(self, steps, future_index):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)

        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                out[beta] = np.nan
            else:
                try:
                    out[beta] = np.asarray(model.forecast(steps=steps))
                except Exception:
                    out[beta] = np.nan
        return out

    def train_rmse(self):
        rmse_train = {}
        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                rmse_train[beta] = np.inf
                continue
            try:
                y_true = self.train_series[beta].values
                y_pred = np.asarray(model.predict())
                n = min(len(y_true), len(y_pred))
                rmse_train[beta] = rmse(y_true[-n:], y_pred[-n:]) if n > 1 else np.inf
            except Exception:
                rmse_train[beta] = np.inf
        return rmse_train


# =========================================================
# 5. MODEL 4: ARIMAX
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

    def forecast(self, steps, future_index, future_macro):
        out = pd.DataFrame(index=future_index, columns=self.beta_cols, dtype=float)

        for beta in self.beta_cols:
            model = self.models.get(beta, None)
            if model is None:
                out[beta] = np.nan
                continue

            try:
                exog_future = future_macro.loc[future_index, self.macro_cols]
                fc = model.forecast(steps=steps, exog=exog_future)
                out[beta] = np.asarray(fc)
            except Exception:
                out[beta] = np.nan
        return out

    def train_rmse(self):
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
# 6. ОЦЕНКА ОДНОГО ФОЛДА
# =========================================================

def evaluate_one_fold(model_name, df_b_train, df_m_train, df_b_test, df_m_test, beta_cols, macro_cols):
    if model_name == "AR1":
        model = Model_AR1(beta_cols=beta_cols, lag=AR_LAG)
        model.fit(df_b_train)
        forecast = model.forecast(steps=len(df_b_test), future_index=df_b_test.index)
        rmse_train_dict = model.train_rmse()

    elif model_name == "ARX":
        future_macro = forecast_macro_naive(df_m_train, df_b_test.index, macro_cols)
        model = Model_ARX(beta_cols=beta_cols, macro_cols=macro_cols, lag=ARX_LAG)
        model.fit(df_b_train, df_m_train)
        forecast = model.forecast(
            steps=len(df_b_test),
            future_index=df_b_test.index,
            future_macro=future_macro
        )
        rmse_train_dict = model.train_rmse()

    elif model_name == "ARIMA":
        model = Model_ARIMA(beta_cols=beta_cols, order_default=ARIMA_ORDER_DEFAULT)
        model.fit(df_b_train)
        forecast = model.forecast(steps=len(df_b_test), future_index=df_b_test.index)
        rmse_train_dict = model.train_rmse()

    elif model_name == "ARIMAX":
        future_macro = forecast_macro_naive(df_m_train, df_b_test.index, macro_cols)
        model = Model_ARIMAX(beta_cols=beta_cols, macro_cols=macro_cols, order_default=ARIMAX_ORDER_DEFAULT)
        model.fit(df_b_train, df_m_train)
        forecast = model.forecast(
            steps=len(df_b_test),
            future_index=df_b_test.index,
            future_macro=future_macro
        )
        rmse_train_dict = model.train_rmse()

    else:
        raise ValueError("Неизвестная модель")

    rmse_test_dict = {}
    for beta in beta_cols:
        y_true = df_b_test[beta].values
        y_pred = forecast.loc[df_b_test.index, beta].values
        rmse_test_dict[beta] = rmse(y_true, y_pred) if not np.isnan(y_pred).any() else np.inf

    avg_rmse_train = float(np.mean([v for v in rmse_train_dict.values() if np.isfinite(v)])) \
        if any(np.isfinite(v) for v in rmse_train_dict.values()) else np.inf

    avg_rmse_test = float(np.mean([v for v in rmse_test_dict.values() if np.isfinite(v)])) \
        if any(np.isfinite(v) for v in rmse_test_dict.values()) else np.inf

    overfit_gap = avg_rmse_test - avg_rmse_train
    overfit_pct = (overfit_gap / avg_rmse_train) if np.isfinite(avg_rmse_train) and avg_rmse_train > 0 else np.inf

    return forecast, rmse_train_dict, rmse_test_dict, avg_rmse_train, avg_rmse_test, overfit_gap, overfit_pct


# =========================================================
# 7. EXPANDING WINDOW CV
# =========================================================

def evaluate_models_expanding_cv(df_betas, df_macro, beta_cols, macro_cols,
                                 initial_train_size=48, horizon=6, step=1):
    print("\n" + "=" * 90)
    print("EXPANDING WINDOW CV: AR1 vs ARX vs ARIMA vs ARIMAX")
    print("=" * 90)

    common_idx = df_betas.index.intersection(df_macro.index).sort_values()
    splits = expanding_window_splits(common_idx, initial_train_size, horizon, step)

    if len(splits) == 0:
        raise ValueError("Недостаточно данных для expanding-window CV.")

    print(f"Количество фолдов: {len(splits)}")
    print(f"Initial train: {initial_train_size}, horizon: {horizon}, step: {step}")

    model_names = ["AR1", "ARX", "ARIMA", "ARIMAX"]
    fold_results = {m: [] for m in model_names}

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        print("\n" + "-" * 90)
        print(f"Fold {fold_num}/{len(splits)}")
        print(f"Train: {train_idx[0].strftime('%Y-%m')} - {train_idx[-1].strftime('%Y-%m')} ({len(train_idx)} мес)")
        print(f"Test : {test_idx[0].strftime('%Y-%m')} - {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} мес)")

        df_b_train = df_betas.loc[train_idx, beta_cols].copy()
        df_m_train = df_macro.loc[train_idx, macro_cols].copy()
        df_b_test = df_betas.loc[test_idx, beta_cols].copy()
        df_m_test = df_macro.loc[test_idx, macro_cols].copy()

        for model_name in model_names:
            try:
                forecast, rmse_train_dict, rmse_test_dict, avg_rmse_train, avg_rmse_test, overfit_gap, overfit_pct = \
                    evaluate_one_fold(
                        model_name=model_name,
                        df_b_train=df_b_train,
                        df_m_train=df_m_train,
                        df_b_test=df_b_test,
                        df_m_test=df_m_test,
                        beta_cols=beta_cols,
                        macro_cols=macro_cols
                    )

                fold_results[model_name].append({
                    "fold": fold_num,
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1],
                    "forecast": forecast,
                    "rmse_train": rmse_train_dict,
                    "rmse_test": rmse_test_dict,
                    "avg_rmse_train": avg_rmse_train,
                    "avg_rmse_test": avg_rmse_test,
                    "overfit_gap": overfit_gap,
                    "overfit_pct": overfit_pct
                })

                print(
                    f"{model_name:8s} | "
                    f"train RMSE = {avg_rmse_train:8.4f} | "
                    f"test RMSE = {avg_rmse_test:8.4f} | "
                    f"gap = {overfit_gap:8.4f} | "
                    f"gap% = {overfit_pct:8.2%}"
                )

            except Exception as e:
                fold_results[model_name].append({
                    "fold": fold_num,
                    "train_start": train_idx[0],
                    "train_end": train_idx[-1],
                    "test_start": test_idx[0],
                    "test_end": test_idx[-1],
                    "forecast": None,
                    "rmse_train": {b: np.inf for b in beta_cols},
                    "rmse_test": {b: np.inf for b in beta_cols},
                    "avg_rmse_train": np.inf,
                    "avg_rmse_test": np.inf,
                    "overfit_gap": np.inf,
                    "overfit_pct": np.inf
                })
                print(f"{model_name:8s} | ошибка: {e}")

    summary = {}

    for model_name in model_names:
        valid = [x for x in fold_results[model_name] if np.isfinite(x["avg_rmse_test"])]

        if len(valid) == 0:
            summary[model_name] = {
                "mean_train_rmse": np.inf,
                "mean_test_rmse": np.inf,
                "std_test_rmse": np.inf,
                "mean_overfit_gap": np.inf,
                "mean_overfit_pct": np.inf,
                "mean_beta_test_rmse": {b: np.inf for b in beta_cols}
            }
            continue

        mean_train_rmse = float(np.mean([x["avg_rmse_train"] for x in valid]))
        mean_test_rmse = float(np.mean([x["avg_rmse_test"] for x in valid]))
        std_test_rmse = float(np.std([x["avg_rmse_test"] for x in valid]))
        mean_overfit_gap = float(np.mean([x["overfit_gap"] for x in valid]))
        mean_overfit_pct = float(np.mean([x["overfit_pct"] for x in valid if np.isfinite(x["overfit_pct"])]))

        mean_beta_test_rmse = {}
        for beta in beta_cols:
            vals = [x["rmse_test"][beta] for x in valid if np.isfinite(x["rmse_test"][beta])]
            mean_beta_test_rmse[beta] = float(np.mean(vals)) if len(vals) else np.inf

        summary[model_name] = {
            "mean_train_rmse": mean_train_rmse,
            "mean_test_rmse": mean_test_rmse,
            "std_test_rmse": std_test_rmse,
            "mean_overfit_gap": mean_overfit_gap,
            "mean_overfit_pct": mean_overfit_pct,
            "mean_beta_test_rmse": mean_beta_test_rmse
        }

    print("\n" + "=" * 90)
    print("ИТОГ CV")
    print("=" * 90)
    print(
        f"{'Модель':<10} | "
        f"{'Train RMSE':>10} | "
        f"{'Test RMSE':>10} | "
        f"{'Std Test':>10} | "
        f"{'Overfit Gap':>12} | "
        f"{'Overfit %':>11}"
    )
    print("-" * 90)

    sorted_summary = sorted(summary.items(), key=lambda x: x[1]["mean_test_rmse"])

    for i, (model_name, res) in enumerate(sorted_summary):
        star = " ★" if i == 0 else ""
        print(
            f"{model_name:<10} | "
            f"{res['mean_train_rmse']:>10.4f} | "
            f"{res['mean_test_rmse']:>10.4f} | "
            f"{res['std_test_rmse']:>10.4f} | "
            f"{res['mean_overfit_gap']:>12.4f} | "
            f"{res['mean_overfit_pct']:>10.2%}"
            f"{star}"
        )

    print("\n" + "-" * 90)
    print("Средний TEST RMSE по каждому beta")
    print("-" * 90)
    print(f"{'Модель':<10} | {'beta0':>10} | {'beta1':>10} | {'beta2':>10}")
    print("-" * 90)

    for model_name, res in sorted_summary:
        print(
            f"{model_name:<10} | "
            f"{res['mean_beta_test_rmse'].get('beta0', np.inf):>10.4f} | "
            f"{res['mean_beta_test_rmse'].get('beta1', np.inf):>10.4f} | "
            f"{res['mean_beta_test_rmse'].get('beta2', np.inf):>10.4f}"
        )

    best_model_name = sorted_summary[0][0]
    return fold_results, summary, best_model_name


# =========================================================
# 8. ФИНАЛЬНОЕ ОБУЧЕНИЕ И ПРОГНОЗ
# =========================================================

def fit_best_and_forecast(best_model_name, df_betas, df_macro, beta_cols, macro_cols, forecast_horizon=6):
    future_index = pd.date_range(
        start=df_betas.index[-1] + pd.DateOffset(months=1),
        periods=forecast_horizon,
        freq="MS"
    )

    print("\n" + "=" * 90)
    print(f"ФИНАЛЬНОЕ ОБУЧЕНИЕ ЛУЧШЕЙ МОДЕЛИ: {best_model_name}")
    print("=" * 90)

    if best_model_name == "AR1":
        model = Model_AR1(beta_cols=beta_cols, lag=AR_LAG)
        model.fit(df_betas)
        forecast = model.forecast(steps=forecast_horizon, future_index=future_index)

    elif best_model_name == "ARX":
        future_macro = forecast_macro_naive(df_macro, future_index, macro_cols)
        model = Model_ARX(beta_cols=beta_cols, macro_cols=macro_cols, lag=ARX_LAG)
        model.fit(df_betas, df_macro)
        forecast = model.forecast(
            steps=forecast_horizon,
            future_index=future_index,
            future_macro=future_macro
        )

    elif best_model_name == "ARIMA":
        model = Model_ARIMA(beta_cols=beta_cols, order_default=ARIMA_ORDER_DEFAULT)
        model.fit(df_betas)
        forecast = model.forecast(steps=forecast_horizon, future_index=future_index)

    elif best_model_name == "ARIMAX":
        future_macro = forecast_macro_naive(df_macro, future_index, macro_cols)
        model = Model_ARIMAX(beta_cols=beta_cols, macro_cols=macro_cols, order_default=ARIMAX_ORDER_DEFAULT)
        model.fit(df_betas, df_macro)
        forecast = model.forecast(
            steps=forecast_horizon,
            future_index=future_index,
            future_macro=future_macro
        )

    else:
        raise ValueError("Неизвестная модель")

    return forecast


# =========================================================
# 9. ГРАФИКИ И ТАБЛИЦЫ
# =========================================================

def plot_cv_test_rmse(fold_results):
    plt.figure(figsize=(12, 6))

    for model_name, res_list in fold_results.items():
        x = [r["fold"] for r in res_list]
        y = [r["avg_rmse_test"] for r in res_list]
        plt.plot(x, y, marker="o", linewidth=1.8, label=model_name)

    plt.title("Expanding-window CV: Test RMSE по фолдам")
    plt.xlabel("Fold")
    plt.ylabel("Average Test RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cv_test_rmse_4_models.png", dpi=150)
    plt.show()


def plot_overfit_gap(fold_results):
    plt.figure(figsize=(12, 6))

    for model_name, res_list in fold_results.items():
        x = [r["fold"] for r in res_list]
        y = [r["overfit_gap"] for r in res_list]
        plt.plot(x, y, marker="o", linewidth=1.8, label=model_name)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Expanding-window CV: Overfit gap = Test RMSE - Train RMSE")
    plt.xlabel("Fold")
    plt.ylabel("Overfit gap")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cv_overfit_gap_4_models.png", dpi=150)
    plt.show()


def build_fold_table(fold_results):
    rows = []

    for model_name, res_list in fold_results.items():
        for r in res_list:
            rows.append({
                "model": model_name,
                "fold": r["fold"],
                "train_start": r["train_start"],
                "train_end": r["train_end"],
                "test_start": r["test_start"],
                "test_end": r["test_end"],
                "avg_rmse_train": r["avg_rmse_train"],
                "avg_rmse_test": r["avg_rmse_test"],
                "overfit_gap": r["overfit_gap"],
                "overfit_pct": r["overfit_pct"],
                "rmse_test_beta0": r["rmse_test"].get("beta0", np.nan),
                "rmse_test_beta1": r["rmse_test"].get("beta1", np.nan),
                "rmse_test_beta2": r["rmse_test"].get("beta2", np.nan),
            })

    return pd.DataFrame(rows)


def build_summary_table(summary, beta_cols):
    rows = []
    for model_name, res in summary.items():
        row = {
            "model": model_name,
            "mean_train_rmse": res["mean_train_rmse"],
            "mean_test_rmse": res["mean_test_rmse"],
            "std_test_rmse": res["std_test_rmse"],
            "mean_overfit_gap": res["mean_overfit_gap"],
            "mean_overfit_pct": res["mean_overfit_pct"],
        }
        for beta in beta_cols:
            row[f"mean_test_rmse_{beta}"] = res["mean_beta_test_rmse"].get(beta, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_test_rmse")


# =========================================================
# 10. MAIN
# =========================================================

def main():
    print("=" * 90)
    print("СРАВНЕНИЕ 4 МОДЕЛЕЙ: AR1 vs ARX vs ARIMA vs ARIMAX")
    print("=" * 90)

    df_betas, df_macro = load_data()
    df_betas, df_macro, beta_cols, macro_cols = prepare_common_sample(
        df_betas, df_macro,
        beta_cols=BETA_COLS,
        macro_cols=MACRO_COLS_CANDIDATE
    )

    print("\nBeta-колонки:", beta_cols)
    print("Macro-колонки:", macro_cols)
    print(f"Размер общей выборки: {len(df_betas)} месяцев")
    print(f"Период данных: {df_betas.index[0].strftime('%Y-%m')} - {df_betas.index[-1].strftime('%Y-%m')}")

    fold_results, summary, best_model_name = evaluate_models_expanding_cv(
        df_betas=df_betas,
        df_macro=df_macro,
        beta_cols=beta_cols,
        macro_cols=macro_cols,
        initial_train_size=INITIAL_TRAIN_SIZE,
        horizon=CV_HORIZON,
        step=CV_STEP
    )

    plot_cv_test_rmse(fold_results)
    plot_overfit_gap(fold_results)

    fold_table = build_fold_table(fold_results)
    fold_table.to_csv("cv_fold_results_4_models.csv", index=False)

    summary_table = build_summary_table(summary, beta_cols)
    summary_table.to_csv("cv_summary_4_models.csv", index=False)

    final_forecast = fit_best_and_forecast(
        best_model_name=best_model_name,
        df_betas=df_betas,
        df_macro=df_macro,
        beta_cols=beta_cols,
        macro_cols=macro_cols,
        forecast_horizon=FORECAST_HORIZON
    )

    print("\nФинальный прогноз лучшей модели:")
    print(final_forecast)

    final_forecast.to_csv(f"beta_forecast_best_{best_model_name}.csv", index=True)

    print("\nФайлы сохранены:")
    print("- cv_fold_results_4_models.csv")
    print("- cv_summary_4_models.csv")
    print("- cv_test_rmse_4_models.png")
    print("- cv_overfit_gap_4_models.png")
    print(f"- beta_forecast_best_{best_model_name}.csv")

    return fold_results, summary, final_forecast


if __name__ == "__main__":
    fold_results, summary, final_forecast = main()