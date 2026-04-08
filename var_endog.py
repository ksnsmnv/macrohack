import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import LinearRegression

# =========================================================
# 0. ПАРАМЕТРЫ
# =========================================================

BETAS_PATH = "data/ns_results/betas_0_7308.csv"
MACRO_PATH = "data/inputs/macro_updated.xlsx"

START_DATE_BETAS = "2019-03-01"
FREQ = "MS"                  # monthly start
FORECAST_HORIZON = 12

# лаги во второй модели для beta-блока
LAG_BETA = 4
LAG_MACRO_IN_BETA = 4

# верхняя граница лагов для VAR по макро
MAX_CANDIDATE_LAG = 12

# названия макропеременных
TARGET_MACRO_COLS = [
    "cbr", "inf", "observed_inf", "expected_inf",
    "usd", "moex", "brent", "vix"
]

# =========================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =========================================================

df_betas = pd.read_csv(BETAS_PATH)

# создаем ежемесячную дату с марта 2019
n_rows = len(df_betas)
df_betas["date"] = pd.date_range(start=START_DATE_BETAS, periods=n_rows, freq=FREQ)
df_betas = df_betas.set_index("date").sort_index()

df_macro = pd.read_excel(MACRO_PATH)

# если столбец даты называется не "date", попробуем найти автоматически
date_candidates = [c for c in df_macro.columns if str(c).lower() in ["date", "dt", "month", "period"]]
if "date" not in df_macro.columns:
    if len(date_candidates) == 0:
        raise ValueError("В macro_updated.xlsx не найден столбец даты. Добавьте столбец 'date'.")
    df_macro = df_macro.rename(columns={date_candidates[0]: "date"})

df_macro["date"] = pd.to_datetime(df_macro["date"])
df_macro = df_macro.set_index("date").sort_index()

# =========================================================
# 2. ПРИВЕДЕНИЕ К ЧИСЛОВОМУ ВИДУ
# =========================================================

for col in df_betas.columns:
    df_betas[col] = pd.to_numeric(df_betas[col], errors="coerce")

for col in df_macro.columns:
    df_macro[col] = pd.to_numeric(df_macro[col], errors="coerce")

# пересечение дат
data_all = df_betas.join(df_macro, how="inner").sort_index()

# определяем beta-столбцы
beta_cols = [c for c in df_betas.columns if str(c).lower().startswith("beta")]
if len(beta_cols) == 0:
    # fallback: если beta не названы beta0/beta1/...,
    # считаем, что все числовые столбцы из df_betas - это NS-параметры
    beta_cols = df_betas.columns.tolist()

macro_cols = [c for c in TARGET_MACRO_COLS if c in data_all.columns]

if len(macro_cols) == 0:
    raise ValueError("В объединенной таблице не найдены макрофакторы.")
if len(beta_cols) == 0:
    raise ValueError("Не найдены beta-столбцы в файле betas_0_7308.csv.")

print("Найдены beta-столбцы:", beta_cols)
print("Найдены macro-столбцы:", macro_cols)
print("Размер объединенных данных:", data_all.shape)

# =========================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Заменить inf на NaN и удалить пустые строки."""
    return (
        df.replace([np.inf, -np.inf], np.nan)
          .dropna(axis=0, how="any")
          .sort_index()
    )

def make_lags(df: pd.DataFrame, nlags: int, prefix: str = "") -> pd.DataFrame:
    """Построить лаги 1..nlags для всех столбцов."""
    parts = []
    for lag in range(1, nlags + 1):
        tmp = df.shift(lag).copy()
        tmp.columns = [f"{prefix}{col}_lag{lag}" for col in df.columns]
        parts.append(tmp)
    return pd.concat(parts, axis=1)

def safe_var_select_and_fit(df_endog: pd.DataFrame, max_candidate_lag: int = 12, trend: str = "c"):
    """
    Безопасный подбор лага и оценка VAR.
    Сначала считаем max_estimable по логике statsmodels,
    затем пробуем select_order() от большего maxlags к меньшему.
    """
    df_endog = clean_df(df_endog)

    nobs = df_endog.shape[0]
    neqs = df_endog.shape[1]

    if nobs <= neqs + 2:
        raise ValueError(
            f"Слишком мало наблюдений для VAR: nobs={nobs}, neqs={neqs}. "
            "Нужно больше данных или меньше переменных."
        )

    # trend="c" -> ntrend = 1
    ntrend = 1 if trend == "c" else 0

    # Приближенно по внутренней логике statsmodels:
    # max_estimable = (nobs - neqs - ntrend) // (1 + neqs)
    max_estimable = (nobs - neqs - ntrend) // (1 + neqs)

    if max_estimable < 1:
        raise ValueError(
            f"Даже лаг 1 не оценивается: nobs={nobs}, neqs={neqs}, "
            f"max_estimable={max_estimable}. "
            "Сократите число переменных или увеличьте выборку."
        )

    safe_maxlags = min(max_candidate_lag, max_estimable)

    print(f"[VAR macro] nobs={nobs}, neqs={neqs}, max_estimable={max_estimable}, safe_maxlags={safe_maxlags}")

    model = VAR(df_endog)

    selected_p = None
    used_maxlags = None
    last_error = None

    # пробуем select_order от safe_maxlags вниз до 1
    for trial_maxlags in range(safe_maxlags, 0, -1):
        try:
            order_res = model.select_order(maxlags=trial_maxlags)
            # корректный способ брать выбранный лаг
            selected_p = order_res.selected_orders.get("aic", None)
            if selected_p is None or selected_p < 1:
                selected_p = 1
            used_maxlags = trial_maxlags
            print(f"[VAR macro] select_order ok: trial_maxlags={trial_maxlags}, selected_p={selected_p}")
            break
        except Exception as e:
            last_error = e
            print(f"[VAR macro] select_order failed for maxlags={trial_maxlags}: {e}")

    # если select_order вообще не сработал, пытаемся просто оценить VAR(1)
    if selected_p is None:
        print("[VAR macro] select_order не сработал, fallback на p=1")
        selected_p = 1

    try:
        fitted = model.fit(selected_p, trend=trend)
    except Exception as e:
        raise ValueError(
            f"Не удалось оценить даже итоговую VAR({selected_p}). "
            f"Последняя ошибка select_order: {last_error}. Ошибка fit: {e}"
        )

    return {
        "model": model,
        "result": fitted,
        "selected_p": selected_p,
        "used_maxlags": used_maxlags,
        "nobs": nobs,
        "neqs": neqs,
        "max_estimable": max_estimable
    }

# =========================================================
# 4. ЭТАП 1: VAR ПО МАКРОФАКТОРАМ
# =========================================================

df_macro_hist = data_all[macro_cols].copy()
df_macro_hist = clean_df(df_macro_hist)

print("\nРазмер macro после очистки:", df_macro_hist.shape)

macro_var_info = safe_var_select_and_fit(
    df_endog=df_macro_hist,
    max_candidate_lag=MAX_CANDIDATE_LAG,
    trend="c"
)

macro_var_res = macro_var_info["result"]
p_macro = macro_var_info["selected_p"]

print("\nВыбранный лаг для macro VAR:", p_macro)
print(macro_var_res.summary())

# прогноз макро
future_dates = pd.date_range(
    start=df_macro_hist.index[-1] + pd.DateOffset(months=1),
    periods=FORECAST_HORIZON,
    freq="MS"
)

macro_forecast = macro_var_res.forecast(
    y=df_macro_hist.values[-p_macro:],
    steps=FORECAST_HORIZON
)

df_macro_forecast = pd.DataFrame(
    macro_forecast,
    index=future_dates,
    columns=macro_cols
)

print("\nПрогноз macro (head):")
print(df_macro_forecast.head())

# =========================================================
# 5. ЭТАП 2: МОДЕЛЬ ДЛЯ BETA С ЛАГАМИ BETA И MACRO
# =========================================================

df_betas_hist = data_all[beta_cols].copy()
df_betas_hist = df_betas_hist.replace([np.inf, -np.inf], np.nan)

# выравниваем историю по общим датам, где есть и beta, и macro
common_hist_idx = df_betas_hist.dropna().index.intersection(df_macro_hist.index)
df_betas_hist = df_betas_hist.loc[common_hist_idx].sort_index()
df_macro_hist2 = df_macro_hist.loc[common_hist_idx].sort_index()

print("\nРазмер beta-истории после выравнивания:", df_betas_hist.shape)
print("Размер macro-истории после выравнивания:", df_macro_hist2.shape)

# лаги
X_beta_lags = make_lags(df_betas_hist, LAG_BETA, prefix="b_")
X_macro_lags = make_lags(df_macro_hist2, LAG_MACRO_IN_BETA, prefix="m_")

Y_beta = df_betas_hist.copy()

# единая train-матрица; чистим после создания лагов
train_all = pd.concat([Y_beta, X_beta_lags, X_macro_lags], axis=1)
train_all = train_all.replace([np.inf, -np.inf], np.nan).dropna()

Y_train = train_all[beta_cols]
X_train = train_all.drop(columns=beta_cols)

if len(train_all) < 10:
    raise ValueError(
        f"После построения лагов осталось слишком мало наблюдений: {len(train_all)}. "
        "Уменьшите LAG_BETA / LAG_MACRO_IN_BETA."
    )

print("\nРазмер train_all для beta-моделей:", train_all.shape)
print("Число признаков X_train:", X_train.shape[1])

# обучаем отдельную линейную модель на каждый beta
beta_models = {}
for beta in beta_cols:
    reg = LinearRegression()
    reg.fit(X_train, Y_train[beta])
    beta_models[beta] = reg

print("Обучено моделей beta:", len(beta_models))

# =========================================================
# 6. РЕКУРСИВНЫЙ ПРОГНОЗ BETA НА ОСНОВЕ ПРОГНОЗОВ MACRO
# =========================================================

# полная macro-таблица: история + прогноз
macro_full = pd.concat([df_macro_hist2, df_macro_forecast], axis=0)

# история beta, которую будем наращивать прогнозами
betas_extended = df_betas_hist.copy()

beta_forecast_rows = []

# шаблон колонок X должен строго совпадать с X_train.columns
expected_x_columns = X_train.columns.tolist()

for dt in future_dates:
    row_dict = {}

    # лаги beta
    for lag in range(1, LAG_BETA + 1):
        lag_date = dt - pd.DateOffset(months=lag)

        if lag_date not in betas_extended.index:
            raise ValueError(f"Для даты {dt} не хватает beta-истории на лаге {lag}.")

        for col in beta_cols:
            row_dict[f"b_{col}_lag{lag}"] = betas_extended.loc[lag_date, col]

    # лаги macro
    for lag in range(1, LAG_MACRO_IN_BETA + 1):
        lag_date = dt - pd.DateOffset(months=lag)

        if lag_date not in macro_full.index:
            raise ValueError(f"Для даты {dt} не хватает macro-истории на лаге {lag}.")

        for col in macro_cols:
            row_dict[f"m_{col}_lag{lag}"] = macro_full.loc[lag_date, col]

    X_step = pd.DataFrame(row_dict, index=[dt])

    # выравниваем порядок колонок строго под train
    X_step = X_step.reindex(columns=expected_x_columns)

    # защита от NaN/inf
    X_step = X_step.replace([np.inf, -np.inf], np.nan)

    if X_step.isna().sum().sum() > 0:
        raise ValueError(
            f"На шаге прогноза {dt} появились NaN после сборки X_step. "
            "Проверьте даты и лаги."
        )

    pred_row = {}
    for beta in beta_cols:
        pred_row[beta] = beta_models[beta].predict(X_step)[0]

    pred_df = pd.DataFrame(pred_row, index=[dt])
    beta_forecast_rows.append(pred_df)

    # добавляем прогноз в историю, чтобы следующий шаг мог использовать лаги beta
    betas_extended = pd.concat([betas_extended, pred_df], axis=0)

df_beta_forecast = pd.concat(beta_forecast_rows, axis=0)

print("\nПрогноз beta (head):")
print(df_beta_forecast.head())

# =========================================================
# 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================================================

df_macro_forecast.to_csv("macro_forecast_model2.csv", index=True)
df_beta_forecast.to_csv("beta_forecast_model2.csv", index=True)

print("\nФайлы сохранены:")
print("- macro_forecast_model2.csv")
print("- beta_forecast_model2.csv")


from sklearn.metrics import mean_squared_error

print("\n" + "="*60)
print("8. БЛОК ОЦЕНКИ RMSE (In‑sample / Hold‑out)")
print("="*60)

# 8.1 RMSE по историческим данным (In‑sample)
# Предсказанные значения на истории
if len(Y_train) > 0:
    Y_pred_in_sample = pd.DataFrame(index=Y_train.index, columns=beta_cols)

    for beta in beta_cols:
        Y_pred_in_sample[beta] = beta_models[beta].predict(X_train)

    rmse_in_sample = {}
    for beta in beta_cols:
        y_true = Y_train[beta].values
        y_pred = Y_pred_in_sample[beta].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_in_sample[beta] = rmse

    print("\nin‑sample RMSE (история beta, лаги beta + macro):")
    for beta, rmse in rmse_in_sample.items():
        print(f"{beta:15s} | RMSE = {rmse:8.4f}")

# 8.2 RMSE на прогнозируемом горизонте (если есть реальные значения, например df_betas_future)
# Предполагаем, что реальные значения на горизонте FORECAST_HORIZON уже есть в data_all
future_true = data_all[beta_cols].loc[df_beta_forecast.index] if df_beta_forecast.index[-1] in data_all.index else None

if future_true is not None and not future_true.isna().any().any():
    Y_true_future = future_true[beta_cols]
    Y_pred_future = df_beta_forecast[beta_cols]

    rmse_future = {}
    for beta in beta_cols:
        y_true = Y_true_future[beta].values
        y_pred = Y_pred_future[beta].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmse_future[beta] = rmse

    print("\nout‑of‑sample RMSE (прогноз на будущее):")
    for beta, rmse in rmse_future.items():
        print(f"{beta:15s} | RMSE = {rmse:8.4f}")
else:
    print("\nout‑of‑sample RMSE: реальные значения beta на прогнозном горизонте недоступны, "
          "выполните out‑of‑sample тест, если есть будущие наблюдения.")