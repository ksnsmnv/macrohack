import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.tsatools import lagmat
import warnings
warnings.filterwarnings("ignore")

# 1.1. Загрузка коэффициентов NS (бета)
df_betas = pd.read_csv("data/ns_results/betas_0_7308.csv")  # столбцы beta0, beta1, beta2, ... (сколько у вас есть)

# Предположим, что строки соответствуют месяцам с марта 2019
n_rows = len(df_betas)
start_date = datetime(2019, 3, 1)
dates = pd.date_range(start=start_date, periods=n_rows, freq="MS")  # MS = начала месяца

df_betas["date"] = dates
df_betas = df_betas.set_index("date")

# 1.2. Загрузка макрофакторов
df_macro = pd.read_excel("data/inputs/macro_updated.xlsx")
df_macro["date"] = pd.to_datetime(df_macro["date"])  # или соответствующий столбец с датой
df_macro = df_macro.set_index("date")

# 1.3. Склеиваем беты и макро по дате (внутреннее пересечение)
data_all = df_betas.join(df_macro, how="inner")  # или how="left"/"right" в зависимости от структуры
data_all = data_all.sort_index()  # убедиться, что дата по возрастанию



# 2.1. Выбор только числовых столбцов
var_cols = data_all.select_dtypes(include=[np.number]).columns.tolist()
df_var1 = data_all[var_cols].copy()

# 2.2. Проверка стационарности (упрощённо, можно сделать по каждому ряду)
for col in df_var1.columns:
    result = adfuller(df_var1[col].dropna())
    pval = result[1]
    print(f"{col}: p‑value = {pval:.4f}")

# 2.3. Подготовка: логарифмирование/дифференцирование (если нужно)
# Для простоты пример без дифференцирования, если ряды стационарны
df_var1_clean = (
    df_var1.replace([np.inf, -np.inf], np.nan)
           .fillna(method="ffill")
           .fillna(method="bfill")
)
y1 = df_var1_clean.values

# 2.4. Подбор оптимального лага (AIC)
model_var1 = VAR(y1, dates=df_var1.index)
# Вместо слишком большого maxlags
max_obs = len(df_var1)  # 79
n_vars = df_var1.shape[1]  # 13

# Оцените максимальный лаг хотя бы грубо:
max_lag_safe = max(1, min(5, (max_obs // (n_vars + 1)) - 1))
print("Максимальный безопасный лаг:", max_lag_safe)

model_var1 = VAR(y1, dates=df_var1.index)
lag_order = model_var1.select_order(maxlags=max_lag_safe)
print("Предпочтительный лаг по AIC:", lag_order.aic)

p = lag_order.aic
var_model1 = model_var1.fit(maxlags=p, trend="c")

# 2.5. Оценка VAR(p)
p = lag_order.aic  # или lag_order.selected_orders["aic"]
var_model1 = model_var1.fit(maxlags=p, trend="c")

print("VAR(1) параметры:\n", var_model1.summary())

# 2.6. Прогноз (например, на 12 месяцев вперёд)
horizon = 12
forecast1 = var_model1.forecast(y1[-p:], steps=horizon)

# 2.7. Можно вернуть в DataFrame с датами
future_dates = pd.date_range(start=df_var1.index[-1] + pd.DateOffset(months=1),
                             periods=horizon, freq="MS")
df_forecast1 = pd.DataFrame(
    forecast1,
    index=future_dates,
    columns=[f"pred_{col}" for col in df_var1.columns]
)
print("\nПрогноз VAR‑1 (первые 3 ряда):")
print(df_forecast1.head(3))



from sklearn.metrics import mean_squared_error
import numpy as np

print("\n" + "="*60)
print("8. БЛОК ОЦЕНКИ RMSE для VAR‑1")
print("="*60)

# 8.1 In‑sample RMSE: ошибки на исторических данных

resid = var_model1.resid
y_train = df_var1_clean.values

if resid.shape == y_train.shape:
    y_pred = y_train - resid
else:
    # если residuals короче (из-за лагов), обрезаем y_train
    y_train = y_train[p:]
    y_pred = y_train - resid

cols = df_var1_clean.columns.tolist()

rmse_in_sample = {}
for i, col in enumerate(cols):
    rmse = np.sqrt(mean_squared_error(y_train[:, i], y_pred[:, i]))
    rmse_in_sample[col] = rmse

print("\nin‑sample RMSE (VAR‑1, исторические beta + macro, lag =", p, "):")
for col, rmse in rmse_in_sample.items():
    print(f"{col:20s} | RMSE = {rmse:8.4f}")