import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller


# =========================================================
# 1. ПАРАМЕТРЫ
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

VAL_SIZE = 6
TEST_SIZE = 6
DELTA = 0.5
MAX_VAR_LAGS = 4


# =========================================================
# 2. БАЗОВЫЕ ФУНКЦИИ
# =========================================================

def normalize_month_index(idx):
    idx = pd.to_datetime(idx)
    return idx.to_period("M").to_timestamp("M")


def to_num_df(df):
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(
            out[c].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        )
    return out


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_adf(series, alpha=0.05):
    s = pd.Series(series).dropna()
    if len(s) < 15:
        return False
    try:
        return adfuller(s)[1] < alpha
    except Exception:
        return False


# =========================================================
# 3. ЗАГРУЗКА ДАННЫХ
# =========================================================

def load_betas():
    df = pd.read_csv(BETAS_PATH)
    df["date"] = pd.date_range(start=START_DATE_BETAS, periods=len(df), freq=FREQ)
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    df = to_num_df(df)
    return df


def load_yield_curve():
    df = pd.read_excel(YC_PATH)
    if "Month" not in df.columns:
        raise ValueError("В yield_curve.xlsx не найден столбец Month")
    df["date"] = pd.to_datetime(df["Month"])
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    df = to_num_df(df)
    return df


def load_macro():
    df = pd.read_excel(MACRO_PATH)

    date_col = None
    for c in df.columns:
        if str(c).lower() in ["date", "dt", "month", "period", "дата"]:
            date_col = c
            break

    if date_col is None:
        raise ValueError("В macro_updated.xlsx не найден столбец даты")

    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.index = normalize_month_index(df.index)
    df = to_num_df(df)
    return df


def load_iv():
    df = pd.read_excel(IV_PATH)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["Date", "Maturity", "Maturity (year fraction)", "Strike", "Volatility"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"В Problem_1_IV_train.xlsx не найден столбец {c}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["Maturity (year fraction)"] = pd.to_numeric(
        df["Maturity (year fraction)"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df["Strike"] = pd.to_numeric(
        df["Strike"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    df["Volatility"] = pd.to_numeric(
        df["Volatility"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Volatility"])

    # Агрегаты по месяцу
    agg = (
        df.groupby("Date")
        .agg(
            iv_mean=("Volatility", "mean"),
            iv_median=("Volatility", "median"),
            iv_std=("Volatility", "std"),
            iv_count=("Volatility", "count"),
            iv_tau_mean=("Maturity (year fraction)", "mean"),
            iv_strike_mean=("Strike", "mean")
        )
        .reset_index()
    )

    agg["date"] = pd.to_datetime(agg["Date"])
    agg = agg.drop(columns=["Date"]).set_index("date").sort_index()
    agg.index = normalize_month_index(agg.index)
    agg["iv_std"] = agg["iv_std"].fillna(0.0)

    return agg


# =========================================================
# 4. NS: БЕТЫ -> YIELD CURVE
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
    return beta0 + beta1 * l1 + beta2 * l2


def reconstruct_yield_curve(df_betas, maturity_map):
    out = pd.DataFrame(index=df_betas.index)
    for label, tau in maturity_map.items():
        out[label] = [
            nelson_siegel_yield(
                tau=tau,
                beta0=row["beta0"],
                beta1=row["beta1"],
                beta2=row["beta2"],
                lam=NS_LAMBDA
            )
            for _, row in df_betas.iterrows()
        ]
    return out


# =========================================================
# 5. МЕТРИКА ИЗ УСЛОВИЯ
# =========================================================

def compute_weighted_rmse_curve(
    df_yc_actual,
    df_yc_forecast,
    cols=YIELD_COLS,
    w_on=0.4,
    w_rest_total=0.6
):
    cols = [c for c in cols if c in df_yc_actual.columns and c in df_yc_forecast.columns]
    if len(cols) < 2 or "ON" not in cols:
        return 100.0

    actual = df_yc_actual[cols].copy()
    pred = df_yc_forecast[cols].copy()

    common_idx = actual.index.intersection(pred.index)
    if len(common_idx) == 0:
        return 100.0

    actual = actual.loc[common_idx]
    pred = pred.loc[common_idx]

    rest = [c for c in cols if c != "ON"]
    if len(rest) == 0:
        return 100.0

    weights = {"ON": w_on}
    w_rest = w_rest_total / len(rest)
    for c in rest:
        weights[c] = w_rest

    total = 0.0
    cnt = 0

    for dt in common_idx:
        for c in cols:
            y_true = actual.loc[dt, c]
            y_pred = pred.loc[dt, c]
            if pd.isna(y_true) or pd.isna(y_pred):
                continue
            total += weights[c] * (y_pred - y_true) ** 2
            cnt += 1

    if cnt == 0:
        return 100.0

    return np.sqrt(total / cnt)


def compute_rmse_total(rmse_m1, rmse_m2, delta=0.5):
    if np.isnan(rmse_m1) or np.isnan(rmse_m2):
        return 100.0
    return delta * rmse_m1 + (1.0 - delta) * rmse_m2


# =========================================================
# 6. ПОДГОТОВКА ДАТАСЕТА
# =========================================================

df_betas = load_betas()
df_yc = load_yield_curve()
df_macro = load_macro()
df_iv = load_iv()

beta_cols = [c for c in BETA_COLS if c in df_betas.columns]
macro_cols = [c for c in MACRO_COLS_CANDIDATE if c in df_macro.columns]
iv_cols = [c for c in df_iv.columns if c.startswith("iv_")]
yc_cols = [c for c in YIELD_COLS if c in df_yc.columns]

common_idx = (
    df_betas.index
    .intersection(df_macro.index)
    .intersection(df_iv.index)
    .intersection(df_yc.index)
    .sort_values()
)

df_betas = df_betas.loc[common_idx, beta_cols].replace([np.inf, -np.inf], np.nan).dropna()
df_macro = df_macro.loc[common_idx, macro_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
df_iv = df_iv.loc[common_idx, iv_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
df_yc = df_yc.loc[common_idx, yc_cols].replace([np.inf, -np.inf], np.nan)

common_idx = (
    df_betas.index
    .intersection(df_macro.index)
    .intersection(df_iv.index)
    .intersection(df_yc.index)
    .sort_values()
)

df_betas = df_betas.loc[common_idx]
df_macro = df_macro.loc[common_idx]
df_iv = df_iv.loc[common_idx]
df_yc = df_yc.loc[common_idx]

print("Shape проверки (после загрузки данных):")
print("df_betas:", df_betas.shape, "| содержит NaN:", int(df_betas.isna().sum().sum()))
print("df_macro:", df_macro.shape, "| NaN:", int(df_macro.isna().sum().sum()))
print("df_iv:", df_iv.shape, "| NaN:", int(df_iv.isna().sum().sum()))
print("df_yc:", df_yc.shape, "| NaN:", int(df_yc.isna().sum().sum()))

train_idx = common_idx[: -(VAL_SIZE + TEST_SIZE)]
val_idx = common_idx[-(VAL_SIZE + TEST_SIZE): -TEST_SIZE]
test_idx = common_idx[-TEST_SIZE:]

print("Общий индекс (train / val / test):")
print("Общее length:", len(common_idx))
print("Train:", len(train_idx))
print("Val:", len(val_idx))
print("Test:", len(test_idx))


# =========================================================
# 7. СТАЦИОНАРИЗАЦИЯ
# =========================================================

def stationarize_df(df, prefer_growth_for_macro=False):
    """
    - если ряд стационарен: оставляем level
    - если нет:
        * для macro/iv при prefer_growth_for_macro=True пробуем pct_change
        * затем пробуем diff
        * если всё плохо, оставляем level
    """
    out = pd.DataFrame(index=df.index)
    transform_info = {}

    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        s = s.ffill().bfill()

        if s.isna().all():
            out[c] = s
            transform_info[c] = "level"
            continue

        if safe_adf(s):
            out[c] = s
            transform_info[c] = "level"
            continue

        if prefer_growth_for_macro:
            g = s.pct_change().replace([np.inf, -np.inf], np.nan)
            g = g.ffill().bfill()
            if g.notna().sum() >= 15 and safe_adf(g):
                out[c] = g
                transform_info[c] = "growth"
                continue

        d = s.diff().replace([np.inf, -np.inf], np.nan)
        d = d.ffill().bfill()
        if d.notna().sum() >= 15 and safe_adf(d):
            out[c] = d
            transform_info[c] = "diff"
            continue

        out[c] = s
        transform_info[c] = "level"

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out, transform_info


def invert_forecast(last_levels, forecast_stat, transform_info):
    """
    Обратное преобразование:
    level -> 그대로
    diff  -> cumulative sum от last level
    growth -> last * (1+g_t) cumulative
    """
    out = pd.DataFrame(index=forecast_stat.index, columns=forecast_stat.columns, dtype=float)
    prev = pd.Series(last_levels).copy()

    for dt in forecast_stat.index:
        for c in forecast_stat.columns:
            tr = transform_info.get(c, "level")
            val = forecast_stat.loc[dt, c]

            if pd.isna(val):
                out.loc[dt, c] = np.nan
                continue

            if tr == "level":
                out.loc[dt, c] = val
                prev[c] = val
            elif tr == "diff":
                base = prev.get(c, np.nan)
                out.loc[dt, c] = base + val if pd.notna(base) else np.nan
                prev[c] = out.loc[dt, c]
            elif tr == "growth":
                base = prev.get(c, np.nan)
                out.loc[dt, c] = base * (1.0 + val) if pd.notna(base) else np.nan
                prev[c] = out.loc[dt, c]
            else:
                out.loc[dt, c] = val
                prev[c] = val

    return out


# =========================================================
# 8. VAR HELPERS
# =========================================================

def fit_var_model(df_train, maxlags=MAX_VAR_LAGS):
    df_train = df_train.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    if len(df_train) < 15 or df_train.shape[1] < 2:
        return None

    maxlags_eff = min(maxlags, max(1, len(df_train) // 4))
    try:
        model = VAR(df_train)
        sel = model.select_order(maxlags=maxlags_eff)
        p = sel.aic if sel.aic is not None else 1
        if isinstance(p, np.integer):
            p = int(p)
        if p < 1:
            p = 1
        res = model.fit(p)
        return res
    except Exception:
        return None


def forecast_var(res, df_train, steps, future_index):
    if res is None:
        return pd.DataFrame(index=future_index, columns=df_train.columns, dtype=float)

    try:
        k_ar = res.k_ar
        init = df_train.values[-k_ar:]
        fc = res.forecast(init, steps=steps)
        return pd.DataFrame(fc, index=future_index, columns=df_train.columns)
    except Exception:
        return pd.DataFrame(index=future_index, columns=df_train.columns, dtype=float)


# =========================================================
# 9. VAR TYPE 1
# Один общий VAR на beta + macro (+ IV)
# =========================================================

def run_var_type1(
    df_betas_hist,
    df_macro_hist,
    df_iv_hist,
    future_index,
    with_iv=False
):
    if with_iv:
        full_level = pd.concat([df_betas_hist, df_macro_hist, df_iv_hist], axis=1)
    else:
        full_level = pd.concat([df_betas_hist, df_macro_hist], axis=1)

    full_level = full_level.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # stationarize: beta через diff fallback, macro/iv через growth/diff fallback
    betas_stat, beta_info = stationarize_df(df_betas_hist, prefer_growth_for_macro=False)
    macro_stat, macro_info = stationarize_df(df_macro_hist, prefer_growth_for_macro=True)

    if with_iv:
        iv_stat, iv_info = stationarize_df(df_iv_hist, prefer_growth_for_macro=True)
        full_stat = pd.concat([betas_stat, macro_stat, iv_stat], axis=1)
        tr_info = {**beta_info, **macro_info, **iv_info}
    else:
        full_stat = pd.concat([betas_stat, macro_stat], axis=1)
        tr_info = {**beta_info, **macro_info}

    full_stat = full_stat.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    res = fit_var_model(full_stat)
    if res is None:
        # fallback: last beta repeat
        beta_fc = pd.DataFrame(
            np.tile(df_betas_hist.iloc[-1].values, (len(future_index), 1)),
            index=future_index,
            columns=df_betas_hist.columns
        )
        yc_fc = reconstruct_yield_curve(beta_fc, {k: MATURITY_MAP[k] for k in yc_cols})
        return beta_fc, yc_fc

    fc_stat = forecast_var(res, full_stat, len(future_index), future_index)

    last_levels = full_level.iloc[-1].reindex(fc_stat.columns)
    fc_level = invert_forecast(last_levels, fc_stat, tr_info)

    beta_fc = fc_level[df_betas_hist.columns].copy()
    beta_fc = beta_fc.ffill().bfill()

    yc_fc = reconstruct_yield_curve(beta_fc, {k: MATURITY_MAP[k] for k in yc_cols})
    return beta_fc, yc_fc


# =========================================================
# 10. VAR TYPE 2
# Сначала VAR по macro (+ IV), затем VAR по beta+macro
# =========================================================

def run_var_type2(
    df_betas_hist,
    df_macro_hist,
    df_iv_hist,
    future_index,
    with_iv=False
):
    # --- STEP 1: forecast macro (+ IV)
    if with_iv:
        macro_level = pd.concat([df_macro_hist, df_iv_hist], axis=1)
    else:
        macro_level = df_macro_hist.copy()

    macro_level = macro_level.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    macro_stat, macro_info = stationarize_df(macro_level, prefer_growth_for_macro=True)

    res_macro = fit_var_model(macro_stat)
    if res_macro is None:
        macro_fc_level = pd.DataFrame(
            np.tile(macro_level.iloc[-1].values, (len(future_index), 1)),
            index=future_index,
            columns=macro_level.columns
        )
    else:
        macro_fc_stat = forecast_var(res_macro, macro_stat, len(future_index), future_index)
        macro_fc_level = invert_forecast(macro_level.iloc[-1].reindex(macro_fc_stat.columns), macro_fc_stat, macro_info)
        macro_fc_level = macro_fc_level.ffill().bfill()

    # --- STEP 2: build beta VAR on beta + macro history
    beta_macro_level_hist = pd.concat([df_betas_hist, macro_level], axis=1).replace([np.inf, -np.inf], np.nan).ffill().bfill()

    beta_stat, beta_info = stationarize_df(df_betas_hist, prefer_growth_for_macro=False)
    macro_hist_stat, _ = stationarize_df(macro_level, prefer_growth_for_macro=True)

    beta_var_stat = pd.concat([beta_stat, macro_hist_stat], axis=1).replace([np.inf, -np.inf], np.nan).ffill().bfill()

    res_beta = fit_var_model(beta_var_stat)
    if res_beta is None:
        beta_fc = pd.DataFrame(
            np.tile(df_betas_hist.iloc[-1].values, (len(future_index), 1)),
            index=future_index,
            columns=df_betas_hist.columns
        )
        yc_fc = reconstruct_yield_curve(beta_fc, {k: MATURITY_MAP[k] for k in yc_cols})
        return beta_fc, yc_fc

    # recursive one-step forecast with replacement of macro block by predicted macro
    full_stat_hist = beta_var_stat.copy()
    beta_fc_stat_rows = []

    for dt in future_index:
        res_tmp = fit_var_model(full_stat_hist)
        if res_tmp is None:
            row = pd.Series(index=full_stat_hist.columns, dtype=float, name=dt)
            for c in df_betas_hist.columns:
                row[c] = 0.0
            beta_fc_stat_rows.append(row)
            full_stat_hist = pd.concat([full_stat_hist, pd.DataFrame([row], index=[dt])], axis=0)
            continue

        one_fc = forecast_var(res_tmp, full_stat_hist, 1, pd.DatetimeIndex([dt]))
        row = one_fc.iloc[0].copy()

        # заменить macro/iv блок на stationarized version of predicted macro
        macro_future_full = pd.concat([macro_level, macro_fc_level], axis=0)

        for c in macro_level.columns:
            tr = macro_info.get(c, "level")
            if dt not in macro_future_full.index:
                continue

            if tr == "level":
                row[c] = macro_fc_level.loc[dt, c]
            elif tr == "diff":
                prev_dt = full_stat_hist.index[-1]
                prev_level = macro_future_full.loc[prev_dt, c] if prev_dt in macro_future_full.index else macro_level.iloc[-1][c]
                row[c] = macro_fc_level.loc[dt, c] - prev_level
            elif tr == "growth":
                prev_dt = full_stat_hist.index[-1]
                prev_level = macro_future_full.loc[prev_dt, c] if prev_dt in macro_future_full.index else macro_level.iloc[-1][c]
                if pd.notna(prev_level) and prev_level != 0:
                    row[c] = (macro_fc_level.loc[dt, c] / prev_level) - 1.0
                else:
                    row[c] = 0.0

        beta_fc_stat_rows.append(pd.Series(row, name=dt))
        full_stat_hist = pd.concat([full_stat_hist, pd.DataFrame([row], index=[dt])], axis=0)

    beta_fc_stat = pd.DataFrame(beta_fc_stat_rows)

    beta_fc_only_stat = beta_fc_stat[df_betas_hist.columns].copy()
    beta_fc_level = invert_forecast(df_betas_hist.iloc[-1], beta_fc_only_stat, beta_info)
    beta_fc_level = beta_fc_level.ffill().bfill()

    yc_fc = reconstruct_yield_curve(beta_fc_level, {k: MATURITY_MAP[k] for k in yc_cols})
    return beta_fc_level, yc_fc


# =========================================================
# 11. ВАЛИДАЦИЯ
# M1 = без IV
# M2 = с IV
# =========================================================

yc_val_actual = reconstruct_yield_curve(df_betas.loc[val_idx, beta_cols], {k: MATURITY_MAP[k] for k in yc_cols})
yc_test_actual = reconstruct_yield_curve(df_betas.loc[test_idx, beta_cols], {k: MATURITY_MAP[k] for k in yc_cols})

# Validation history = только train
b_hist_val = df_betas.loc[train_idx]
m_hist_val = df_macro.loc[train_idx]
iv_hist_val = df_iv.loc[train_idx]

# --- VAR TYPE 1 ---
_, yc_var1_m1_val = run_var_type1(b_hist_val, m_hist_val, iv_hist_val, val_idx, with_iv=False)
_, yc_var1_m2_val = run_var_type1(b_hist_val, m_hist_val, iv_hist_val, val_idx, with_iv=True)

rmse_var1_m1_val = compute_weighted_rmse_curve(yc_val_actual, yc_var1_m1_val, cols=yc_cols)
rmse_var1_m2_val = compute_weighted_rmse_curve(yc_val_actual, yc_var1_m2_val, cols=yc_cols)
rmse_total_var1_val = compute_rmse_total(rmse_var1_m1_val, rmse_var1_m2_val, delta=DELTA)

# --- VAR TYPE 2 ---
_, yc_var2_m1_val = run_var_type2(b_hist_val, m_hist_val, iv_hist_val, val_idx, with_iv=False)
_, yc_var2_m2_val = run_var_type2(b_hist_val, m_hist_val, iv_hist_val, val_idx, with_iv=True)

rmse_var2_m1_val = compute_weighted_rmse_curve(yc_val_actual, yc_var2_m1_val, cols=yc_cols)
rmse_var2_m2_val = compute_weighted_rmse_curve(yc_val_actual, yc_var2_m2_val, cols=yc_cols)
rmse_total_var2_val = compute_rmse_total(rmse_var2_m1_val, rmse_var2_m2_val, delta=DELTA)


# =========================================================
# 12. FINAL TEST
# train+val -> test
# =========================================================

trainval_idx = common_idx[:-TEST_SIZE]

b_hist_test = df_betas.loc[trainval_idx]
m_hist_test = df_macro.loc[trainval_idx]
iv_hist_test = df_iv.loc[trainval_idx]

# --- VAR TYPE 1 ---
_, yc_var1_m1_test = run_var_type1(b_hist_test, m_hist_test, iv_hist_test, test_idx, with_iv=False)
_, yc_var1_m2_test = run_var_type1(b_hist_test, m_hist_test, iv_hist_test, test_idx, with_iv=True)

rmse_var1_m1_test = compute_weighted_rmse_curve(yc_test_actual, yc_var1_m1_test, cols=yc_cols)
rmse_var1_m2_test = compute_weighted_rmse_curve(yc_test_actual, yc_var1_m2_test, cols=yc_cols)
rmse_total_var1_test = compute_rmse_total(rmse_var1_m1_test, rmse_var1_m2_test, delta=DELTA)

# --- VAR TYPE 2 ---
_, yc_var2_m1_test = run_var_type2(b_hist_test, m_hist_test, iv_hist_test, test_idx, with_iv=False)
_, yc_var2_m2_test = run_var_type2(b_hist_test, m_hist_test, iv_hist_test, test_idx, with_iv=True)

rmse_var2_m1_test = compute_weighted_rmse_curve(yc_test_actual, yc_var2_m1_test, cols=yc_cols)
rmse_var2_m2_test = compute_weighted_rmse_curve(yc_test_actual, yc_var2_m2_test, cols=yc_cols)
rmse_total_var2_test = compute_rmse_total(rmse_var2_m1_test, rmse_var2_m2_test, delta=DELTA)


# =========================================================
# 13. ВЫВОД РЕЗУЛЬТАТОВ
# =========================================================

print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)
print(f"VAR type 1 | M1 no IV        : {rmse_var1_m1_val:.4f}")
print(f"VAR type 1 | M2 with IV      : {rmse_var1_m2_val:.4f}")
print(f"VAR type 1 | RMSEtotal       : {rmse_total_var1_val:.4f}")
print("-"*70)
print(f"VAR type 2 | M1 no IV        : {rmse_var2_m1_val:.4f}")
print(f"VAR type 2 | M2 with IV      : {rmse_var2_m2_val:.4f}")
print(f"VAR type 2 | RMSEtotal       : {rmse_total_var2_val:.4f}")

best_var_type = "VAR type 1" if rmse_total_var1_val <= rmse_total_var2_val else "VAR type 2"
print("Лучший тип VAR по validation:", best_var_type)

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)
print(f"VAR type 1 | M1 no IV        : {rmse_var1_m1_test:.4f}")
print(f"VAR type 1 | M2 with IV      : {rmse_var1_m2_test:.4f}")
print(f"VAR type 1 | RMSEtotal       : {rmse_total_var1_test:.4f}")
print("-"*70)
print(f"VAR type 2 | M1 no IV        : {rmse_var2_m1_test:.4f}")
print(f"VAR type 2 | M2 with IV      : {rmse_var2_m2_test:.4f}")
print(f"VAR type 2 | RMSEtotal       : {rmse_total_var2_test:.4f}")


# =========================================================
# 14. ГРАФИКИ ДЛЯ TEST
# =========================================================

def plot_yc_block(yc_actual, yc_forecast, title_prefix, yc_cols=yc_cols):
    x = np.arange(len(yc_cols))
    for dt in yc_actual.index.intersection(yc_forecast.index):
        plt.figure(figsize=(10, 5))
        y_true = yc_actual.loc[dt, yc_cols].values.astype(float)
        y_pred = yc_forecast.loc[dt, yc_cols].values.astype(float)

        plt.plot(x, y_true, marker="o", color="black", linewidth=2, label="Actual YC")
        plt.plot(x, y_pred, marker="o", color="crimson", linewidth=2, label="Forecast YC")

        plt.xticks(x, yc_cols, rotation=45)
        plt.ylabel("Yield")
        plt.title(f"{title_prefix} | {dt.strftime('%Y-%m')}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

plot_yc_block(yc_test_actual, yc_var1_m1_test, "VAR type 1 | M1 no IV")
plot_yc_block(yc_test_actual, yc_var1_m2_test, "VAR type 1 | M2 with IV")
# plot_yc_block(yc_test_actual, yc_var2_m1_test, "VAR type 2 | M1 no IV")
# plot_yc_block(yc_test_actual, yc_var2_m2_test, "VAR type 2 | M2 with IV")