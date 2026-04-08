# ============================================================
# Dynamic Nelson–Siegel + Kalman filter для твоего yield_curve
# ============================================================

library(tidyverse)
library(lubridate)

# --- 1. Чтение данных из CSV, подготовленного в Python -----

yields_df <- readxl::read_excel("data/yield_curve.xlsx")
yields_df$date <- ymd(yields_df$Month)

# Ожидается структура: date,1W,2W,1M,2M,3M,6M,1Y,2Y
print(colnames(yields_df))

# Вектор сроков в годах (в том же порядке, что и колонки)
tau <- c(
  7/365,
  14/365,
  1/12,
  2/12,
  3/12,
  6/12,
  1.0,
  2.0
)

y_mat  <- as.matrix(yields_df[ , c("1W","2W","1M","2M","3M","6M","1Y","2Y")])
dates  <- yields_df$date
T_obs  <- nrow(y_mat)
N_tau  <- length(tau)

# --- 2. Настройки DNS ---------------------------------------

# Фиксируем λ (можно взять 0.7308 или найденную тобой)
lambda <- 0.7308

ns_loadings <- function(tau, lambda) {
  x <- tau / lambda
  x[abs(x) < 1e-8] <- 1e-8
  L1 <- (1 - exp(-x)) / x
  L2 <- L1 - exp(-x)
  list(L1 = L1, L2 = L2)
}

L <- ns_loadings(tau, lambda)
L1 <- L$L1
L2 <- L$L2

# Матрица измерений H: y_t = H %*% beta_t + eps_t
# beta_t = (beta0, beta1, beta2)'
H <- cbind(rep(1, N_tau), L1, L2)   # N_tau x 3

# --- 3. Задаём динамику β и начальные параметры --------------

k_state <- 3  # три фактора: уровень, наклон, кривизна

# начальное состояние – можно взять средний уровень и нули
beta_init <- c(mean(y_mat[,1], na.rm = TRUE), 0, 0)
P_init    <- diag(0.1, k_state)

# Простая динамика: beta_t = c + F * beta_{t-1} + u_t
F_mat <- diag(c(0.8, 0.8, 0.8))  # можно позже подстроить
c_vec <- rep(0, k_state)

# Дисперсии шумов состояния (настройка грубая, но рабочая)
Q_mat <- diag(c(0.01, 0.01, 0.01))

# Ковариация шума наблюдений: пропорциональна дисперсии доходностей по срокам
R_diag <- apply(y_mat, 2, function(col) var(col, na.rm = TRUE) * 0.1)
R_mat  <- diag(R_diag)

# --- 4. Фильтр Калмана для DNS -------------------------------

kalman_dns <- function(y_mat, H, F_mat, c_vec, Q_mat, R_mat,
                       beta_init, P_init) {
  T_obs <- nrow(y_mat)
  N_tau <- ncol(y_mat)
  k_state <- length(beta_init)

  beta_filt <- matrix(NA_real_, nrow = T_obs, ncol = k_state)
  beta_pred <- matrix(NA_real_, nrow = T_obs, ncol = k_state)
  P_filt_list <- vector("list", T_obs)
  P_pred_list <- vector("list", T_obs)

  beta_t_t <- beta_init
  P_t_t    <- P_init

  for (t in 1:T_obs) {
    y_t <- y_mat[t, ]

    # Шаг предсказания состояния
    beta_t_tmin1 <- c_vec + F_mat %*% beta_t_t
    P_t_tmin1    <- F_mat %*% P_t_t %*% t(F_mat) + Q_mat

    # Учитываем только те сроки, по которым есть наблюдения
    mask   <- !is.na(y_t)
    H_t    <- H[mask, , drop = FALSE]
    y_t_obs <- y_t[mask]

    if (length(y_t_obs) > 0) {
      # предсказанное наблюдение
      y_pred <- as.vector(H_t %*% beta_t_tmin1)

      # innovation covariance
      R_t <- R_mat[mask, mask, drop = FALSE]
      S_t <- H_t %*% P_t_tmin1 %*% t(H_t) + R_t

      # калмановский коэффициент
      K_t <- P_t_tmin1 %*% t(H_t) %*% solve(S_t)

      # обновление
      beta_t_t <- beta_t_tmin1 + K_t %*% (y_t_obs - y_pred)
      P_t_t    <- (diag(k_state) - K_t %*% H_t) %*% P_t_tmin1
    } else {
      beta_t_t <- beta_t_tmin1
      P_t_t    <- P_t_tmin1
    }

    beta_pred[t, ] <- as.vector(beta_t_tmin1)
    beta_filt[t, ] <- as.vector(beta_t_t)
    P_pred_list[[t]] <- P_t_tmin1
    P_filt_list[[t]] <- P_t_t
  }

  list(
    beta_filt = beta_filt,
    beta_pred = beta_pred,
    P_filt    = P_filt_list,
    P_pred    = P_pred_list
  )
}

kf_res <- kalman_dns(
  y_mat    = y_mat,
  H        = H,
  F_mat    = F_mat,
  c_vec    = c_vec,
  Q_mat    = Q_mat,
  R_mat    = R_mat,
  beta_init = beta_init,
  P_init    = P_init
)

beta_filt <- kf_res$beta_filt   # T x 3: beta0, beta1, beta2 (filtered)
beta_pred <- kf_res$beta_pred   # T x 3: one-step-ahead предсказания

colnames(beta_filt) <- c("beta0", "beta1", "beta2")
colnames(beta_pred) <- c("beta0_pred", "beta1_pred", "beta2_pred")

# --- 5. Восстановление fitted и прогнозных кривых ------------

ns_yield <- function(tau, beta0, beta1, beta2, lambda) {
  L <- ns_loadings(tau, lambda)
  beta0 + beta1 * L$L1 + beta2 * L$L2
}

y_fitted   <- matrix(NA_real_, nrow = T_obs, ncol = N_tau)
y_forecast <- matrix(NA_real_, nrow = T_obs, ncol = N_tau)

for (t in 1:T_obs) {
  b  <- beta_filt[t, ]
  bp <- beta_pred[t, ]

  y_fitted[t, ]   <- ns_yield(tau, b[1],  b[2],  b[3],  lambda)
  y_forecast[t, ] <- ns_yield(tau, bp[1], bp[2], bp[3], lambda)
}

colnames(y_fitted)   <- colnames(y_mat)
colnames(y_forecast) <- paste0(colnames(y_mat), "_fcst")

# --- 5a. Ошибка fit по датам (RMSE, MAE) ----------------------

errors_fit <- y_mat - y_fitted

rmse_fit_by_time <- apply(errors_fit, 1, function(e) {
  e <- e[!is.na(e)]
  if (length(e) == 0) return(NA_real_)
  sqrt(mean(e^2))
})

mae_fit_by_time <- apply(errors_fit, 1, function(e) {
  e <- e[!is.na(e)]
  if (length(e) == 0) return(NA_real_)
  mean(abs(e))
})

# --- 6. Собираем таблицы для экспорта -------------------------

betas_df <- cbind(
  date  = dates,
  as.data.frame(beta_filt),
  rmse_fit = rmse_fit_by_time,
  mae_fit  = mae_fit_by_time
)

fitted_df <- cbind(
  date = dates,
  as.data.frame(y_fitted)
)

forecast_df <- cbind(
  date = dates,
  as.data.frame(y_forecast)
)

# --- 8. Сохранение результатов --------------------------------

write.csv(betas_df,   "data/ns_results/ns_betas_kalman.csv",    row.names = FALSE)
write.csv(fitted_df,  "data/ns_results/ns_fitted_kalman.csv",   row.names = FALSE)
write.csv(forecast_df,"data/ns_results/ns_forecast_kalman.csv", row.names = FALSE)

cat("Готово: сохранены ns_betas_kalman.csv, ns_fitted_kalman.csv, ns_forecast_kalman.csv\n")