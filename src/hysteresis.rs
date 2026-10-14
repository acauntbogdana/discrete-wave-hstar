//! Уровень VI. Гистерезисная надстройка.
//!
//! Реэкспортирует стек разворотов из `dwt`, VaR-модели из `var_models`
//! и DM/GDM из `dm_test`. Здесь остаётся только обёртка `run_level6`.

// Реэкспорты из существующих модулей
pub use crate::dwt::{Extreme, Kind, OnlineExtrema, push_into_stack};
pub use crate::var_models::{dwt_volatility, ewma_variance, quantile_loss, var_forecast};
pub use crate::dm_test::{dm_statistic, gdm_statistic};

use crate::config::{DELTA_REL, VAR_LEVEL};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Level6Result {
    pub ticker: String,
    pub n_regimes: usize,
    pub mean_regime_len_bars: f64,
    pub regimes_per_100k: f64,
    pub dm_by_h: Vec<(f64, f64)>,
    pub gdm_stat: f64,
    pub gdm_pvalue: f64,
}

/// Прогон уровня VI по одному тикеру.
pub fn run_level6(
    ticker: &str,
    closes: &[f64],
    horizons: &[usize],
) -> Level6Result {
    // 1. Строим стек онлайн
    let mut extrema = OnlineExtrema::new(DELTA_REL);
    let mut states_per_bar: Vec<Vec<Extreme>> = Vec::with_capacity(closes.len());
    let mut regime_ids: Vec<usize> = Vec::with_capacity(closes.len());
    let mut regime_id = 0usize;

    for &p in closes {
        let (stack, wiped) = extrema.push(p);
        if wiped {
            regime_id += 1;
        }
        states_per_bar.push(stack.to_vec());
        regime_ids.push(regime_id);
    }

    // 2. Диагностика режимов
    let n_regimes = regime_id;
    let n_obs = closes.len();
    let mean_regime_len_bars = if n_regimes > 0 {
        n_obs as f64 / (n_regimes as f64 + 1.0)
    } else {
        n_obs as f64
    };
    let regimes_per_100k = if n_obs > 0 {
        n_regimes as f64 / n_obs as f64 * 100_000.0
    } else {
        f64::NAN
    };

    // 3. DM-тест VaR
    let log_returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    let sigma2_ewma = ewma_variance(&log_returns, 0.94);

    let tau = VAR_LEVEL;
    let warmup = 200;
    let h_max = horizons.iter().copied().max().unwrap_or(0);

    let mut dm_by_h: Vec<(f64, f64)> = Vec::with_capacity(horizons.len());
    let mut d_mat: Vec<Vec<f64>> = vec![Vec::new(); horizons.len()];

    if log_returns.len() > warmup + h_max + 1 {
        for t in warmup..(log_returns.len() - h_max) {
            let sigma_ewma = sigma2_ewma[t].sqrt();
            let stack = &states_per_bar[t];
            // Вычисляем DWT-волатильность через публичную функцию
            // из var_models.rs: она принимает DwtState, но у нас Vec<Extreme>.
            // Используем простую внутреннюю формулу, совместимую с var_models:
            let sigma_dwt = dwt_volatility_from_stack(stack);
            if !sigma_dwt.is_finite() || !sigma_ewma.is_finite() {
                continue;
            }

            for (h_idx, &h) in horizons.iter().enumerate() {
                let r_forward: f64 = log_returns[t..t + h].iter().sum();
                if !r_forward.is_finite() {
                    continue;
                }
                let v_dwt = var_forecast(0.0, sigma_dwt, h, tau);
                let v_ewma = var_forecast(0.0, sigma_ewma, h, tau);
                if !v_dwt.is_finite() || !v_ewma.is_finite() {
                    continue;
                }
                let l_dwt = quantile_loss(r_forward, v_dwt, tau);
                let l_ewma = quantile_loss(r_forward, v_ewma, tau);
                d_mat[h_idx].push(l_dwt - l_ewma);
            }
        }
    }

    // 4. DM-статистика по горизонтам
    for h_idx in 0..horizons.len() {
        if d_mat[h_idx].len() < 30 {
            dm_by_h.push((f64::NAN, f64::NAN));
            continue;
        }
        let (dm, p) = dm_statistic(&d_mat[h_idx]);
        dm_by_h.push((dm, p));
    }

    // 5. GDM
    let (gdm_stat, gdm_pvalue) = gdm_statistic(&d_mat);

    Level6Result {
        ticker: ticker.to_string(),
        n_regimes,
        mean_regime_len_bars,
        regimes_per_100k,
        dm_by_h,
        gdm_stat,
        gdm_pvalue,
    }
}

/// Вспомогательная функция: RMS-амплитуда / sqrt(mean_gap) из Vec<Extreme>.
/// Дублирует логику `var_models::dwt_volatility`, но принимает срез, а не DwtState.
fn dwt_volatility_from_stack(stack: &[Extreme]) -> f64 {
    if stack.len() < 2 {
        return f64::NAN;
    }
    let mut sum_a2: f64 = 0.0;   // ← явный тип
    let mut sum_g: f64 = 0.0;    // ← явный тип
    let m = stack.len() - 1;
    for w in stack.windows(2) {
        let a: f64 = (w[1].x / w[0].x).ln().abs();
        let g: f64 = (w[1].t as f64 - w[0].t as f64).max(1.0);
        sum_a2 += a * a;
        sum_g += g;
    }
    let sigma_s: f64 = (sum_a2 / m as f64).sqrt();
    let g_bar: f64 = sum_g / m as f64;
    if g_bar < 1e-12 {
        return f64::NAN;
    }
    sigma_s / g_bar.sqrt()
}