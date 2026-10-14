//! Уровень II. Эргодичность и устойчивость моментов.
//!
//! Реализует:
//!   1) f_μ(t), f_σ²(t) — отношения наблюдаемого к ожидаемому
//!      (Nerantzaki et al., 2023, стр. 3, раздел 3.2);
//!   2) Theil–Sen–Siegel estimator и Wilcoxon test (там же, стр. 5);
//!   3) обёртку для CUSUM-BWB и MF-DFA (см. equations.rs).

use crate::statistics::median;

// =====================================================================
// 1. Серия моментов на скользящем окне
// =====================================================================

/// Оценки μ и σ² на трейлинг-окне длины W.
#[derive(Debug, Clone)]
pub struct MomentSeries {
    pub mu: Vec<f64>,
    pub sigma2: Vec<f64>,
    /// Индексы начала окна (для восстановления t).
    pub start_idx: Vec<usize>,
}

/// Скользящие моменты на окне W.
///
/// Формулы из Nerantzaki et al. (2023), стр. 4:
///   μ̂_{i,t}(W) = (1/W) Σ_{s=t−W+1}^{t} r_{i,s}
///   σ̂²_{i,t}(W) = (1/(W−1)) Σ (r_{i,s} − μ̂_{i,t}(W))²
pub fn rolling_moments(returns: &[f64], window: usize) -> MomentSeries {
    let n = returns.len();
    if n < window { return MomentSeries { mu: vec![], sigma2: vec![], start_idx: vec![] }; }

    let mut mu = Vec::with_capacity(n - window + 1);
    let mut sigma2 = Vec::with_capacity(n - window + 1);
    let mut start_idx = Vec::with_capacity(n - window + 1);

    // Начальное окно
    let mut sum = 0.0;
    for &r in &returns[..window] { sum += r; }
    let mut m = sum / window as f64;
    let mut ss = 0.0;
    for &r in &returns[..window] { ss += (r - m).powi(2); }
    mu.push(m);
    sigma2.push(ss / (window as f64 - 1.0));
    start_idx.push(0);

    // Скользящее обновление
    for t in window..n {
        sum += returns[t] - returns[t - window];
        m = sum / window as f64;
        // Пересчитываем ss (для корректности; можно оптимизировать через Welford)
        ss = 0.0;
        for &r in &returns[t - window + 1..=t] { ss += (r - m).powi(2); }
        mu.push(m);
        sigma2.push(ss / (window as f64 - 1.0));
        start_idx.push(t - window + 1);
    }

    MomentSeries { mu, sigma2, start_idx }
}

// =====================================================================
// 2. Отношения наблюдаемого к ожидаемому (Nerantzaki et al., 2023)
// =====================================================================

/// Отношения f_μ(t), f_σ²(t) и их эталоны.
#[derive(Debug, Clone)]
pub struct StationarityRatios {
    pub f_mu: Vec<f64>,
    pub f_sigma2: Vec<f64>,
    pub mu_bar: f64,
    pub sigma2_bar: f64,
}

/// Отношения наблюдаемого к ожидаемому.
///
/// Формулы:
///   μ̄_i = (1/(T−W+1)) Σ_t μ̂_{i,t}(W)
///   f_μ(t) = μ̂_{i,t}(W) / μ̄_i
///   f_σ²(t) = σ̂²_{i,t}(W) / σ̄²_i
///
/// Конвенция «наблюдаемое/ожидаемое» соответствует глобальной версии
/// Nerantzaki et al. (2023, стр. 3): f_y = n_y / n̄_y.
pub fn stationarity_ratios(series: &MomentSeries) -> StationarityRatios {
    let n = series.mu.len();
    if n == 0 {
        return StationarityRatios {
            f_mu: vec![], f_sigma2: vec![],
            mu_bar: f64::NAN, sigma2_bar: f64::NAN,
        };
    }
    let mu_bar = series.mu.iter().sum::<f64>() / n as f64;
    let sigma2_bar = series.sigma2.iter().sum::<f64>() / n as f64;

    let f_mu: Vec<f64> = series.mu.iter()
        .map(|&m| if mu_bar.abs() > 1e-15 { m / mu_bar } else { f64::NAN })
        .collect();
    let f_sigma2: Vec<f64> = series.sigma2.iter()
        .map(|&s| if sigma2_bar > 1e-15 { s / sigma2_bar } else { f64::NAN })
        .collect();

    StationarityRatios { f_mu, f_sigma2, mu_bar, sigma2_bar }
}

// =====================================================================
// 3. Theil–Sen–Siegel estimator (Nerantzaki et al., 2023, стр. 5)
// =====================================================================

/// Theil–Sen–Siegel repeated medians estimator.
///
/// Формула:
///   β̂_TSS = median_i( median_j ( (x_j − x_i) / (t_j − t_i) ) )
///
/// Для больших n — O(n²). При n > 1000 рекомендуется сэмплировать.
pub fn theil_sen_siegel(x: &[f64], t: &[usize]) -> f64 {
    let n = x.len().min(t.len());
    if n < 3 { return f64::NAN; }

    // При больших n сэмплируем пары для скорости
    let max_n = 500;
    let step = if n > max_n { n / max_n } else { 1 };

    let mut all_slopes: Vec<f64> = Vec::new();
    for i in (0..n).step_by(step) {
        let mut local: Vec<f64> = Vec::new();
        for j in (0..n).step_by(step) {
            if i == j { continue; }
            let dt = (t[j] as f64) - (t[i] as f64);
            if dt.abs() < 1e-12 { continue; }
            let dx = x[j] - x[i];
            if !dx.is_finite() { continue; }
            local.push(dx / dt);
        }
        local.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if !local.is_empty() {
            let local_median = if local.len() % 2 == 0 {
                0.5 * (local[local.len()/2 - 1] + local[local.len()/2])
            } else {
                local[local.len()/2]
            };
            all_slopes.push(local_median);
        }
    }
    if all_slopes.is_empty() { return f64::NAN; }
    all_slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    median(&all_slopes)
}

// =====================================================================
// 4. Wilcoxon signed-rank test (Nerantzaki et al., 2023, стр. 5)
// =====================================================================

/// Wilcoxon signed-rank test для проверки значимости тренда.
/// Простая реализация нормальной аппроксимации для больших выборок.
///
/// H0: медиана тренда = 0.
/// Возвращает (Z-статистика, p-value, n_effective).
pub fn wilcoxon_test(residuals: &[f64]) -> (f64, f64, usize) {
    let n = residuals.len();
    if n < 10 { return (f64::NAN, f64::NAN, 0); }

    // Ранжируем |residuals|, отбрасываем нули
    let mut indexed: Vec<(usize, f64)> = residuals.iter().enumerate()
        .filter(|(_, &r)| r.abs() > 1e-15)
        .map(|(i, &r)| (i, r))
        .collect();
    if indexed.len() < 5 { return (f64::NAN, f64::NAN, 0); }

    indexed.sort_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap());

    // Средние ранги (tie correction)
    let n_eff = indexed.len();
    let mut ranks: Vec<f64> = vec![0.0; n_eff];
    let mut i = 0;
    while i < n_eff {
        let mut j = i;
        while j < n_eff && (indexed[j].1.abs() - indexed[i].1.abs()).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // средний ранг (1-indexed)
        for k in i..j { ranks[k] = avg_rank; }
        i = j;
    }

    // W+ = Σ рангов положительных residuals
    let w_plus: f64 = indexed.iter().zip(ranks.iter())
        .filter(|(_, _)| true)
        .enumerate()
        .filter(|(k, _)| indexed[*k].1 > 0.0)
        .map(|(k, _)| ranks[k])
        .sum();

    // Нормальная аппроксимация
    let n_f = n_eff as f64;
    let mu_w = n_f * (n_f + 1.0) / 4.0;
    let sigma_w = (n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0).sqrt();
    if sigma_w < 1e-12 { return (f64::NAN, f64::NAN, n_eff); }
    let z = (w_plus - mu_w) / sigma_w;

    // Двусторонний p-value через нормальное распределение
    use statrs::distribution::{ContinuousCDF, Normal};
    let nd = Normal::new(0.0, 1.0).unwrap();
    let p = 2.0 * (1.0 - nd.cdf(z.abs()));

    (z, p, n_eff)
}

// =====================================================================
// 5. Обёртка уровня II
// =====================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Level2Result {
    pub ticker: String,
    pub f_mu_slope: f64,
    pub f_sigma2_slope: f64,
    pub f_mu_slope_tss: f64,
    pub f_sigma2_slope_tss: f64,
    pub f_mu_wilcoxon_z: f64,
    pub f_mu_wilcoxon_p: f64,
    pub f_sigma2_wilcoxon_z: f64,
    pub f_sigma2_wilcoxon_p: f64,
    pub cusum_mean_pvalue: f64,
    pub cusum_sigma_pvalue: f64,
    pub mfdfa_delta_alpha: f64,
    pub variance_decay_slope: f64,
    /// t-статистика наклона f_μ (OLS).
    pub f_mu_t_stat: f64,
    /// t-статистика наклона f_σ² (OLS).
    pub f_sigma2_t_stat: f64,
    /// Значим ли наклон f_μ: |t| > 1.96.
    pub f_mu_significant: bool,
    /// Значим ли наклон f_σ²: |t| > 1.96.
    pub f_sigma2_significant: bool,
    /// Сравнение наклонов на AM и POT-сериях: (имя, slope_mu, slope_sigma2).
    pub stability_comparison: Vec<(String, f64, f64)>,
    /// HAC-SE / OLS-SE — диагностика автокорреляции.
    pub f_mu_se_ratio: f64,
    pub f_sigma2_se_ratio: f64,
    /// t стандартизованного μ̂ (не зависит от μ̄≈0).
    pub z_mu_t_stat: f64,
}

/// Прогон уровня II по лог-доходностям тикера.
pub fn run_level2(
    ticker: &str,
    log_returns: &[f64],
    window: usize,
) -> Level2Result {
    // 1. Скользящие моменты и отношения
    let series = rolling_moments(log_returns, window);
    let ratios = stationarity_ratios(&series);

    // 2. LR-наклоны f_μ, f_σ² с t-статистиками
    let t_idx: Vec<usize> = (0..ratios.f_mu.len()).collect();
    let (f_mu_slope, f_mu_t, f_mu_se_hac, f_mu_se_ols) =
        linear_slope_with_t(&t_idx, &ratios.f_mu, window);
    let (f_sigma2_slope, f_sigma2_t, f_s2_se_hac, f_s2_se_ols) =
        linear_slope_with_t(&t_idx, &ratios.f_sigma2, window);
    let z_mu = mu_standardized(&series.mu, ratios.sigma2_bar, window);
    let (_, z_mu_t, _, _) = linear_slope_with_t(&t_idx, &z_mu, window);

    // 3. TSS-наклоны
    let f_mu_slope_tss = theil_sen_siegel(&ratios.f_mu, &t_idx);
    let f_sigma2_slope_tss = theil_sen_siegel(&ratios.f_sigma2, &t_idx);

    // 4. Wilcoxon на отклонениях от 1
    let f_mu_dev: Vec<f64> = ratios.f_mu.iter().map(|&v| v - 1.0).collect();
    let f_sigma2_dev: Vec<f64> = ratios.f_sigma2.iter().map(|&v| v - 1.0).collect();
    let (f_mu_w_z, f_mu_w_p, _) = wilcoxon_test(&f_mu_dev);
    let (f_sigma2_w_z, f_sigma2_w_p, _) = wilcoxon_test(&f_sigma2_dev);

    // 5. CUSUM-BWB
    let (_, cusum_mu_p) = crate::equations::cusum_bwb_mean_test(
        log_returns, None, crate::config::CUSUM_BOOTSTRAP_B);
    let (_, cusum_sigma_p) = crate::equations::cusum_bwb_variance_test(
        log_returns, None, crate::config::CUSUM_BOOTSTRAP_B);

    // 6. MF-DFA
    let mf = crate::equations::mf_dfa(log_returns, -5.0, 5.0, 0.5, None, 1);
    let mfdfa_delta_alpha = mf.delta_alpha;

    // 7. Variance decay: наклон log Var(r̄_m) против log m
    let vd = crate::equations::variance_decay_check(log_returns, 50, 200);
    let log_m: Vec<f64> = vd.m_values.iter().map(|&m| (m as f64).ln()).collect();
    let log_ratio: Vec<f64> = vd.ratio.iter().map(|&r| if r > 0.0 { r.ln() } else { f64::NAN }).collect();
    let valid: Vec<usize> = log_ratio.iter().enumerate()
        .filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect();
    let log_m_v: Vec<f64> = valid.iter().map(|&i| log_m[i]).collect();
    let log_r_v: Vec<f64> = valid.iter().map(|&i| log_ratio[i]).collect();
    let variance_decay_slope = linear_slope_f64(&log_m_v, &log_r_v);

    let stability_comparison = compare_stability_across_series(
        log_returns, window, 390,
    );

    Level2Result {
        ticker: ticker.to_string(),
        f_mu_slope,
        f_sigma2_slope,
        f_mu_slope_tss,
        f_sigma2_slope_tss,
        f_mu_wilcoxon_z: f_mu_w_z,
        f_mu_wilcoxon_p: f_mu_w_p,
        f_sigma2_wilcoxon_z: f_sigma2_w_z,
        f_sigma2_wilcoxon_p: f_sigma2_w_p,
        cusum_mean_pvalue: cusum_mu_p,
        cusum_sigma_pvalue: cusum_sigma_p,
        mfdfa_delta_alpha,
        variance_decay_slope,
        f_mu_t_stat: f_mu_t,
        f_sigma2_t_stat: f_sigma2_t,
        f_mu_significant: f_mu_t.abs() > 1.96,
        f_sigma2_significant: f_sigma2_t.abs() > 1.96,
        stability_comparison,
        f_mu_se_ratio: if f_mu_se_ols > 1e-15 { f_mu_se_hac / f_mu_se_ols } else { f64::NAN },
        f_sigma2_se_ratio: if f_s2_se_ols > 1e-15 { f_s2_se_hac / f_s2_se_ols } else { f64::NAN },
        z_mu_t_stat: z_mu_t,
    }
}

/// OLS-наклон с HAC-t (Newey–West, ядро Бартлетта).
/// `min_lag` ≥ ширины окна W: соседние значения f(t) перекрываются на W−1 точек.
/// Возвращает (slope, t_hac, se_hac, se_ols).
fn linear_slope_with_t(t: &[usize], x: &[f64], min_lag: usize) -> (f64, f64, f64, f64) {
    let nan = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    let n = t.len().min(x.len());
    if n < 5 { return nan; }

    let pairs: Vec<(f64, f64)> = t[..n].iter().zip(x[..n].iter())
        .filter(|(_, &xv)| xv.is_finite())
        .map(|(&tv, &xv)| (tv as f64, xv))
        .collect();
    let m = pairs.len();
    if m < 10 { return nan; }

    let mt = pairs.iter().map(|p| p.0).sum::<f64>() / m as f64;
    let mx = pairs.iter().map(|p| p.1).sum::<f64>() / m as f64;
    let (mut sxx, mut sxy) = (0.0, 0.0);
    for (tv, xv) in &pairs {
        sxx += (tv - mt) * (tv - mt);
        sxy += (tv - mt) * (xv - mx);
    }
    if sxx < 1e-15 { return nan; }
    let slope = sxy / sxx;
    let intercept = mx - slope * mt;
    let resid: Vec<f64> = pairs.iter().map(|(tv, xv)| xv - intercept - slope * tv).collect();

    // Классический SE — для диагностики.
    let ss_res: f64 = resid.iter().map(|e| e * e).sum();
    let se_ols = (ss_res / (m as f64 - 2.0).max(1.0) / sxx).sqrt();

    // HAC: Var(β̂) = Ω / Sxx², где Ω = Σ_t Σ_s u_t u_s — именно СУММА, не среднее.
    // Прежняя формула s_hac/(sxx/m)² завышала SE в √m раз, t занижался в √m.
    let nw = (4.0 * (m as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize;
    let lag = nw.max(2 * min_lag).min(m - 1).max(1);
    let u: Vec<f64> = (0..m).map(|i| (pairs[i].0 - mt) * resid[i]).collect();
    let mut omega = 0.0;
    for l in 0..=lag {
        let w = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
        let mut g = 0.0;
        for tt in l..m { g += u[tt] * u[tt - l]; }
        omega += if l == 0 { g } else { 2.0 * w * g };
    }
    if !(omega > 0.0 && omega.is_finite()) {
        omega = u.iter().map(|v| v * v).sum();
    }
    let se_hac = omega.sqrt() / sxx;
    let t_stat = if se_hac > 1e-15 { slope / se_hac } else { f64::NAN };

    (slope, t_stat, se_hac, se_ols)
}

/// μ̂ в единицах собственной шумовой ошибки: z_t = μ̂_t · √W / σ̄.
/// Вместо f_μ = μ̂/μ̄, который взрывается при μ̄ ≈ 0.
pub fn mu_standardized(mu: &[f64], sigma2_bar: f64, w: usize) -> Vec<f64> {
    let k = (w as f64).sqrt() / sigma2_bar.sqrt();
    mu.iter().map(|&m| m * k).collect()
}

fn linear_slope_f64(t: &[f64], x: &[f64]) -> f64 {
    let n = t.len().min(x.len());
    if n < 3 { return f64::NAN; }
    let mt = t.iter().sum::<f64>() / n as f64;
    let mx = x.iter().sum::<f64>() / n as f64;
    let mut num = 0.0; let mut den = 0.0;
    for i in 0..n {
        num += (t[i] - mt) * (x[i] - mx);
        den += (t[i] - mt).powi(2);
    }
    if den.abs() > 1e-15 { num / den } else { f64::NAN }
}

// =====================================================================
// 6. AM и POT серии (Nerantzaki et al., 2023, раздел 3.3)
// =====================================================================

/// AM-серия: один экстремум в год.
pub fn annual_maxima_series(returns: &[f64], day_len: usize) -> Vec<f64> {
    let days_per_year = 252usize;
    let year_len = day_len * days_per_year;
    if returns.len() < year_len { return vec![]; }
    let n_years = returns.len() / year_len;
    let mut out = Vec::with_capacity(n_years);
    for y in 0..n_years {
        let start = y * year_len;
        let end = start + year_len;
        let max_abs = returns[start..end].iter()
            .map(|r| r.abs())
            .fold(0.0_f64, f64::max);
        out.push(max_abs);
    }
    out
}

/// POT-серия: значения выше p-квантиля |r|.
pub fn pot_series(returns: &[f64], percentile: f64) -> Vec<f64> {
    if returns.is_empty() { return vec![]; }
    let mut abs_ret: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
    abs_ret.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((percentile * abs_ret.len() as f64) as usize)
        .min(abs_ret.len() - 1);
    let threshold = abs_ret[idx];
    returns.iter()
        .filter(|r| r.abs() > threshold)
        .copied()
        .collect()
}

/// Сравнение наклонов f_μ и f_σ² на полной, AM- и POT-сериях.
pub fn compare_stability_across_series(
    log_returns: &[f64],
    window: usize,
    day_len: usize,
) -> Vec<(String, f64, f64)> {
    let mut out = Vec::new();

    let series_list: Vec<(&str, Vec<f64>)> = vec![
        ("full", log_returns.to_vec()),
        ("AM", annual_maxima_series(log_returns, day_len)),
        ("POT90", pot_series(log_returns, 0.90)),
        ("POT95", pot_series(log_returns, 0.95)),
        ("POT99", pot_series(log_returns, 0.99)),
    ];

    for (name, series) in series_list {
        if series.len() < window + 10 { continue; }
        let s = rolling_moments(&series, window);
        let r = stationarity_ratios(&s);
        let t_idx: Vec<usize> = (0..r.f_mu.len()).collect();
        let (slope_mu, _, _, _) = linear_slope_with_t(&t_idx, &r.f_mu, window);
        let (slope_sig2, _, _, _) = linear_slope_with_t(&t_idx, &r.f_sigma2, window);
        out.push((name.to_string(), slope_mu, slope_sig2));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_moments_constant() {
        let r = vec![0.0; 100];
        let s = rolling_moments(&r, 10);
        assert!(s.mu.iter().all(|&m| m.abs() < 1e-12));
        assert!(s.sigma2.iter().all(|&v| v.abs() < 1e-12));
    }

    #[test]
    fn test_stationarity_ratios() {
        // На iid данных отношения должны быть около 1
        let r: Vec<f64> = (0..500).map(|_| rand::random::<f64>() - 0.5).collect();
        let s = rolling_moments(&r, 50);
        let ratios = stationarity_ratios(&s);
        let mean_f_mu: f64 = ratios.f_mu.iter().sum::<f64>() / ratios.f_mu.len() as f64;
        assert!((mean_f_mu - 1.0).abs() < 0.3);
    }

    #[test]
    fn test_tss_linear() {
        // Идеальная линейная зависимость x_t = 2t
        let x: Vec<f64> = (0..20).map(|i| 2.0 * i as f64).collect();
        let t: Vec<usize> = (0..20).collect();
        let slope = theil_sen_siegel(&x, &t);
        assert!((slope - 2.0).abs() < 1e-6);
    }
}