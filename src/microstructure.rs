//! Уровень III. Микроструктура и скачки.
//!
//! Реализует:
//!   1) Med9 RV и стандартизированные доходности (Chen et al., 2025);
//!   2) PC-IVP — piecewise constant intraday volatility patterns;
//!   3) OFI на нескольких уровнях стакана (Cont et al., 2023);
//!   4) экономический фильтр Φ > 1.

use crate::config::{MED9_C, JUMP_THRESHOLD_STD};

// =====================================================================
// 1. Med9 RV (Chen et al., 2025, ур. 11–12)
// =====================================================================

/// Локальная волатильность LV9 (Chen et al., 2025, ур. 12).
///
/// LV9_{t,j} = c_9 · med(|r_{j−4}|, …, |r_j|, …, |r_{j+4}|)²
fn local_vol_9(returns: &[f64], j: usize) -> f64 {
    if j < 4 || j + 4 >= returns.len() { return f64::NAN; }
    let mut window: Vec<f64> = (j - 4..=j + 4)
        .map(|i| returns[i].abs())
        .collect();
    window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = window[window.len() / 2];
    MED9_C * med * med
}

/// Med9RV_{t,M} (Chen et al., 2025, ур. 11).
///
/// Med9RV = (M / (M − 8)) · Σ_{j=5}^{M−4} LV9_{t,j}
pub fn med9_rv(returns: &[f64]) -> f64 {
    let m = returns.len();
    if m < 9 { return f64::NAN; }
    let mut sum = 0.0;
    let mut count = 0;
    for j in 4..(m - 4) {
        let lv = local_vol_9(returns, j);
        if lv.is_finite() {
            sum += lv;
            count += 1;
        }
    }
    if count < 1 { return f64::NAN; }
    (m as f64 / (m as f64 - 8.0)) * sum
}

/// Стандартизированные доходности (Chen et al., 2025, ур. 13).
///
/// r*_{t,j} = r_{t,j} / sqrt(Med9RV / M)
pub fn standardized_returns(returns: &[f64]) -> Vec<f64> {
    let rv = med9_rv(returns);
    if !rv.is_finite() || rv <= 0.0 { return vec![]; }
    let m = returns.len() as f64;
    let denom = (rv / m).sqrt();
    if denom < 1e-15 { return vec![]; }
    returns.iter().map(|&r| r / denom).collect()
}

/// Детекция скачков J_4: |r*| > 4 (Chen et al., 2025, Table 3).
pub fn detect_jumps(returns: &[f64], threshold: f64) -> Vec<usize> {
    let std_ret = standardized_returns(returns);
    if std_ret.is_empty() { return vec![]; }
    std_ret.iter().enumerate()
        .filter(|(_, &r)| r.abs() > threshold)
        .map(|(i, _)| i)
        .collect()
}

// =====================================================================
// 2. PC-IVP (Chen et al., 2025, стр. 12–13)
// =====================================================================

/// Паттерн внутридневной волатильности P_n.
#[derive(Debug, Clone)]
pub struct IcvPattern {
    pub n: usize,
    pub labels: Vec<i8>,   // +1 (высокая), −1 (низкая)
    pub values: Vec<f64>,
}

/// Piecewise-constant IVP: делим день на n подынтервалов, в каждом
/// считаем σ и классифицируем как высокую (+) или низкую (−) относительно медианы.
pub fn piecewise_ivp(returns_intraday: &[f64], n: usize) -> IcvPattern {
    let m = returns_intraday.len();
    if m < n || n == 0 {
        return IcvPattern { n, labels: vec![], values: vec![] };
    }
    let block = m / n;
    let mut values = Vec::with_capacity(n);
    for k in 0..n {
        let start = k * block;
        let end = if k == n - 1 { m } else { (k + 1) * block };
        let slice = &returns_intraday[start..end];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        let var = slice.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / slice.len() as f64;
        values.push(var.sqrt());
    }
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted[sorted.len() / 2];
    let labels: Vec<i8> = values.iter()
        .map(|&v| if v >= med { 1 } else { -1 })
        .collect();
    IcvPattern { n, labels, values }
}

// =====================================================================
// 3. OFI (Cont, Cucuringu & Zhang, 2023, ур. 1–4)
// =====================================================================

/// Событие стакана: обновление цены/объёма на заданном уровне.
#[derive(Debug, Clone, Copy)]
pub struct BookEvent {
    pub timestamp_ns: i64,
    pub level: usize,
    pub price: f64,
    pub size: u32,
    pub is_bid: bool,
}

/// Best-level OFI (Cont et al., 2023, ур. 1).
///
/// OFI^{1,h}_{i,t} = Σ_{n=N(t−h)+1}^{N(t)} (OF^{1,b}_{i,n} − OF^{1,a}_{i,n})
pub fn ofi_best_level(events: &[BookEvent], t_start_ns: i64, t_end_ns: i64) -> f64 {
    let mut ofi = 0.0;
    for e in events.iter().filter(|e| e.timestamp_ns > t_start_ns && e.timestamp_ns <= t_end_ns) {
        if e.level != 1 { continue; }
        // OFI считаем как знаковое изменение: рост цены bid = +,
        // падение = −; аналогично для ask
        // Упрощённо: side_bid → +size, side_ask → −size
        if e.is_bid { ofi += e.size as f64; }
        else { ofi -= e.size as f64; }
    }
    ofi
}

/// Deeper-level OFI (Cont et al., 2023, ур. 2).
pub fn ofi_level(events: &[BookEvent], level: usize, t_start_ns: i64, t_end_ns: i64) -> f64 {
    let mut ofi = 0.0;
    for e in events.iter().filter(|e| e.timestamp_ns > t_start_ns && e.timestamp_ns <= t_end_ns) {
        if e.level != level { continue; }
        if e.is_bid { ofi += e.size as f64; }
        else { ofi -= e.size as f64; }
    }
    ofi
}

/// Нормировка на среднюю глубину стакана (Cont et al., 2023, ур. 3).
pub fn ofi_scaled(ofi_raw: f64, avg_depth: f64) -> f64 {
    if avg_depth.abs() < 1e-12 { return f64::NAN; }
    ofi_raw / avg_depth
}

/// Интегрированный OFI через PCA (Cont et al., 2023, ур. 4).
///
/// of^I_{i,t} = (w_1^T · of^{(h)}_{i,t}) / ||w_1||_1
pub fn ofi_integrated(ofi_by_level: &[f64], w1: &[f64]) -> f64 {
    let l1: f64 = w1.iter().map(|v| v.abs()).sum();
    if l1 < 1e-15 { return f64::NAN; }
    let dot: f64 = ofi_by_level.iter().zip(w1.iter()).map(|(a, b)| a * b).sum();
    dot / l1
}

// =====================================================================
// 4. Экономический фильтр Φ > 1 (исходная статья, ур. 25)
// =====================================================================

/// Φ = |r_k| / S_pred_k, где S_pred = (ask − bid)/P.
pub fn economic_filter_phi(r_k: f64, spread: f64) -> Option<f64> {
    if spread <= 0.0 { None } else { Some(r_k.abs() / spread) }
}

// =====================================================================
// 5. Обёртка уровня III
// =====================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Level3Result {
    pub ticker: String,
    pub med9_rv_median: f64,
    pub n_jumps_total: usize,
    pub n_jumps_economic: usize,
    pub frac_economic: f64,
    pub stsrv_mean: f64,
    pub stsrv_ci_lo: f64,
    pub stsrv_ci_hi: f64,
    pub n_jumps_level5: usize,
    pub n_jumps_level5_economic: usize,
    pub frac_economic_level5: f64,
    pub n_ivp_u_days: usize,
    pub frac_ivp_u_days: f64,
    pub ofi_median: f64,

    // === Налоговые даты ===
    /// Число торговых дней, попавших в окно ±3 дней от налоговой даты.
    pub n_days_near_tax: usize,
    /// Число торговых дней вне окна.
    pub n_days_outside_tax: usize,
    /// Число скачков в дни около налоговых дат.
    pub n_jumps_near_tax: usize,
    /// Число скачков вне окна.
    pub n_jumps_outside_tax: usize,
    /// Средняя плотность скачков в дни около налоговых дат
    /// (скачков на день).
    pub density_near_tax: f64,
    /// Средняя плотность скачков вне окна (скачков на день).
    pub density_outside_tax: f64,
    /// Отношение плотности в окне к плотности вне окна (lift).
    /// > 1 — кластеризация, ≈ 1 — нет эффекта, < 1 — антикластеризация.
    pub tax_lift: f64,
}

/// Прогон уровня III по барам одного тикера.
///
/// `day_unix_ids` — unix-день для каждого торгового дня (тот же формат,
/// что `BarAgg::date`). Нужен для проверки кластеризации скачков
/// вокруг налоговых дат.
pub fn run_level3(
    ticker: &str,
    closes: &[f64],
    best_bids: &[Option<f64>],
    best_asks: &[Option<f64>],
    day_len: usize,
    day_unix_ids: &[i64],
) -> Level3Result {
    // Best-level OFI: аппроксимация через изменения best_bid/best_ask.
    let mut ofi_values: Vec<f64> = Vec::new();
    let mut prev_bid: Option<f64> = None;
    let mut prev_ask: Option<f64> = None;
    for i in 0..closes.len() {
        let (bid, ask) = (
            best_bids.get(i).copied().flatten(),
            best_asks.get(i).copied().flatten(),
        );
        if let (Some(b), Some(a)) = (bid, ask) {
            let mut ofi = 0.0;
            if let Some(pb) = prev_bid {
                if b > pb { ofi += 1.0; }
                else if b < pb { ofi -= 1.0; }
            }
            if let Some(pa) = prev_ask {
                if a > pa { ofi -= 1.0; }
                else if a < pa { ofi += 1.0; }
            }
            ofi_values.push(ofi);
        }
        prev_bid = bid;
        prev_ask = ask;
    }
    let ofi_median = crate::statistics::median(&ofi_values);

    let mut med9_vals: Vec<f64> = Vec::new();
    let mut total_jumps = 0;
    let mut economic_jumps = 0;
    let mut n_jumps_level5 = 0usize;
    let mut n_jumps_level5_economic = 0usize;

    let tax_dates = crate::tax_calendar::tax_dates_2018_2022();
    let tax_window_days: i64 = 3;

    let mut n_days_near_tax = 0usize;
    let mut n_days_outside_tax = 0usize;
    let mut n_jumps_near_tax = 0usize;
    let mut n_jumps_outside_tax = 0usize;

    let mut n_ivp_u_days = 0usize;
    let mut n_ivp_days = 0usize;

    let mut day_start = 0usize;
    let mut day_idx = 0usize;
    while day_start + day_len <= closes.len() {
        let day_closes = &closes[day_start..day_start + day_len];
        let returns: Vec<f64> = day_closes.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let date_unix = day_unix_ids.get(day_idx).copied().unwrap_or(0);
        let date_naive = crate::data_loading::unix_day_to_date(date_unix);
        let near_tax = crate::tax_calendar::is_near_tax_date(
            date_naive, tax_window_days, &tax_dates,
        );

        if returns.len() >= 30 {
            let n_sub = crate::config::PC_IVP_N.max(3);
            let pattern = piecewise_ivp(&returns, n_sub);
            if pattern.labels.len() == n_sub {
                n_ivp_days += 1;
                let is_u = pattern.labels[0] == 1
                    && pattern.labels[n_sub - 1] == 1
                    && pattern.labels[n_sub / 2] == -1;
                if is_u { n_ivp_u_days += 1; }
            }
        }


        if returns.len() >= 9 {
            let rv = med9_rv(&returns);
            if rv.is_finite() { med9_vals.push(rv); }

            let jumps = detect_jumps(&returns, JUMP_THRESHOLD_STD);
            total_jumps += jumps.len();

            // Параллельно — скачки уровня V (|r| > 3σ_EWMA)
            let jumps_l5 = detect_jumps_level5(&returns, crate::config::JUMP_THRESHOLD_MULT);
            n_jumps_level5 += jumps_l5.len();
            for &j in &jumps_l5 {
                let abs_ret = returns[j].abs();
                let idx = day_start + j + 1;
                if idx < best_bids.len() && idx < best_asks.len() {
                    if let (Some(bid), Some(ask)) = (best_bids[idx], best_asks[idx]) {
                        let mid = (bid + ask) / 2.0;
                        if mid > 0.0 {
                            let spread = (ask - bid) / mid;
                            if let Some(phi) = economic_filter_phi(abs_ret, spread) {
                                if phi > 1.0 { n_jumps_level5_economic += 1; }
                            }
                        }
                    }
                }
            }

            if near_tax {
                n_days_near_tax += 1;
                n_jumps_near_tax += jumps.len();
            } else {
                n_days_outside_tax += 1;
                n_jumps_outside_tax += jumps.len();
            }

            for &j in &jumps {
                let abs_ret = returns[j].abs();
                let idx = day_start + j + 1;
                if idx < best_bids.len() && idx < best_asks.len() {
                    if let (Some(bid), Some(ask)) = (best_bids[idx], best_asks[idx]) {
                        let mid = (bid + ask) / 2.0;
                        if mid > 0.0 {
                            let spread = (ask - bid) / mid;
                            if let Some(phi) = economic_filter_phi(abs_ret, spread) {
                                if phi > 1.0 { economic_jumps += 1; }
                            }
                        }
                    }
                }
            }
        }
        day_start += day_len;
        day_idx += 1;
    }

    // === Truncated S-TSRV по дневным блокам ===
    let mut stsrv_by_day: Vec<f64> = Vec::new();
    let mut day_start_s = 0usize;
    while day_start_s + day_len <= closes.len() {
        let day_prices = &closes[day_start_s..day_start_s + day_len];
        if day_prices.len() >= crate::config::STSRV_PARAMS.m_preavg + 10 {
            let log_prices: Vec<f64> = day_prices.iter().map(|&p| p.ln()).collect();
            let smoothed = crate::equations::preaverage_and_truncate(
                &log_prices,
                &crate::config::STSRV_PARAMS,
            );
            if smoothed.len() > crate::config::STSRV_PARAMS.k + crate::config::STSRV_PARAMS.j {
                let sts = crate::equations::truncated_stsrv(
                    &smoothed, &smoothed, &crate::config::STSRV_PARAMS,
                );
                if sts.is_finite() && sts > 0.0 {
                    stsrv_by_day.push(sts);
                }
            }
        }
        day_start_s += day_len;
    }

    let stsrv_mean = if stsrv_by_day.is_empty() {
        f64::NAN
    } else {
        stsrv_by_day.iter().sum::<f64>() / stsrv_by_day.len() as f64
    };

    // Блочный бутстреп по дням для CI
    let (stsrv_ci_lo, stsrv_ci_hi) = if stsrv_by_day.len() >= 20 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n_days = stsrv_by_day.len();
        let block = ((n_days as f64).powf(1.0 / 3.0)).ceil() as usize;
        let block = block.clamp(2, n_days);
        let num_blocks = (n_days + block - 1) / block;
        let mut means = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let mut sample = Vec::with_capacity(n_days);
            for _ in 0..num_blocks {
                let start = rng.gen_range(0..=(n_days - block));
                sample.extend_from_slice(&stsrv_by_day[start..start + block]);
            }
            sample.truncate(n_days);
            let m = sample.iter().sum::<f64>() / n_days as f64;
            means.push(m);
        }
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = means[(0.025 * 1000.0) as usize];
        let hi = means[(0.975 * 1000.0) as usize];
        (lo, hi)
    } else {
        (f64::NAN, f64::NAN)
    };

    let frac_economic = if total_jumps > 0 {
        economic_jumps as f64 / total_jumps as f64
    } else { f64::NAN };

    let density_near_tax = if n_days_near_tax > 0 {
        n_jumps_near_tax as f64 / n_days_near_tax as f64
    } else { f64::NAN };
    let density_outside_tax = if n_days_outside_tax > 0 {
        n_jumps_outside_tax as f64 / n_days_outside_tax as f64
    } else { f64::NAN };
    let tax_lift = if density_outside_tax > 1e-12 {
        density_near_tax / density_outside_tax
    } else { f64::NAN };

    Level3Result {
        ticker: ticker.to_string(),
        med9_rv_median: crate::statistics::median(&med9_vals),
        n_jumps_total: total_jumps,
        n_jumps_economic: economic_jumps,
        frac_economic,
        stsrv_mean,
        stsrv_ci_lo,
        stsrv_ci_hi,
        n_jumps_level5,
        n_jumps_level5_economic,
        frac_economic_level5: if n_jumps_level5 > 0 {
            n_jumps_level5_economic as f64 / n_jumps_level5 as f64
        } else {
            f64::NAN
        },
        n_ivp_u_days,
        frac_ivp_u_days: if n_ivp_days > 0 {
            n_ivp_u_days as f64 / n_ivp_days as f64
        } else {
            f64::NAN
        },
        ofi_median,
        n_days_near_tax,
        n_days_outside_tax,
        n_jumps_near_tax,
        n_jumps_outside_tax,
        density_near_tax,
        density_outside_tax,
        tax_lift,
    }
}

// =====================================================================
// 6. Внутридневная волатильность (уровень V)
// =====================================================================

/// EWMA-дисперсия RiskMetrics для уровня V.
/// σ²_t = (1−λ)·r²_{t−1} + λ·σ²_{t−1}, λ = EWMA_LAMBDA.
pub fn ewma_variance_intraday(returns: &[f64], lambda: f64) -> Vec<f64> {
    let n = returns.len();
    let mut sigma2 = vec![f64::NAN; n];
    if n == 0 { return sigma2; }
    let mut s2 = returns[0].powi(2).max(1e-12);
    for t in 0..n {
        sigma2[t] = s2;
        s2 = (1.0 - lambda) * returns[t].powi(2) + lambda * s2;
        if !s2.is_finite() || s2 <= 0.0 { s2 = 1e-12; }
    }
    sigma2
}

/// Скачки уровня V: |r_t| > 3·σ_t, где σ_t — EWMA-волатильность.
pub fn detect_jumps_level5(returns: &[f64], mult: f64) -> Vec<usize> {
    let sigma2 = ewma_variance_intraday(returns, crate::config::EWMA_LAMBDA);
    let mut jumps = Vec::new();
    for t in 0..returns.len() {
        if !sigma2[t].is_finite() || sigma2[t] <= 0.0 { continue; }
        let sigma = sigma2[t].sqrt();
        if sigma > 0.0 && returns[t].abs() > mult * sigma {
            jumps.push(t);
        }
    }
    jumps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_med9_no_jumps() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.001).sin() * 0.01).collect();
        let rv = med9_rv(&returns);
        assert!(rv.is_finite() && rv > 0.0);
    }

    #[test]
    fn test_detect_jumps() {
        let mut returns = vec![0.001; 50];
        returns[25] = 0.5; // явный скачок
        let jumps = detect_jumps(&returns, 4.0);
        assert!(jumps.contains(&25));
    }

    #[test]
    fn test_piecewise_ivp() {
        let r: Vec<f64> = (0..300).map(|i| {
            let block = i / 100;
            match block {
                0 => 0.01 * rand::random::<f64>(),
                1 => 0.001 * rand::random::<f64>(),
                2 => 0.01 * rand::random::<f64>(),
                _ => 0.0,
            }
        }).collect();
        let p = piecewise_ivp(&r, 3);
        assert_eq!(p.n, 3);
        assert_eq!(p.labels[0], 1);  // высокая
        assert_eq!(p.labels[1], -1); // низкая
        assert_eq!(p.labels[2], 1);  // высокая
    }
}