//! Гипотеза H5. Ускорение эксцесса как escape из барьера
//! Гроссмана–Миллера / Стиглица.
//!
//! H5a. Доля скачков с Φ > 1 в блоке j растёт с Ã_{K,j−1}.
//! H5b. Знаковая доходность преодолевших скачков продолжается
//!      (не разворачивается) при высоком Ã_{K,j−1}.
//!
//! Строгость как у H4: непересекающиеся блоки, HAC по (x·e),
//! плацебо-перестановка блоков, профиль сдвигов, BH-поправка,
//! winsorization 1/99, чувствительность по W.

use crate::factors::leverage::{
    BlockGrid, LeverageConfig, OlsHac, Rng,
    build_grid, norm_sf, nw_lag, ols_hac,
    standardize, winsorize,
};

/// Один скачок.
#[derive(Debug, Clone, Copy)]
pub struct JumpRecord {
    /// Глобальный индекс в log_returns.
    pub idx: usize,
    /// Знак скачка: +1 или −1.
    pub direction: f64,
    /// Экономический фильтр Φ = |r| / spread.
    pub phi: f64,
    /// Φ > 1.
    pub passed: bool,
}

/// Результат H5 по одному тикеру.
#[derive(Debug, Clone)]
pub struct EscapeResult {
    pub n_blocks: usize,
    // H5a: доля преодолевших скачков
    pub frac_beta: f64,
    pub frac_se: f64,
    pub frac_t: f64,
    pub frac_perm_p: f64,
    pub frac_p_asym: f64,
    // H5b: знаковая доходность
    pub mom_beta: f64,
    pub mom_se: f64,
    pub mom_t: f64,
    pub mom_perm_p: f64,
    pub mom_p_asym: f64,
    // Комбинированный вывод
    pub is_escape: bool,
    // Диагностика
    pub profile_frac: Vec<(i32, f64)>,
    pub profile_mom: Vec<(i32, f64)>,
    pub sens_frac: Vec<(usize, f64)>,
    pub sens_mom: Vec<(usize, f64)>,
}

// =====================================================================
// Внутренние структуры
// =====================================================================

struct H5Rows {
    y_frac: Vec<f64>,
    y_mom: Vec<f64>,   // может содержать NaN
    a: Vec<f64>,       // стандартизованное Ã
    jidx: Vec<usize>,  // индекс блока для каждой строки (для сдвигов)
}

/// Строит блочные ряды H5a и H5b. Регрессор — Ã_{K,j-1}, отклик — блок j.
fn build_h5_rows(
    returns: &[f64],
    jumps: &[JumpRecord],
    g: &BlockGrid,
    cfg: &LeverageConfig,
    horizon_blocks: usize,
) -> Option<H5Rows> {
    let w = cfg.window;
    let nb = g.k.len();
    if nb < 10 {
        return None;
    }

    // Кумулятивная лог-цена.
    let mut cum: Vec<f64> = Vec::with_capacity(returns.len() + 1);
    cum.push(0.0);
    let mut s = 0.0;
    for &r in returns {
        s += r;
        cum.push(s);
    }

    let mut total_in_block = vec![0usize; nb];
    let mut passed_in_block = vec![0usize; nb];
    let mut mom_sum = vec![0.0_f64; nb];
    let mut mom_cnt = vec![0usize; nb];

    for jr in jumps {
        let b = jr.idx / w;
        if b >= nb {
            continue;
        }
        total_in_block[b] += 1;
        if jr.passed {
            passed_in_block[b] += 1;
            let t_end = (jr.idx + horizon_blocks * w).min(returns.len());
            if t_end > jr.idx {
                let signed = jr.direction * (cum[t_end] - cum[jr.idx]);
                mom_sum[b] += signed;
                mom_cnt[b] += 1;
            }
        }
    }

    let acc = |j: isize| -> f64 {
        if j < 2 || j as usize >= nb {
            return f64::NAN;
        }
        let j = j as usize;
        g.k[j] - 2.0 * g.k[j - 1] + g.k[j - 2]
    };

    let (mut y_frac, mut y_mom, mut a, mut jidx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for j in 2..nb {
        let total_j = total_in_block[j];
        if total_j == 0 {
            continue;
        }
        let a_prev = acc(j as isize - 1);
        if !a_prev.is_finite() {
            continue;
        }
        y_frac.push(passed_in_block[j] as f64 / total_j as f64);
        if mom_cnt[j] > 0 {
            y_mom.push(mom_sum[j] / mom_cnt[j] as f64);
        } else {
            y_mom.push(f64::NAN);
        }
        a.push(a_prev);
        jidx.push(j);
    }

    if y_frac.len() < cfg.min_obs {
        return None;
    }
    standardize(&mut a);
    Some(H5Rows { y_frac, y_mom, a, jidx })
}

/// OLS y_c = β·a_c + e, с центрированием и winsorization y/a.
/// Возвращает OlsHac, где coef[0] = β, se[0] = SE(β), t[0] = t_β.
fn fit_centered(y: &[f64], a: &[f64], winsor_p: f64) -> Option<OlsHac> {
    let n = y.len().min(a.len());
    if n < 10 {
        return None;
    }
    let mut yv: Vec<f64> = Vec::with_capacity(n);
    let mut av: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if y[i].is_finite() && a[i].is_finite() {
            yv.push(y[i]);
            av.push(a[i]);
        }
    }
    let m = yv.len();
    if m < 10 {
        return None;
    }
    if winsor_p > 0.0 {
        winsorize(&mut yv, winsor_p);
        winsorize(&mut av, winsor_p);
    }
    let my = yv.iter().sum::<f64>() / m as f64;
    let ma = av.iter().sum::<f64>() / m as f64;
    let yc: Vec<f64> = yv.iter().map(|v| v - my).collect();
    let ac: Vec<f64> = av.iter().map(|v| v - ma).collect();
    ols_hac(&yc, &[&ac], nw_lag(m))
}

/// Плацебо: перестановка Ã блоками длины `perm_block`; y фиксирован.
fn perm_p_h5(y: &[f64], a: &[f64], t_obs: f64, cfg: &LeverageConfig) -> f64 {
    let n = y.len().min(a.len());
    if n < 20 {
        return f64::NAN;
    }
    let mut yv: Vec<f64> = Vec::with_capacity(n);
    let mut av: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        if y[i].is_finite() && a[i].is_finite() {
            yv.push(y[i]);
            av.push(a[i]);
        }
    }
    let m = yv.len();
    if m < 20 {
        return f64::NAN;
    }
    let my = yv.iter().sum::<f64>() / m as f64;
    let yc: Vec<f64> = yv.iter().map(|v| v - my).collect();

    let b = cfg.perm_block.max(1);
    let nblk = (m + b - 1) / b;
    let mut rng = Rng(cfg.seed ^ (m as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let mut order: Vec<usize> = (0..nblk).collect();
    let mut a_perm = vec![0.0_f64; m];
    let lag = nw_lag(m);
    let mut ge = 0usize;
    let mut done = 0usize;
    for _ in 0..cfg.n_perm {
        for i in (1..nblk).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut p = 0;
        for &bk in &order {
            for i in bk * b..((bk + 1) * b).min(m) {
                a_perm[p] = av[i];
                p += 1;
            }
        }
        let ma = a_perm.iter().sum::<f64>() / m as f64;
        for v in a_perm.iter_mut() {
            *v -= ma;
        }
        if let Some(f) = ols_hac(&yc, &[&a_perm], lag) {
            if f.t[0].is_finite() {
                done += 1;
                if f.t[0] >= t_obs {
                    ge += 1;
                }
            }
        }
    }
    if done == 0 {
        f64::NAN
    } else {
        (1 + ge) as f64 / (1 + done) as f64
    }
}

// =====================================================================
// Верхний уровень
// =====================================================================

pub fn run_escape(
    returns: &[f64],
    jumps: &[JumpRecord],
    cfg: &LeverageConfig,
    bad: Option<&[bool]>,
    horizon_blocks: usize,
) -> Option<EscapeResult> {
    if jumps.is_empty() {
        return None;
    }
    let g = build_grid(returns, cfg, bad);
    let rows = build_h5_rows(returns, jumps, &g, cfg, horizon_blocks)?;

    // H5a
    let frac_fit = fit_centered(&rows.y_frac, &rows.a, cfg.winsor_p)?;
    let frac_beta = frac_fit.coef[0];
    let frac_se = frac_fit.se[0];
    let frac_t = frac_fit.t[0];
    let frac_perm_p = perm_p_h5(&rows.y_frac, &rows.a, frac_t, cfg);
    let frac_p_asym = norm_sf(frac_t);

    // H5b
    let mom_fit = fit_centered(&rows.y_mom, &rows.a, cfg.winsor_p);
    let (mom_beta, mom_se, mom_t, mom_perm_p, mom_p_asym) = match &mom_fit {
        Some(f) => (
            f.coef[0],
            f.se[0],
            f.t[0],
            perm_p_h5(&rows.y_mom, &rows.a, f.t[0], cfg),
            norm_sf(f.t[0]),
        ),
        None => (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN),
    };

    // Профиль сдвигов: shift = -1, 0, +1
    let build_shifted_a = |shift: i32| -> Vec<f64> {
        let mut alt = rows.a.clone();
        for (i, &jj) in rows.jidx.iter().enumerate() {
            let target = (jj as isize - 1 + shift as isize) as isize;
            let av = if target >= 2 && (target as usize) < g.k.len() {
                let tj = target as usize;
                g.k[tj] - 2.0 * g.k[tj - 1] + g.k[tj - 2]
            } else {
                f64::NAN
            };
            alt[i] = av;
        }
        standardize(&mut alt);
        alt
    };

    let mut profile_frac = Vec::new();
    let mut profile_mom = Vec::new();
    for sh in -1..=1 {
        let a_shift = if sh == 0 { rows.a.clone() } else { build_shifted_a(sh) };
        let tf = fit_centered(&rows.y_frac, &a_shift, cfg.winsor_p)
            .map(|f| f.t[0])
            .unwrap_or(f64::NAN);
        profile_frac.push((sh, tf));
        let tm = fit_centered(&rows.y_mom, &a_shift, cfg.winsor_p)
            .map(|f| f.t[0])
            .unwrap_or(f64::NAN);
        profile_mom.push((sh, tm));
    }

    // Чувствительность по W
    let mut sens_frac = Vec::new();
    let mut sens_mom = Vec::new();
    for &w2 in &[cfg.window / 2, cfg.window, cfg.window * 2] {
        let mut c2 = cfg.clone();
        c2.window = w2;
        let g2 = build_grid(returns, &c2, bad);
        if let Some(r2) = build_h5_rows(returns, jumps, &g2, &c2, horizon_blocks) {
            if let Some(f2) = fit_centered(&r2.y_frac, &r2.a, c2.winsor_p) {
                sens_frac.push((w2, f2.t[0]));
            }
            if let Some(f2) = fit_centered(&r2.y_mom, &r2.a, c2.winsor_p) {
                sens_mom.push((w2, f2.t[0]));
            }
        }
    }

    // Критерий escape.
    let is_escape = frac_t > 1.645
        && frac_perm_p <= 0.05
        && (mom_t.is_nan() || (mom_t > 1.645 && mom_perm_p <= 0.05));

    Some(EscapeResult {
        n_blocks: rows.y_frac.len(),
        frac_beta,
        frac_se,
        frac_t,
        frac_perm_p,
        frac_p_asym,
        mom_beta,
        mom_se,
        mom_t,
        mom_perm_p,
        mom_p_asym,
        is_escape,
        profile_frac,
        profile_mom,
        sens_frac,
        sens_mom,
    })
}