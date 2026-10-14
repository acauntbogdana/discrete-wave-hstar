//! Уровень IV (v2). Ускорение эксцесса как рычаг — переработанный тест.
//!
//! Что исправлено относительно прежней версии:
//!   1) ВРЕМЕННО́Е ВЫРАВНИВАНИЕ. Все регрессоры (K, σ², Ä_K) строятся по окнам,
//!      ЗАКАНЧИВАЮЩИМСЯ в момент t. Отклик — доходность СТРОГО после t.
//!      Выравнивание делается по явным индексам блоков, а не по min() длин.
//!   2) БЕЗ ПЕРЕКРЫТИЯ. Окна не перекрываются: блоки по W баров, ускорение —
//!      вторая разность по блокам K_j − 2K_{j−1} + K_{j−2}. Отклик — доходность
//!      СЛЕДУЮЩЕГО блока. Наблюдения почти независимы, HAC работает честно.
//!   3) УСТОЙЧИВАЯ СПЕЦИФИКАЦИЯ. R_{j+1} = (α + β·Ã_j)·s_j² + ε, s_j² = W·σ²_j.
//!      Делим на s_j (а не на s_j²): y = R/s_j, регрессоры s_j и s_j·Ã_j, без
//!      свободного члена. Остатки примерно гомоскедастичны, 1/σ² не взрывается.
//!   4) ДЫРЯВЫЕ/ПЛОСКИЕ ОКНА НЕ ИМПУТИРУЮТСЯ. Окно с нулевой дисперсией,
//!      избытком нулевых доходностей или пересекающее границу торгового дня
//!      помечается невалидным; наблюдения, использующие такие блоки, выбрасываются
//!      (прежнее `unwrap_or(0.0)` создавало фиктивные скачки K).
//!   5) ЧЕСТНЫЙ ВЫВОД. Newey–West (Бартлетт) для ПРАВИЛЬНОЙ «сэндвич»-матрицы
//!      (x·e, а не e), плацебо с перестановкой блоков, профиль сдвигов,
//!      разбиение выборки пополам, контроль Δlog σ², робастный (винзоризованный)
//!      и «сырой» варианты.
//!
//! Модуль зависит только от std.

use std::f64::consts::SQRT_2;

// ====================================================================
// Конфигурация
// ====================================================================

#[derive(Debug, Clone)]
pub struct LeverageConfig {
    /// Длина блока W (баров). Горизонт отклика = W.
    pub window: usize,
    /// Минимум наблюдений в регрессии.
    pub min_obs: usize,
    /// Квантиль σ² (по тикеру), ниже которого наблюдение отбрасывается.
    pub var_trim_q: f64,
    /// Максимальная доля нулевых доходностей в блоке.
    pub max_zero_frac: f64,
    /// Двусторонняя винзоризация y и Ã (доля). 0 = выключена.
    pub winsor_p: f64,
    /// Число перестановок для плацебо.
    pub n_perm: usize,
    /// Длина блока перестановки (в блоках W).
    pub perm_block: usize,
    /// Зерно ГСЧ.
    pub seed: u64,
    /// Уточняющее число перестановок при малом p-value.
    pub n_perm_refine: usize,
    /// Порог perm_p, ниже которого запускается уточнение.
    pub refine_below: f64,
}

impl LeverageConfig {
    pub fn new(window: usize) -> Self {
        Self {
            window,
            min_obs: 60,
            var_trim_q: 0.05,
            max_zero_frac: 0.5,
            winsor_p: 0.01,
            n_perm: 199,
            perm_block: 5,
            seed: 0x9E37_79B9_7F4A_7C15,
            n_perm_refine: 4999,
            refine_below: 0.02,
        }
    }
}

// ====================================================================
// Блочная сетка
// ====================================================================

/// Данные на сетке непересекающихся блоков. Блок j = r[j·W .. (j+1)·W).
#[derive(Debug, Clone)]
pub struct BlockGrid {
    pub w: usize,
    /// Эксцесс блока (NaN, если блок невалиден).
    pub k: Vec<f64>,
    /// Выборочная дисперсия блока на бар (NaN, если невалиден).
    pub v: Vec<f64>,
    /// Сумма доходностей блока (NaN, если блок содержит «плохие» доходности).
    pub sum: Vec<f64>,
}

fn block_stats(x: &[f64]) -> Option<(f64, f64)> {
    let n = x.len() as f64;
    if x.len() < 5 {
        return None;
    }
    let mean = x.iter().sum::<f64>() / n;
    let (mut m2, mut m4) = (0.0, 0.0);
    for &r in x {
        let d = r - mean;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    let var_unbiased = m2 / (n - 1.0);
    m2 /= n;
    m4 /= n;
    if !(m2 > 0.0) || !m2.is_finite() || !m4.is_finite() {
        return None;
    }
    Some((m4 / (m2 * m2) - 3.0, var_unbiased))
}

/// `bad[i] == true` — доходность i пересекает границу дня/пропуск; блок с
/// такой доходностью считается невалидным (и как регрессор, и как отклик).
pub fn build_grid(returns: &[f64], cfg: &LeverageConfig, bad: Option<&[bool]>) -> BlockGrid {
    let w = cfg.window;
    let nb = returns.len() / w;
    let (mut k, mut v, mut sum) = (
        Vec::with_capacity(nb),
        Vec::with_capacity(nb),
        Vec::with_capacity(nb),
    );
    for j in 0..nb {
        let sl = &returns[j * w..(j + 1) * w];
        let has_bad = bad
            .map(|b| b[j * w..(j + 1) * w].iter().any(|&f| f))
            .unwrap_or(false);
        if has_bad || sl.iter().any(|r| !r.is_finite()) {
            k.push(f64::NAN);
            v.push(f64::NAN);
            sum.push(f64::NAN);
            continue;
        }
        sum.push(sl.iter().sum());
        let zeros = sl.iter().filter(|&&r| r == 0.0).count() as f64 / w as f64;
        match block_stats(sl) {
            Some((kk, vv)) if zeros <= cfg.max_zero_frac => {
                k.push(kk);
                v.push(vv);
            }
            _ => {
                k.push(f64::NAN);
                v.push(f64::NAN);
            }
        }
    }
    BlockGrid { w, k, v, sum }
}

// ====================================================================
// Вспомогательные функции
// ====================================================================

pub(crate) fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

pub(crate) fn winsorize(x: &mut [f64], p: f64) {
    if p <= 0.0 || x.len() < 20 {
        return;
    }
    let mut s: Vec<f64> = x.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (lo, hi) = (quantile(&s, p), quantile(&s, 1.0 - p));
    for v in x.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}

pub(crate) fn standardize(x: &mut [f64]) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)).sqrt();
    if sd > 0.0 && sd.is_finite() {
        for v in x.iter_mut() {
            *v = (*v - m) / sd;
        }
    } else {
        for v in x.iter_mut() {
            *v = 0.0;
        }
    }
}

/// splitmix64 — маленький детерминированный ГСЧ без зависимостей.
pub(crate) struct Rng(pub(crate) u64);
impl Rng {
    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ====================================================================
// OLS + HAC (Newey–West, Бартлетт) для малого числа регрессоров
// ====================================================================

#[derive(Debug, Clone)]
pub struct OlsHac {
    pub coef: Vec<f64>,
    pub se: Vec<f64>,
    pub t: Vec<f64>,
    pub n: usize,
    /// Нецентрированный R² (регрессия без свободного члена).
    pub r2: f64,
}

fn invert(k: usize, a: &[f64]) -> Option<Vec<f64>> {
    // Гаусс–Жордан с частичным выбором главного элемента.
    let mut m = vec![0.0; k * 2 * k];
    for i in 0..k {
        for j in 0..k {
            m[i * 2 * k + j] = a[i * k + j];
        }
        m[i * 2 * k + k + i] = 1.0;
    }
    for c in 0..k {
        let mut p = c;
        for r in c + 1..k {
            if m[r * 2 * k + c].abs() > m[p * 2 * k + c].abs() {
                p = r;
            }
        }
        if m[p * 2 * k + c].abs() < 1e-300 {
            return None;
        }
        if p != c {
            for j in 0..2 * k {
                m.swap(c * 2 * k + j, p * 2 * k + j);
            }
        }
        let d = m[c * 2 * k + c];
        for j in 0..2 * k {
            m[c * 2 * k + j] /= d;
        }
        for r in 0..k {
            if r != c {
                let f = m[r * 2 * k + c];
                if f != 0.0 {
                    for j in 0..2 * k {
                        m[r * 2 * k + j] -= f * m[c * 2 * k + j];
                    }
                }
            }
        }
    }
    let mut inv = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            inv[i * k + j] = m[i * 2 * k + k + j];
        }
    }
    Some(inv)
}

/// y = Xb + e, без свободного члена; `cols[c][i]` — i-е наблюдение c-го регрессора.
pub fn ols_hac(y: &[f64], cols: &[&[f64]], lag: usize) -> Option<OlsHac> {
    let n = y.len();
    let k = cols.len();
    if n < k + 5 || cols.iter().any(|c| c.len() != n) {
        return None;
    }
    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];
    for i in 0..n {
        for a in 0..k {
            xty[a] += cols[a][i] * y[i];
            for b in 0..k {
                xtx[a * k + b] += cols[a][i] * cols[b][i];
            }
        }
    }
    let inv = invert(k, &xtx)?;
    let mut coef = vec![0.0; k];
    for a in 0..k {
        for b in 0..k {
            coef[a] += inv[a * k + b] * xty[b];
        }
    }
    // u_t = x_t · e_t
    let mut e = vec![0.0; n];
    let (mut ss_res, mut ss_tot) = (0.0, 0.0);
    for i in 0..n {
        let mut f = 0.0;
        for a in 0..k {
            f += coef[a] * cols[a][i];
        }
        e[i] = y[i] - f;
        ss_res += e[i] * e[i];
        ss_tot += y[i] * y[i];
    }
    let u = |i: usize, a: usize| cols[a][i] * e[i];
    let lag = lag.min(n - 1);
    let mut s = vec![0.0; k * k];
    for l in 0..=lag {
        let wgt = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
        for a in 0..k {
            for b in 0..k {
                let mut g = 0.0;
                for t in l..n {
                    g += u(t, a) * u(t - l, b);
                }
                if l == 0 {
                    s[a * k + b] += g;
                } else {
                    // Γ_l + Γ_l'
                    s[a * k + b] += wgt * g;
                    s[b * k + a] += wgt * g;
                }
            }
        }
    }
    // V = c · inv · S · inv, c = n/(n−k)
    let c = n as f64 / (n - k) as f64;
    let mut tmp = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            for m in 0..k {
                tmp[a * k + b] += inv[a * k + m] * s[m * k + b];
            }
        }
    }
    let mut se = vec![0.0; k];
    for a in 0..k {
        let mut var = 0.0;
        for m in 0..k {
            var += tmp[a * k + m] * inv[m * k + a];
        }
        se[a] = (c * var).max(0.0).sqrt();
    }
    let t: Vec<f64> = (0..k)
        .map(|a| if se[a] > 0.0 { coef[a] / se[a] } else { f64::NAN })
        .collect();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { f64::NAN };
    Some(OlsHac { coef, se, t, n, r2 })
}

pub(crate) fn nw_lag(n: usize) -> usize {
    (((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor()) as usize).max(3)
}

// ====================================================================
// Строки регрессии для заданного сдвига регрессора
// ====================================================================

/// Одна строка наблюдения: всё известно в конце блока j, отклик — блок j+1.
#[derive(Debug, Clone)]
struct Rows {
    y: Vec<f64>,
    s: Vec<f64>,
    /// A_j — дискретное ускорение (вторая разность K).
    a: Vec<f64>,
    /// K_j — уровень эксцесса.
    a_level: Vec<f64>,
    /// ΔK_j = K_j − K_{j−1} — первая разность.
    a_delta: Vec<f64>,
    dv: Vec<f64>,
    r_fwd: Vec<f64>,
    jidx: Vec<usize>,
}

/// shift = 0 : Ã_j (информация на конец блока j) → R_{j+1}. ЭТО ОСНОВНОЙ ТЕСТ.
/// shift < 0 : более старое ускорение (Ã_{j+shift}).
/// shift = +1: Ã_{j+1} использует блок j+1, т.е. сам отклик. Это диагностика
///             механической связи, а не тест.
fn build_rows(g: &BlockGrid, cfg: &LeverageConfig, shift: i32, winsor_p: f64) -> Option<Rows> {
    let nb = g.k.len();
    if nb < 6 {
        return None;
    }
    // Порог по σ² — по валидным блокам тикера.
    let mut vv: Vec<f64> = g.v.iter().copied().filter(|x| x.is_finite()).collect();
    if vv.len() < cfg.min_obs {
        return None;
    }
    vv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = quantile(&vv, cfg.var_trim_q);

    let acc = |j: isize| -> f64 {
        if j < 2 || j as usize >= nb {
            return f64::NAN;
        }
        let j = j as usize;
        g.k[j] - 2.0 * g.k[j - 1] + g.k[j - 2] // NaN распространяется
    };

    // Уровень, первая разность, вторая разность — все на блоке j+shift.
    let level = |j: isize| -> f64 {
        if j < 0 || j as usize >= nb { return f64::NAN; }
        g.k[j as usize]
    };
    let delta = |j: isize| -> f64 {
        if j < 1 || j as usize >= nb { return f64::NAN; }
        g.k[j as usize] - g.k[j as usize - 1]
    };

    let (mut y, mut s, mut a, mut a_level, mut a_delta, mut dv, mut rf, mut jx) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(),
         Vec::new(), Vec::new(), Vec::new());
    for j in 2..nb.saturating_sub(1) {
        let (vj, vjm1) = (g.v[j], g.v[j - 1]);
        let r_next = g.sum[j + 1];
        let jr = j as isize + shift as isize;
        let aj = acc(jr);
        let al = level(jr);
        let ad = delta(jr);
        if !(vj.is_finite() && vj >= thr && vj > 0.0 && vjm1.is_finite() && r_next.is_finite()
            && aj.is_finite() && al.is_finite() && ad.is_finite())
        {
            continue;
        }
        let sj = (g.w as f64 * vj).sqrt();
        y.push(r_next / sj);
        s.push(sj);
        a.push(aj);
        a_level.push(al);
        a_delta.push(ad);
        dv.push(vj.ln() - vjm1.ln());
        rf.push(r_next);
        jx.push(j);
    }
    if y.len() < cfg.min_obs {
        return None;
    }
    standardize(&mut a);
    standardize(&mut a_level);
    standardize(&mut a_delta);
    standardize(&mut dv);
    if winsor_p > 0.0 {
        winsorize(&mut y, winsor_p);
        winsorize(&mut a, winsor_p);
        winsorize(&mut a_level, winsor_p);
        winsorize(&mut a_delta, winsor_p);
        winsorize(&mut dv, winsor_p);
    }
    Some(Rows { y, s, a, a_level, a_delta, dv, r_fwd: rf, jidx: jx })
}

fn fit_main(r: &Rows) -> Option<OlsHac> {
    let c1: Vec<f64> = r.s.iter().zip(&r.a).map(|(s, a)| s * a).collect();
    ols_hac(&r.y, &[&r.s, &c1], nw_lag(r.y.len()))
}

fn fit_k_level(r: &Rows) -> Option<OlsHac> {
    let c1: Vec<f64> = r.s.iter().zip(&r.a_level).map(|(s, a)| s * a).collect();
    ols_hac(&r.y, &[&r.s, &c1], nw_lag(r.y.len()))
}

fn fit_k_delta(r: &Rows) -> Option<OlsHac> {
    let c1: Vec<f64> = r.s.iter().zip(&r.a_delta).map(|(s, a)| s * a).collect();
    ols_hac(&r.y, &[&r.s, &c1], nw_lag(r.y.len()))
}

fn fit_ctrl(r: &Rows) -> Option<OlsHac> {
    let c1: Vec<f64> = r.s.iter().zip(&r.a).map(|(s, a)| s * a).collect();
    let c2: Vec<f64> = r.s.iter().zip(&r.dv).map(|(s, d)| s * d).collect();
    ols_hac(&r.y, &[&r.s, &c1, &c2], nw_lag(r.y.len()))
}

fn slice_rows(r: &Rows, lo: usize, hi: usize) -> Rows {
    Rows {
        y: r.y[lo..hi].to_vec(),
        s: r.s[lo..hi].to_vec(),
        a: r.a[lo..hi].to_vec(),
        a_level: r.a_level[lo..hi].to_vec(),
        a_delta: r.a_delta[lo..hi].to_vec(),
        dv: r.dv[lo..hi].to_vec(),
        r_fwd: r.r_fwd[lo..hi].to_vec(),
        jidx: r.jidx[lo..hi].to_vec(),
    }
}

/// Плацебо: переставляем Ã блоками длины b; y и σ не трогаем.
/// Односторонний p-value для t_β: P(t_perm ≥ t_obs).
fn permutation_p_for(predictor: &[f64], s: &[f64], y: &[f64], t_obs: f64, cfg: &LeverageConfig) -> f64 {
    let n = predictor.len().min(s.len()).min(y.len());
    if n < 20 { return f64::NAN; }
    let b = cfg.perm_block.max(1);
    let nblk = (n + b - 1) / b;
    let mut rng = Rng(cfg.seed ^ (n as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
    let mut order: Vec<usize> = (0..nblk).collect();
    let mut pred_perm = vec![0.0_f64; n];
    let mut c1 = vec![0.0_f64; n];
    let lag = nw_lag(n);
    let mut ge = 0usize;
    let mut done = 0usize;
    for _ in 0..cfg.n_perm {
        for i in (1..nblk).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut p = 0;
        for &bk in &order {
            for i in bk * b..((bk + 1) * b).min(n) {
                pred_perm[p] = predictor[i];
                p += 1;
            }
        }
        for i in 0..n {
            c1[i] = s[i] * pred_perm[i];
        }
        if let Some(f) = ols_hac(y, &[s, &c1], lag) {
            if f.t[1].is_finite() {
                done += 1;
                if f.t[1] >= t_obs { ge += 1; }
            }
        }
    }
    if done == 0 { f64::NAN } else { (1 + ge) as f64 / (1 + done) as f64 }
}

// ====================================================================
// Результаты и основной прогон
// ====================================================================

#[derive(Debug, Clone)]
pub struct LeverageResult {
    pub n_obs: usize,

    // H4c — ускорение (основной тест)
    pub alpha: f64,
    pub t_alpha: f64,
    pub beta: f64,
    pub se_beta: f64,
    pub t_beta: f64,
    pub r2: f64,
    pub perm_p: f64,
    pub t_raw: f64,
    pub beta_ctrl: f64,
    pub t_ctrl: f64,
    pub beta_first: f64,
    pub t_first: f64,
    pub beta_second: f64,
    pub t_second: f64,
    pub profile: Vec<(i32, f64)>,
    pub sdf_mean: f64,
    pub sdf_std: f64,
    pub scatter: Vec<(f64, f64)>,
    pub p_asym: f64,

    // H4a — уровень K_j
    pub beta_k_level: f64,
    pub t_k_level: f64,
    pub perm_p_k_level: f64,

    // H4b — первая разность ΔK_j
    pub beta_dk: f64,
    pub t_dk: f64,
    pub perm_p_dk: f64,
}

#[derive(Debug, Clone)]
pub struct Level4KurtosisLeverage {
    pub ticker: String,
    pub result: Option<LeverageResult>,
    /// H4c (ускорение): t_β > 1.645 ∧ perm_p ≤ 0.05 ∧ β>0 в обеих половинах.
    pub is_leverage: bool,
    /// H4a (уровень).
    pub is_leverage_k_level: bool,
    /// H4b (первая разность).
    pub is_leverage_dk: bool,
}

pub fn norm_sf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327 * (-0.5 * x * x).exp();
    let p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
        + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 { p } else { 1.0 - p }
}

pub fn run_leverage(
    returns: &[f64],
    cfg: &LeverageConfig,
    bad: Option<&[bool]>,
    with_scatter: bool,
) -> Option<LeverageResult> {
    let g = build_grid(returns, cfg, bad);
    let rows = build_rows(&g, cfg, 0, cfg.winsor_p)?;
    let main = fit_main(&rows)?;
    let (alpha, beta) = (main.coef[0], main.coef[1]);

    let mut perm_p = permutation_p_for(&rows.a, &rows.s, &rows.y, main.t[1], cfg);
    if cfg.n_perm > 0 && perm_p.is_finite() && perm_p <= cfg.refine_below
        && cfg.n_perm_refine > cfg.n_perm
    {
        let mut c2 = cfg.clone();
        c2.n_perm = cfg.n_perm_refine;
        perm_p = permutation_p_for(&rows.a, &rows.s, &rows.y, main.t[1], &c2);
    }

    // H4a — уровень K_j.
    let (beta_k_level, t_k_level, perm_p_k_level) = match fit_k_level(&rows) {
        Some(f) => {
            let mut pp = permutation_p_for(&rows.a_level, &rows.s, &rows.y, f.t[1], cfg);
            if cfg.n_perm > 0 && pp.is_finite() && pp <= cfg.refine_below
                && cfg.n_perm_refine > cfg.n_perm
            {
                let mut c2 = cfg.clone();
                c2.n_perm = cfg.n_perm_refine;
                pp = permutation_p_for(&rows.a_level, &rows.s, &rows.y, f.t[1], &c2);
            }
            (f.coef[1], f.t[1], pp)
        }
        None => (f64::NAN, f64::NAN, f64::NAN),
    };

    // H4b — первая разность ΔK_j.
    let (beta_dk, t_dk, perm_p_dk) = match fit_k_delta(&rows) {
        Some(f) => {
            let mut pp = permutation_p_for(&rows.a_delta, &rows.s, &rows.y, f.t[1], cfg);
            if cfg.n_perm > 0 && pp.is_finite() && pp <= cfg.refine_below
                && cfg.n_perm_refine > cfg.n_perm
            {
                let mut c2 = cfg.clone();
                c2.n_perm = cfg.n_perm_refine;
                pp = permutation_p_for(&rows.a_delta, &rows.s, &rows.y, f.t[1], &c2);
            }
            (f.coef[1], f.t[1], pp)
        }
        None => (f64::NAN, f64::NAN, f64::NAN),
    };

    let t_raw = build_rows(&g, cfg, 0, 0.0)
        .and_then(|r| fit_main(&r))
        .map(|f| f.t[1])
        .unwrap_or(f64::NAN);

    let (beta_ctrl, t_ctrl) = fit_ctrl(&rows)
        .map(|f| (f.coef[1], f.t[1]))
        .unwrap_or((f64::NAN, f64::NAN));

    let half = rows.y.len() / 2;
    let f1 = fit_main(&slice_rows(&rows, 0, half));
    let f2 = fit_main(&slice_rows(&rows, half, rows.y.len()));
    let (beta_first, t_first) = f1.map(|f| (f.coef[1], f.t[1])).unwrap_or((f64::NAN, f64::NAN));
    let (beta_second, t_second) = f2.map(|f| (f.coef[1], f.t[1])).unwrap_or((f64::NAN, f64::NAN));

    let mut profile = Vec::new();
    for sh in -2..=1 {
        let t = if sh == 0 {
            main.t[1]
        } else {
            build_rows(&g, cfg, sh, cfg.winsor_p)
                .and_then(|r| fit_main(&r))
                .map(|f| f.t[1])
                .unwrap_or(f64::NAN)
        };
        profile.push((sh, t));
    }

    // SDF: M = 1 − (α + β·Ã)·R_{j+1}
    let m: Vec<f64> = (0..rows.y.len())
        .map(|i| 1.0 - (alpha + beta * rows.a[i]) * rows.r_fwd[i])
        .collect();
    let mm = m.iter().sum::<f64>() / m.len() as f64;
    let sd = (m.iter().map(|x| (x - mm).powi(2)).sum::<f64>() / (m.len() - 1).max(1) as f64).sqrt();

    // Частичный скаттер (FWL): убираем s из y и из s·Ã.
    let scatter = if with_scatter {
        let sxx: f64 = rows.s.iter().map(|s| s * s).sum();
        let c1: Vec<f64> = rows.s.iter().zip(&rows.a).map(|(s, a)| s * a).collect();
        let by = rows.s.iter().zip(&rows.y).map(|(s, y)| s * y).sum::<f64>() / sxx;
        let bx = rows.s.iter().zip(&c1).map(|(s, c)| s * c).sum::<f64>() / sxx;
        let mut pts: Vec<(f64, f64)> = (0..rows.y.len())
            .map(|i| (c1[i] - bx * rows.s[i], rows.y[i] - by * rows.s[i]))
            .collect();
        // Для картинки: обрезаем по 1/99 перцентилям, чтобы масштаб был рабочим.
        let (mut xs, mut ys): (Vec<f64>, Vec<f64>) = pts.iter().cloned().unzip();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (xl, xh, yl, yh) = (
            quantile(&xs, 0.01),
            quantile(&xs, 0.99),
            quantile(&ys, 0.01),
            quantile(&ys, 0.99),
        );
        pts.retain(|&(x, y)| x >= xl && x <= xh && y >= yl && y <= yh);
        pts
    } else {
        Vec::new()
    };

    Some(LeverageResult {
        n_obs: main.n,
        alpha,
        t_alpha: main.t[0],
        beta,
        se_beta: main.se[1],
        t_beta: main.t[1],
        r2: main.r2,
        perm_p,
        t_raw,
        beta_ctrl,
        t_ctrl,
        beta_first,
        t_first,
        beta_second,
        t_second,
        profile,
        sdf_mean: mm,
        sdf_std: sd,
        scatter,
        p_asym: norm_sf(main.t[1]),
        beta_k_level,
        t_k_level,
        perm_p_k_level,
        beta_dk,
        t_dk,
        perm_p_dk,
    })
}

/// Прогон для одного тикера. `day_len` — длина торгового дня в барах (390);
/// доходности, пересекающие границу дня, помечаются «плохими».
pub fn run_level4_leverage(
    ticker: &str,
    log_returns: &[f64],
    window: usize,
    day_len: Option<usize>,
    with_scatter: bool,
) -> Level4KurtosisLeverage {
    let cfg = LeverageConfig::new(window);
    let bad: Option<Vec<bool>> = day_len.map(|d| {
        // Доходность i — между барами i и i+1; бар i последний в дне, если (i+1) % d == 0.
        (0..log_returns.len()).map(|i| (i + 1) % d == 0).collect()
    });
    let result = run_leverage(log_returns, &cfg, bad.as_deref(), with_scatter);
    let is_leverage = result
        .as_ref()
        .map(|r| {
            r.t_beta > 1.645
                && r.perm_p <= 0.05
                && r.beta_first.is_finite()
                && r.beta_second.is_finite()
                && r.beta_first > 0.0
                && r.beta_second > 0.0
        })
        .unwrap_or(false);
    let is_leverage_k_level = result
        .as_ref()
        .map(|r| r.t_k_level > 1.645 && r.perm_p_k_level <= 0.05)
        .unwrap_or(false);
    let is_leverage_dk = result
        .as_ref()
        .map(|r| r.t_dk > 1.645 && r.perm_p_dk <= 0.05)
        .unwrap_or(false);
    Level4KurtosisLeverage {
        ticker: ticker.to_string(),
        result,
        is_leverage,
        is_leverage_k_level,
        is_leverage_dk,
    }
}

/// Чувствительность к длине блока: t_β (без плацебо) для набора W.
pub fn sensitivity_t(returns: &[f64], windows: &[usize], day_len: Option<usize>) -> Vec<(usize, f64)> {
    windows
        .iter()
        .map(|&w| {
            let mut cfg = LeverageConfig::new(w);
            cfg.n_perm = 0;
            let bad: Option<Vec<bool>> =
                day_len.map(|d| (0..returns.len()).map(|i| (i + 1) % d == 0).collect());
            let t = run_leverage(returns, &cfg, bad.as_deref(), false)
                .map(|r| r.t_beta)
                .unwrap_or(f64::NAN);
            (w, t)
        })
        .collect()
}

/// Бенджамини–Хохберг: число отвергнутых гипотез при уровне q.
pub fn benjamini_hochberg(pvals: &[f64], q: f64) -> usize {
    let mut p: Vec<f64> = pvals.iter().copied().filter(|x| x.is_finite()).collect();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = p.len();
    let mut kmax = 0;
    for (i, &pv) in p.iter().enumerate() {
        if pv <= q * (i + 1) as f64 / m as f64 {
            kmax = i + 1;
        }
    }
    kmax
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum L4Status { Ok, TooFewBlocks, TooFewValidBlocks, TooFewRows, FitFailed }

pub fn l4_status(returns: &[f64], cfg: &LeverageConfig, bad: Option<&[bool]>) -> L4Status {
    let g = build_grid(returns, cfg, bad);
    if g.k.len() < 6 { return L4Status::TooFewBlocks; }
    if g.v.iter().filter(|x| x.is_finite()).count() < cfg.min_obs {
        return L4Status::TooFewValidBlocks;
    }
    let rows = match build_rows(&g, cfg, 0, cfg.winsor_p) {
        Some(r) => r,
        None => return L4Status::TooFewRows,
    };
    if fit_main(&rows).is_none() { return L4Status::FitFailed; }
    L4Status::Ok
}

pub fn inject_effect(
    returns: &[f64], cfg: &LeverageConfig, bad: Option<&[bool]>, beta: f64,
) -> Vec<f64> {
    let g = build_grid(returns, cfg, bad);
    let nb = g.k.len();
    let mut acc = vec![f64::NAN; nb];
    for j in 2..nb { acc[j] = g.k[j] - 2.0 * g.k[j - 1] + g.k[j - 2]; }
    let fin: Vec<f64> = acc.iter().copied().filter(|x| x.is_finite()).collect();
    if fin.len() < 10 { return returns.to_vec(); }
    let m = fin.iter().sum::<f64>() / fin.len() as f64;
    let sd = (fin.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (fin.len() - 1) as f64).sqrt();
    if !(sd > 0.0) { return returns.to_vec(); }
    let mut out = returns.to_vec();
    for j in 2..nb.saturating_sub(1) {
        if !(acc[j].is_finite() && g.v[j].is_finite()) { continue; }
        let a = (acc[j] - m) / sd;
        let shift = beta * a * g.v[j];
        for x in out[(j + 1) * g.w..(j + 2) * g.w].iter_mut() { *x += shift; }
    }
    out
}

pub fn power_curve(
    returns: &[f64], window: usize, day_len: Option<usize>, betas: &[f64],
) -> Vec<(f64, bool)> {
    let cfg = LeverageConfig::new(window);
    let bad: Option<Vec<bool>> =
        day_len.map(|d| (0..returns.len()).map(|i| (i + 1) % d == 0).collect());
    betas.iter().map(|&b| {
        let r = inject_effect(returns, &cfg, bad.as_deref(), b);
        let hit = run_leverage(&r, &cfg, bad.as_deref(), false)
            .map(|x| x.t_beta > 1.645 && x.perm_p <= 0.05)
            .unwrap_or(false);
        (b, hit)
    }).collect()
}

#[derive(Debug, Clone)]
pub struct PanelObs { pub ticker: u32, pub t_key: i64, pub y: f64, pub s: f64, pub c1: f64 }

pub fn export_panel_rows(
    ticker_id: u32, returns: &[f64], cfg: &LeverageConfig,
    bad: Option<&[bool]>, block_offset: i64,
) -> Vec<PanelObs> {
    let g = build_grid(returns, cfg, bad);
    match build_rows(&g, cfg, 0, cfg.winsor_p) {
        Some(r) => (0..r.y.len()).map(|i| PanelObs {
            ticker: ticker_id,
            t_key: block_offset + r.jidx[i] as i64,
            y: r.y[i], s: r.s[i], c1: r.s[i] * r.a[i],
        }).collect(),
        None => Vec::new(),
    }
}

#[derive(Debug, Clone)]
pub struct PanelFit {
    pub alpha: f64, pub beta: f64, pub se_beta: f64, pub t_beta: f64,
    pub n_obs: usize, pub n_tickers: usize, pub n_periods: usize,
}

pub fn panel_driscoll_kraay(obs: &[PanelObs]) -> Option<PanelFit> {
    use std::collections::{BTreeMap, HashSet};
    if obs.len() < 200 { return None; }
    let (mut xx, mut xy) = ([0.0f64; 4], [0.0f64; 2]);
    for o in obs {
        let x = [o.s, o.c1];
        for a in 0..2 {
            xy[a] += x[a] * o.y;
            for b in 0..2 { xx[a * 2 + b] += x[a] * x[b]; }
        }
    }
    let inv = invert(2, &xx)?;
    let coef = [
        inv[0] * xy[0] + inv[1] * xy[1],
        inv[2] * xy[0] + inv[3] * xy[1],
    ];
    let mut by_t: BTreeMap<i64, [f64; 2]> = BTreeMap::new();
    for o in obs {
        let e = o.y - coef[0] * o.s - coef[1] * o.c1;
        let sc = by_t.entry(o.t_key).or_insert([0.0; 2]);
        sc[0] += o.s * e;
        sc[1] += o.c1 * e;
    }
    let tmin = *by_t.keys().next()?;
    let tmax = *by_t.keys().last()?;
    let len = (tmax - tmin + 1) as usize;
    let mut h = vec![[0.0f64; 2]; len];
    for (k, v) in &by_t { h[(*k - tmin) as usize] = *v; }
    let lag = nw_lag(len).min(len - 1);
    let mut s = [0.0f64; 4];
    for l in 0..=lag {
        let w = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
        for t in l..len {
            for a in 0..2 {
                for b in 0..2 {
                    let g = h[t][a] * h[t - l][b];
                    if l == 0 { s[a * 2 + b] += g; }
                    else { s[a * 2 + b] += w * g; s[b * 2 + a] += w * g; }
                }
            }
        }
    }
    let c = len as f64 / (len as f64 - 1.0);
    let mut var = 0.0;
    for m in 0..2 { for n in 0..2 { var += inv[2 + m] * s[m * 2 + n] * inv[n * 2 + 1]; } }
    let se = (c * var).max(0.0).sqrt();
    Some(PanelFit {
        alpha: coef[0], beta: coef[1], se_beta: se,
        t_beta: if se > 0.0 { coef[1] / se } else { f64::NAN },
        n_obs: obs.len(),
        n_tickers: obs.iter().map(|o| o.ticker).collect::<HashSet<_>>().len(),
        n_periods: len,
    })
}

#[allow(dead_code)]
const _UNUSED: f64 = SQRT_2;

// ====================================================================
// Публичные утилиты для H*(β) в analysis_core.
// ====================================================================
//
// Эти функции используются уровнем VII для построения ряда
// ускорения эксцесса, который подаётся в measure_h_star как θ = β.
// Внутри самого leverage.rs ускорение считается по блокам, но
// наружу нужны именно эти две тривиальные функции.

/// Вторая разность: Ä_K(t) = K(t) − 2·K(t−1) + K(t−2).
/// Возвращает ряд длины n−2 для входа длины n.
pub fn kurtosis_acceleration(kurt_series: &[f64]) -> Vec<f64> {
    let n = kurt_series.len();
    if n < 3 { return Vec::new(); }
    let mut out = Vec::with_capacity(n - 2);
    for t in 2..n {
        let a = kurt_series[t] - 2.0 * kurt_series[t - 1] + kurt_series[t - 2];
        out.push(a);
    }
    out
}

/// Demeaned ускорение: Ã_K(t) = Ä_K(t) − E[Ä_K].
pub fn demeaned_acceleration(acc: &[f64]) -> Vec<f64> {
    if acc.is_empty() { return Vec::new(); }
    let mean = acc.iter().sum::<f64>() / acc.len() as f64;
    acc.iter().map(|&v| v - mean).collect()
}


// ====================================================================
// Тесты
// ====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_normal(rng: &mut Rng) -> f64 {
        // Бокс–Мюллер
        let u1 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 + 1e-18;
        let u2 = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn sim_returns(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng(seed);
        (0..n).map(|_| 1e-3 * lcg_normal(&mut rng)).collect()
    }

    /// Изменение r[t+1..] не должно менять регрессоры до t; меняет только отклик.
    #[test]
    fn no_lookahead_in_regressors() {
        let mut cfg = LeverageConfig::new(60);
        cfg.min_obs = 10;
        let r = sim_returns(60 * 40, 1);
        let g1 = build_grid(&r, &cfg, None);
        // Портим ПОСЛЕДНИЙ блок (j = 39).
        let mut r2 = r.clone();
        for x in r2[60 * 39..].iter_mut() {
            *x *= 50.0;
        }
        let g2 = build_grid(&r2, &cfg, None);
        for j in 0..39 {
            assert_eq!(g1.k[j], g2.k[j]);
            assert_eq!(g1.v[j], g2.v[j]);
        }
        // Строки: x_j для j ≤ 37 совпадают, только y_38 (блок 39) меняется.
        let a = build_rows(&g1, &cfg, 0, 0.0).unwrap();
        let b = build_rows(&g2, &cfg, 0, 0.0).unwrap();
        // Сравниваем сырые (до стандартизации это не работает) — проверяем через s и r_fwd.
        let n = a.s.len().min(b.s.len());
        for i in 0..n - 1 {
            assert_eq!(a.s[i], b.s[i]);
            assert_eq!(a.r_fwd[i], b.r_fwd[i]);
        }
    }

    /// Плоские окна не импутируются, а выбрасываются.
    #[test]
    fn flat_windows_are_invalid() {
        let cfg = LeverageConfig::new(60);
        let mut r = sim_returns(60 * 10, 2);
        for x in r[120..180].iter_mut() {
            *x = 0.0;
        }
        let g = build_grid(&r, &cfg, None);
        assert!(g.k[2].is_nan() && g.v[2].is_nan());
        assert!(g.k[1].is_finite());
    }

    /// Блок с границей дня невалиден.
    #[test]
    fn day_boundary_invalidates_block() {
        let cfg = LeverageConfig::new(60);
        let r = sim_returns(60 * 10, 3);
        let bad: Vec<bool> = (0..r.len()).map(|i| (i + 1) % 150 == 0).collect();
        let g = build_grid(&r, &cfg, Some(&bad));
        // Возврат 149 лежит в блоке 2 (120..180).
        assert!(g.k[2].is_nan());
        assert!(g.k[1].is_finite());
    }

    #[test]
    fn ols_hac_recovers_known_coefficients() {
        let mut rng = Rng(7);
        let n = 2000;
        let x1: Vec<f64> = (0..n).map(|_| lcg_normal(&mut rng)).collect();
        let x2: Vec<f64> = (0..n).map(|_| lcg_normal(&mut rng)).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 * x1[i] + 2.0 * x2[i] + 0.1 * lcg_normal(&mut rng))
            .collect();
        let f = ols_hac(&y, &[&x1, &x2], 5).unwrap();
        assert!((f.coef[0] - 1.0).abs() < 0.02);
        assert!((f.coef[1] - 2.0).abs() < 0.02);
    }

    #[test]
    fn bh_basic() {
        let p = [0.001, 0.01, 0.02, 0.5, 0.9];
        assert_eq!(benjamini_hochberg(&p, 0.05), 3);
    }
}