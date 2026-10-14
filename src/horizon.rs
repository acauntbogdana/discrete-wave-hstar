//! Уровень VII. Горизонт статистической пригодности H*(θ).
//!
//! Реализует:
//!   1) коинтегрирующую VAR(2) и VECM(2) (Stephenson et al., 2023, ур. 7–8);
//!   2) Johansen trace test для определения ранга Π;
//!   3) масштабирующий фактор β_coint = −β_2 / β_1;
//!   4) измерение H*(θ) и проверку иерархии.

use nalgebra::{DMatrix, DVector};

// =====================================================================
// 1. Оценка VAR(2) (Stephenson et al., 2023, ур. 7)
// =====================================================================

/// Параметры VAR(2).
#[derive(Debug, Clone)]
pub struct Var2Params {
    pub mu: DVector<f64>,       // 2×1
    pub pi1: DMatrix<f64>,      // 2×2
    pub pi2: DMatrix<f64>,      // 2×2
    pub sigma: DMatrix<f64>,    // 2×2 (остаточная ковариация)
}

/// Оценка VAR(2) методом наименьших квадратов.
///
///   z_t = μ + Π₁ z_{t−1} + Π₂ z_{t−2} + ε_t
///
/// z_t — вектор (y_t, x_t) наблюдаемого и эталонного моментов.
pub fn fit_var2(y: &[f64], x: &[f64]) -> Option<Var2Params> {
    let n = y.len().min(x.len());
    if n < 10 { return None; }

    // Формируем регрессионную матрицу Z = [1, z_{t-1}, z_{t-2}] для t = 2..n
    let m = n - 2;
    let mut z = DMatrix::<f64>::zeros(m, 5); // [1, y1, x1, y2, x2]
    let mut y_target = DMatrix::<f64>::zeros(m, 2);

    for t in 2..n {
        let row = t - 2;
        z[(row, 0)] = 1.0;
        z[(row, 1)] = y[t - 1];
        z[(row, 2)] = x[t - 1];
        z[(row, 3)] = y[t - 2];
        z[(row, 4)] = x[t - 2];
        y_target[(row, 0)] = y[t];
        y_target[(row, 1)] = x[t];
    }

    // OLS: B = (Z'Z)^{-1} Z' Y
    let ztz = z.transpose() * &z;
    let zty = z.transpose() * &y_target;
    let b = ztz.lu().solve(&zty)?;

    // Извлекаем параметры
    let mu = DVector::from_vec(vec![b[(0, 0)], b[(0, 1)]]);
    let pi1 = DMatrix::from_row_slice(2, 2, &[
        b[(1, 0)], b[(2, 0)],
        b[(1, 1)], b[(2, 1)],
    ]);
    let pi2 = DMatrix::from_row_slice(2, 2, &[
        b[(3, 0)], b[(4, 0)],
        b[(3, 1)], b[(4, 1)],
    ]);

    // Остатки и ковариация
    let residuals = &y_target - &z * &b;
    let sigma = residuals.transpose() * &residuals / m as f64;

    Some(Var2Params { mu, pi1, pi2, sigma })
}

// =====================================================================
// 2. VECM(2) (Stephenson et al., 2023, ур. 8)
// =====================================================================

/// Параметры VECM(2).
#[derive(Debug, Clone)]
pub struct VecmParams {
    pub pi: DMatrix<f64>,       // Π = Π₁ + Π₂ − I
    pub gamma: DMatrix<f64>,    // Γ = −Π₂
    pub rank: usize,
    pub coint_vector: Option<DVector<f64>>,   // β
    pub beta_coint: f64,        // масштабирующий фактор
}

/// VECM-разложение: Π = Π₁ + Π₂ − I, Γ = −Π₂.
pub fn vecm_from_var2(var2: &Var2Params) -> VecmParams {
    let identity = DMatrix::<f64>::identity(2, 2);
    let pi = &var2.pi1 + &var2.pi2 - &identity;
    let gamma = -&var2.pi2;
    VecmParams {
        pi, gamma,
        rank: 0,
        coint_vector: None,
        beta_coint: f64::NAN,
    }
}

// =====================================================================
// 3. Ранг Π (Stephenson et al., 2023, стр. 5)
// =====================================================================

/// Определение ранга Π и коинтеграционного вектора.
///
/// r = 2 ⇒ Π полного ранга, z_t стационарен;
/// r = 0 ⇒ Π = 0, два независимых случайных блуждания;
/// r = 1 ⇒ есть коинтеграционная связь.
///
/// Использует сингулярное разложение Π.
pub fn detect_rank_and_cointegration(
    var2: &Var2Params,
) -> VecmParams {
    let mut vecm = vecm_from_var2(var2);

    // SVD матрицы Π
    let svd = var2_svd(&vecm.pi);
    let sigma = &svd.1;
    if sigma.len() < 2 { return vecm; }

    // Определение ранга по относительной величине сингулярных чисел
    let s1 = sigma[0];
    let s2 = sigma[1];
    let threshold = 1e-3 * s1.max(1e-12);
    let rank = if s2 < threshold { 1 } else { 2 };

    vecm.rank = rank;
    if rank == 1 {
        // β = правый сингулярный вектор, соответствующий меньшему σ
        // (или, эквивалентно, ядро Π по столбцам)
        let v = &svd.2;
        let beta = DVector::from_vec(vec![v[(0, 1)], v[(1, 1)]]);
        // Нормировка: β₁ = 1
        let b1 = beta[0];
        let beta_norm = if b1.abs() > 1e-12 { &beta / b1 } else { beta.clone() };
        vecm.beta_coint = -beta_norm[1] / beta_norm[0]; // −β₂/β₁
        vecm.coint_vector = Some(beta_norm);
    }
    vecm
}

/// SVD для 2×2 матрицы через nalgebra.
fn var2_svd(m: &DMatrix<f64>) -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>) {
    let svd = nalgebra::linalg::SVD::new(m.clone(), true, true);
    let u = svd.u.unwrap_or_else(|| DMatrix::identity(2, 2));
    let v_t = svd.v_t.unwrap_or_else(|| DMatrix::identity(2, 2));
    let s = svd.singular_values.iter().copied().collect::<Vec<_>>();
    (u, s, v_t.transpose())
}

// =====================================================================
// 4. Johansen trace test (Stephenson et al., 2023, стр. 5)
// =====================================================================

/// Результат Johansen trace test.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JohansenResult {
    pub trace_r0: f64,      // H0: r ≤ 0
    pub trace_r1: f64,      // H0: r ≤ 1
    pub reject_r0: bool,    // отвергаем r = 0 (есть хотя бы одна связь)
    pub reject_r1: bool,    // отвергаем r ≤ 1 (есть более одной связи)
    pub rank: usize,
}

/// Упрощённый Johansen trace test.
///
/// Полный тест требует решения обобщённой задачи на собственные значения
/// для матриц S₁₁, S₀₀, S₀₁. Здесь используем SVD Π как аппроксимацию.
/// Johansen trace test через обобщённую задачу на собственные значения.
///
/// |S01 · S11⁻¹ · S10 − λ · S00| = 0
///
/// где S_ij — матрицы моментов остатков регрессий Δz_t на z_{t-1}
/// и Δz_{t-1}. Trace-статистика для H0: r ≤ q:
///   trace_q = −T · Σ_{i>q} log(1 − λ̂_i)
pub fn johansen_trace(
    y: &[f64],
    x: &[f64],
    crit_r0: f64,
    crit_r1: f64,
) -> JohansenResult {
    let n = y.len().min(x.len());
    if n < 15 {
        return JohansenResult {
            trace_r0: f64::NAN, trace_r1: f64::NAN,
            reject_r0: false, reject_r1: false, rank: 0,
        };
    }

    // Регрессия Δz_t на z_{t-1} и Δz_{t-1}
    // Δz_t = (Δy_t, Δx_t), z_{t-1} = (y_{t-1}, x_{t-1}), Δz_{t-1} = (Δy_{t-1}, Δx_{t-1})
    let m = n - 2;
    if m < 10 {
        return JohansenResult {
            trace_r0: f64::NAN, trace_r1: f64::NAN,
            reject_r0: false, reject_r1: false, rank: 0,
        };
    }

    // Формируем матрицы регрессоров: Z0 = Δz_t, Z1 = z_{t-1}, Z2 = Δz_{t-1}
    let mut z0 = DMatrix::<f64>::zeros(m, 2);   // Δz_t
    let mut z1 = DMatrix::<f64>::zeros(m, 2);   // z_{t-1}
    let mut z2 = DMatrix::<f64>::zeros(m, 2);   // Δz_{t-1}

    for t in 2..n {
        let row = t - 2;
        z0[(row, 0)] = y[t] - y[t - 1];
        z0[(row, 1)] = x[t] - x[t - 1];
        z1[(row, 0)] = y[t - 1];
        z1[(row, 1)] = x[t - 1];
        z2[(row, 0)] = y[t - 1] - y[t - 2];
        z2[(row, 1)] = x[t - 1] - x[t - 2];
    }

    // Остатки R0 = Z0 − Z2·B0, R1 = Z1 − Z2·B1,
    // где B0 = (Z2'Z2)⁻¹·Z2'·Z0, B1 = (Z2'Z2)⁻¹·Z2'·Z1.
    //
    // ВАЖНО: расстановка скобок критична. Если написать
    //   &z2 * &z2tz2_inv * z2.transpose() * &z0
    // то из-за левоассоциативности сначала получится
    //   (m×2)·(2×m) = m×m
    // — матрица 122 ГБ при m ≈ 124k. Правильная группировка:
    let z2tz2 = z2.transpose() * &z2;
    let z2tz2_inv = match z2tz2.try_inverse() {
        Some(inv) => inv,
        None => return JohansenResult {
            trace_r0: f64::NAN, trace_r1: f64::NAN,
            reject_r0: false, reject_r1: false, rank: 0,
        },
    };
    let b0 = &z2tz2_inv * (z2.transpose() * &z0);   // 2×2
    let b1 = &z2tz2_inv * (z2.transpose() * &z1);   // 2×2
    let r0 = &z0 - &z2 * b0;                         // m×2
    let r1 = &z1 - &z2 * b1;                         // m×2

    // Моментные матрицы
    let s00 = r0.transpose() * &r0 / m as f64;
    let s01 = r0.transpose() * &r1 / m as f64;
    let s10 = s01.transpose();
    let s11 = r1.transpose() * &r1 / m as f64;

    let s11_inv = match s11.try_inverse() {
        Some(inv) => inv,
        None => return JohansenResult {
            trace_r0: f64::NAN, trace_r1: f64::NAN,
            reject_r0: false, reject_r1: false, rank: 0,
        },
    };

    let s00_inv = match s00.try_inverse() {
        Some(inv) => inv,
        None => return JohansenResult {
            trace_r0: f64::NAN, trace_r1: f64::NAN,
            reject_r0: false, reject_r1: false, rank: 0,
        },
    };

    // Обобщённая задача: |S01 S11⁻¹ S10 − λ S00| = 0
    // Эквивалентно: |S00⁻¹ S01 S11⁻¹ S10 − λ I| = 0
    let m_matrix = &s00_inv * &s01 * &s11_inv * &s10;

    // Собственные числа 2×2 через характеристическое уравнение
    let a = m_matrix[(0, 0)];
    let b = m_matrix[(0, 1)];
    let c = m_matrix[(1, 0)];
    let d = m_matrix[(1, 1)];
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    let (lambda1, lambda2) = if disc >= 0.0 {
        let sq = disc.sqrt();
        let l1 = (trace + sq) / 2.0;
        let l2 = (trace - sq) / 2.0;
        if l1 >= l2 { (l1, l2) } else { (l2, l1) }
    } else {
        // Комплексные — берём действительную часть как приближение
        let l = trace / 2.0;
        (l, l)
    };

    // Защита от выхода за [0, 1)
    let lambda1 = lambda1.clamp(0.0, 0.999);
    let lambda2 = lambda2.clamp(0.0, 0.999);

    let t_eff = m as f64;
    let trace_r0 = -t_eff * ((1.0 - lambda1).ln() + (1.0 - lambda2).ln());
    let trace_r1 = -t_eff * (1.0 - lambda2).ln();

    let reject_r0 = trace_r0 > crit_r0;
    let reject_r1 = trace_r1 > crit_r1;

    let rank = if reject_r1 { 2 } else if reject_r0 { 1 } else { 0 };

    JohansenResult {
        trace_r0, trace_r1,
        reject_r0, reject_r1,
        rank,
    }
}

// =====================================================================
// 5. Измерение H*(θ) через прямой тест стабильности момента
// =====================================================================
//
// Идея: для каждого масштаба W строим серию моментов θ̂_t(W) на
// скользящем окне длины W. H*(θ) — максимальное W, при котором
// θ̂_t(W) не имеет статистически значимого тренда за всю историю.
//
// Это прямая операционализация определения:
//   H*(θ, i, W, δ) = max{W : момент θ стабилен на масштабе W}
// из диссертации.
//
// Метод:
//   1) для W от min_window до max_window:
//        a) rolling_moments(theta, W) → серия θ̂_t(W)
//        b) OLS-наклон b и его стандартная ошибка se_b
//        c) если |b| < δ_relative · mean(|θ̂|)  И  |b| / se_b < z_crit
//           → момент стабилен на масштабе W
//   2) H*(θ) = максимальное такое W
//
// δ_relative — допуск из HSTAR_DELTA (5% от среднего уровня).
// z_crit = 1.96 (95% доверие).

/// Конфигурация измерения H*(θ) локальным методом.
#[derive(Debug, Clone, Copy)]
pub struct HStarConfig {
    pub min_window: usize,
    pub max_window: usize,
    pub local_mult: usize,
    pub drift_tol: f64,
    pub max_bad_fraction: f64,
    pub min_segments: usize,
    /// Число перемешиваний для нулевого распределения drift (0 = фиксированный порог).
    pub null_shuffles: usize,
    /// Квантиль нулевого drift, принимаемый за порог.
    pub null_quantile: f64,
    /// Длина блока перемешивания. 1 = iid; 390 = день.
    pub null_block: usize,
}

/// Результат измерения H*(θ) локальным методом.
///
/// Поля slope/se_slope/t_stat убраны: определение H* больше не
/// использует глобальный t-тест по всей истории. Вместо этого H*
/// измеряется как максимальный масштаб, на котором доля локальных
/// сегментов с существенным drift не превышает порог.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HStarResult {
    pub ticker: String,
    pub theta: String,
    pub h_star: usize,
    /// Доля локальных сегментов с drift > drift_tol на найденном W.
    pub bad_fraction: f64,
    /// Медиана локальных drift на найденном W.
    pub median_drift: f64,
    /// 90-й перцентиль локальных drift.
    pub p90_drift: f64,
    /// Медианный robust scale момента на найденном W.
    pub scale: f64,
    /// Средний уровень момента (по всему ряду).
    pub mean_level: f64,
    /// Число валидных локальных сегментов на найденном W.
    pub n_segments: usize,
}

/// Перемешивание блоками длины `block` (сохраняет короткую зависимость).
fn block_shuffled(x: &[f64], block: usize, seed: u64) -> Vec<f64> {
    let b = block.max(1);
    let nblk = (x.len() + b - 1) / b;
    let mut order: Vec<usize> = (0..nblk).collect();
    let mut st = seed;
    let mut next = || {
        st = st.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = st;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..nblk).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut out = Vec::with_capacity(x.len());
    for &bk in &order {
        out.extend_from_slice(&x[bk * b..((bk + 1) * b).min(x.len())]);
    }
    out
}

fn theta_of(series: &[f64], kind: &str, w: usize) -> Option<Vec<f64>> {
    match kind {
        "mean" => Some(rolling_mean(series, w)),
        "var"  => Some(rolling_variance(series, w)),
        "raw"  => Some(series.to_vec()),
        _      => None,
    }
}

/// Robust scale: 1.4826·MAD. Устойчив к выбросам, не зависит от
/// среднего уровня. Для β, центрированного около нуля, применение
/// mean(|θ|) было бы концептуально неверным.
fn robust_scale(x: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.len() < 3 { return f64::NAN; }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = sorted[sorted.len() / 2];
    let mut dev: Vec<f64> = sorted.iter().map(|v| (v - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = dev[dev.len() / 2];
    1.4826 * mad
}

/// OLS-наклон ряда y_t по t = 0..n-1, без свободного члена в возврате.
/// Используется внутри локального сегмента фиксированной длины.
fn ols_slope(y: &[f64]) -> f64 {
    let n = y.len();
    if n < 3 { return f64::NAN; }
    let nf = n as f64;
    let mx = (n - 1) as f64 / 2.0;
    let my = y.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let dx = i as f64 - mx;
        sxx += dx * dx;
        sxy += dx * (v - my);
    }
    if sxx < 1e-15 { return f64::NAN; }
    sxy / sxx
}

fn chunk_drifts(theta: &[f64], w: usize, cfg: &HStarConfig) -> Option<(Vec<f64>, Vec<f64>)> {
    let clean: Vec<f64> = theta.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.len() < 60 { return None; }
    let local_len = (cfg.local_mult * w).max(60);
    if clean.len() / local_len < cfg.min_segments { return None; }
    let (mut drifts, mut scales) = (Vec::new(), Vec::new());
    for chunk in clean.chunks(local_len) {
        if chunk.len() < local_len { continue; }
        let scale = robust_scale(chunk);
        if !scale.is_finite() || scale < 1e-15 { continue; }
        let slope = ols_slope(chunk);
        if !slope.is_finite() { continue; }
        drifts.push(slope.abs() * chunk.len() as f64 / scale);
        scales.push(scale);
    }
    if drifts.len() < cfg.min_segments { None } else { Some((drifts, scales)) }
}

fn measure_h_star_local(
    series: &[f64],
    null_sets: &[Vec<f64>],
    theta_kind: &str,
    w: usize,
    cfg: &HStarConfig,
) -> Option<(f64, f64, f64, f64, usize)> {
    let theta = theta_of(series, theta_kind, w)?;
    let (mut drifts, mut scales) = chunk_drifts(&theta, w, cfg)?;

    let tol = if cfg.null_shuffles > 0 && !null_sets.is_empty() {
        let mut pooled: Vec<f64> = Vec::new();
        for ns in null_sets {
            if let Some(th) = theta_of(ns, theta_kind, w) {
                if let Some((d, _)) = chunk_drifts(&th, w, cfg) { pooled.extend(d); }
            }
        }
        if pooled.len() < 5 { return None; }
        pooled.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((pooled.len() as f64 * cfg.null_quantile) as usize).min(pooled.len() - 1);
        pooled[idx]
    } else {
        cfg.drift_tol
    };

    let n_seg = drifts.len();
    let n_bad = drifts.iter().filter(|&&d| d > tol).count();
    let bad_fraction = n_bad as f64 / n_seg as f64;

    drifts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_drift = drifts[drifts.len() / 2];
    let p90_drift = drifts[((drifts.len() as f64 * 0.9) as usize).min(drifts.len() - 1)];
    scales.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_scale = scales[scales.len() / 2];
    Some((bad_fraction, median_drift, p90_drift, median_scale, n_seg))
}

/// Скользящее среднее длины W для ряда x.
/// Возвращает серию той же длины, что и x, с NaN для первых W−1 точек.
pub fn rolling_mean(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || window > n { return out; }

    let mut sum: f64 = x[..window].iter().sum();
    out[window - 1] = sum / window as f64;

    for t in window..n {
        sum += x[t] - x[t - window];
        out[t] = sum / window as f64;
    }
    out
}

/// Скользящая дисперсия длины W для ряда x (среднее по окну).
pub fn rolling_variance(x: &[f64], window: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if window < 2 || window > n { return out; }

    for t in (window - 1)..n {
        let slice = &x[t + 1 - window..=t];
        let m = slice.iter().sum::<f64>() / window as f64;
        let v = slice.iter().map(|&r| (r - m).powi(2)).sum::<f64>()
              / (window - 1) as f64;
        out[t] = v;
    }
    out
}

/// Измерение H*(θ) локальным методом.
///
/// Идея. При большом n глобальный t-тест наклона по всей истории
/// ловит любой микроскопический тренд. Он не отвечает на вопрос
/// «на каком масштабе момент локально стабилен». Заменяем:
///   1) история разбивается на непересекающиеся сегменты длины
///      L = max(60, local_mult·W);
///   2) в каждом сегменте считается OLS-наклон и robust scale (1.4826·MAD);
///   3) drift = |slope| · L / scale — безразмерная величина,
///      применимая и к μ, и к σ², и к β;
///   4) H* = max W, при котором доля сегментов с drift > drift_tol
///      не превышает max_bad_fraction.
///
/// `theta_kind` ∈ {"mean", "var", "raw"}.
pub fn measure_h_star(
    ticker: &str,
    theta_name: &str,
    theta_kind: &str,
    series: &[f64],
    cfg: HStarConfig,
) -> HStarResult {
    let empty = |h: usize| HStarResult {
        ticker: ticker.to_string(),
        theta: theta_name.to_string(),
        h_star: h,
        bad_fraction: f64::NAN,
        median_drift: f64::NAN,
        p90_drift: f64::NAN,
        scale: f64::NAN,
        mean_level: f64::NAN,
        n_segments: 0,
    };

    let n = series.len();
    if n < cfg.min_window + 10 { return empty(0); }

    let max_w = cfg.max_window.min(n / 2);
    if max_w < cfg.min_window { return empty(0); }

    // Средний уровень момента по всему ряду — считаем один раз.
    let finite_vals: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
    let global_mean_level = if finite_vals.is_empty() {
        f64::NAN
    } else {
        finite_vals.iter().sum::<f64>() / finite_vals.len() as f64
    };

    let mut best_h = 0usize;
    let mut best_bad_fraction = f64::NAN;
    let mut best_median_drift = f64::NAN;
    let mut best_p90_drift = f64::NAN;
    let mut best_scale = f64::NAN;
    let mut best_n_seg = 0usize;

    let block = if theta_kind == "raw" { cfg.null_block.max(60) } else { cfg.null_block };
    let null_sets: Vec<Vec<f64>> = (0..cfg.null_shuffles)
        .map(|i| block_shuffled(series, block, 0xC0FFEE ^ (i as u64 + 1)))
        .collect();

    let steps = 25usize;
    let mut ws: Vec<usize> = Vec::new();
    for i in 0..=steps {
        let w = (cfg.min_window as f64
            * (max_w as f64 / cfg.min_window as f64).powf(i as f64 / steps as f64)).round() as usize;
        if ws.last() != Some(&w) && w >= cfg.min_window && w <= max_w {
            ws.push(w);
        }
    }

    for &w in &ws {
        if let Some((bad_fraction, median_drift, p90_drift, scale, n_seg)) =
            measure_h_star_local(series, &null_sets, theta_kind, w, &cfg)
        {
            if bad_fraction <= cfg.max_bad_fraction {
                best_h = w;
                best_bad_fraction = bad_fraction;
                best_median_drift = median_drift;
                best_p90_drift = p90_drift;
                best_scale = scale;
                best_n_seg = n_seg;
            }
        }
    }

    HStarResult {
        ticker: ticker.to_string(),
        theta: theta_name.to_string(),
        h_star: best_h,
        bad_fraction: best_bad_fraction,
        median_drift: best_median_drift,
        p90_drift: best_p90_drift,
        scale: best_scale,
        mean_level: global_mean_level,
        n_segments: best_n_seg,
    }
}

/// Коинтеграционный метод измерения H*(θ) (Stephenson et al., 2023).
///
/// Для каждого W:
///   1. Строим θ̂_t(W) — скользящий момент;
///   2. Эталон θ̂^ref_t(W) — тот же момент на удвоенном окне 2W;
///   3. z_t = (θ̂_t, θ̂^ref_t);
///   4. fit_var2 + detect_rank_and_cointegration + johansen_trace;
///   5. Стабильность: rank = 1 (есть коинтеграция).
/// H*(θ) = max{W : rank(W) = 1}.
pub fn measure_h_star_coint(
    ticker: &str,
    theta_name: &str,
    theta_kind: &str,
    series: &[f64],
    min_window: usize,
    max_window: usize,
) -> HStarResult {
    use crate::config::{TRACE_R0_CRIT_5PCT, TRACE_R1_CRIT_5PCT};

    let empty = |h: usize| HStarResult {
        ticker: ticker.to_string(),
        theta: format!("{}_coint", theta_name),
        h_star: h,
        bad_fraction: f64::NAN,
        median_drift: f64::NAN,
        p90_drift: f64::NAN,
        scale: f64::NAN,
        mean_level: f64::NAN,
        n_segments: 0,
    };

    // Прореживание: VAR(2)+Johansen работают O(m) на итерацию,
    // но цикл по W идёт до 480 раз. Прореживание до 5000 точек
    // не влияет на асимптотику теста и ускоряет в 10–20 раз.
    const MAX_N: usize = 5_000;
    let series_owned: Vec<f64>;
    let series: &[f64] = if series.len() > MAX_N {
        let step = series.len() / MAX_N + 1;
        series_owned = series.iter().step_by(step).copied().collect();
        &series_owned
    } else {
        series
    };

    if theta_kind != "mean" && theta_kind != "var" && theta_kind != "raw" {
        return empty(0);
    }

    let n = series.len();
    if n < min_window + 10 {
        return empty(0);
    }
    let max_w = max_window.min(n / 3).max(min_window);

    let mut best_h = 0usize;
    let mut best_level = f64::NAN;

    for w in min_window..=max_w {
        // Оба ряда моментов строим от ОДНОГО исходного series.
        // Прежняя версия строила reference от отфильтрованного clean,
        // из-за чего индексы не совпадали по времени.
        let theta_w: Vec<f64> = match theta_kind {
            "mean" => rolling_mean(series, w),
            "var"  => rolling_variance(series, w),
            "raw"  => series.to_vec(),
            _      => continue,
        };
        let theta_2w: Vec<f64> = match theta_kind {
            "mean" => rolling_mean(series, 2 * w),
            "var"  => rolling_variance(series, 2 * w),
            "raw"  => series.to_vec(),
            _      => continue,
        };

        // Оба ряда индексированы с правым краем окна. Первые 2w−1
        // значений theta_2w = NaN, они отфильтруются на zip.
        let pairs: Vec<(f64, f64)> = theta_w
            .iter()
            .zip(theta_2w.iter())
            .filter(|(a, b)| a.is_finite() && b.is_finite())
            .map(|(&a, &b)| (a, b))
            .collect();
        if pairs.len() < 60 {
            continue;
        }

        let y: Vec<f64> = pairs.iter().map(|p| p.0).collect();
        let x: Vec<f64> = pairs.iter().map(|p| p.1).collect();

        let joh = johansen_trace(&y, &x, TRACE_R0_CRIT_5PCT, TRACE_R1_CRIT_5PCT);
        if joh.rank == 1 {
            best_h = w;
            best_level = y.iter().sum::<f64>() / y.len() as f64;
        }
    }

    HStarResult {
        ticker: ticker.to_string(),
        theta: format!("{}_coint", theta_name),
        h_star: best_h,
        // Поля локального метода к коинтеграционной проверке не применимы.
        bad_fraction: f64::NAN,
        median_drift: f64::NAN,
        p90_drift: f64::NAN,
        scale: f64::NAN,
        mean_level: best_level,
        n_segments: 0,
    }
}

/// Проверка иерархии H*(μ) ≥ H*(σ²).
///
/// β исключён: H*(β) измеряется в блоках W=60 баров, а H*(μ), H*(σ²)
/// — в барах. Сравнение несопоставимых шкал бессмысленно.
/// H*(Σ) := H*(σ²) в скалярном случае, поэтому третьего условия нет.
/// Равенство разрешено: при H*(μ)=H*(σ²)=440 (типичный случай)
/// иерархия должна подтверждаться, а не отвергаться.
pub fn verify_hierarchy(h_mu: usize, h_sigma2: usize) -> HierarchyVerdict {
    let mut violations: Vec<String> = Vec::new();
    if h_mu < h_sigma2 {
        violations.push("H*(μ) < H*(σ²)".to_string());
    }

    HierarchyVerdict {
        holds: violations.is_empty(),
        violations,
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HierarchyVerdict {
    pub holds: bool,
    pub violations: Vec<String>,
}