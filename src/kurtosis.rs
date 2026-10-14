//! Уровень I. Дискретность и агрегация эксцесса.
//!
//! Реализует:
//!   1) робастный эксцесс RK (Liu, Maynard & Tsiakas, 2026);
//!   2) Model Architecture Search для идентификации типа функции
//!      убывания эксцесса (Miyahara et al., 2024);
//!   3) KL-дивергенцию между стандартизированными распределениями.

use statrs::distribution::{ContinuousCDF, Normal};

use crate::equations::{empirical_kurtosis, estimate_beta};

// =====================================================================
// 1. Робастный эксцесс (Liu, Maynard & Tsiakas, 2026, ур. 5, 11, 12)
// =====================================================================

/// Квантили, используемые для QS и QK.
/// α = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875).

/// Константы c_1..c_6 из Liu et al. (2026), ур. (10), (12).
/// Значения — функции квантилей стандартного нормального распределения.
/// Приводятся как в оригинале (см. Online Appendix A.3–A.4 исходной статьи).
const C1: f64 = 3.0288;
const C2: f64 = 0.7460;
const C3: f64 = 1.4264;
const C4: f64 = 5.3364;
const C5: f64 = 1.2735;
const C6: f64 = 4.9527;

/// Квантильная функция стандартного нормального распределения.
fn z(p: f64) -> f64 {
    Normal::new(0.0, 1.0).unwrap().inverse_cdf(p)
}

/// Выборочный квантиль с линейной интерполяцией (тип 7, как в R).
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    if n == 1 { return sorted[0]; }
    let pos = p * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi { return sorted[lo]; }
    let w = pos - lo as f64;
    sorted[lo] + w * (sorted[hi] - sorted[lo])
}

/// Квантильная асимметрия Боули QS (Liu et al., 2026, ур. 3).
fn bowley_qs(sorted: &[f64]) -> f64 {
    let q25 = quantile(sorted, 0.25);
    let q50 = quantile(sorted, 0.50);
    let q75 = quantile(sorted, 0.75);
    let denom = q75 - q25;
    if denom.abs() < 1e-12 { return f64::NAN; }
    ((q75 - q50) - (q50 - q25)) / denom
}

/// Квантильный эксцесс Мура QK (Liu et al., 2026, ур. 5).
fn moors_qk(sorted: &[f64]) -> f64 {
    let q125 = quantile(sorted, 0.125);
    let q375 = quantile(sorted, 0.375);
    let q625 = quantile(sorted, 0.625);
    let q875 = quantile(sorted, 0.875);
    let q25  = quantile(sorted, 0.25);
    let q75  = quantile(sorted, 0.75);
    let denom = q75 - q25;
    if denom.abs() < 1e-12 { return f64::NAN; }
    ((q875 - q625) + (q375 - q125)) / denom
}

/// Сглаженная винзоризованная QK* (Liu et al., 2026, ур. 11).
/// G(x) = 1 / (1 + exp(b(x − a))), a = 1.1, b = 1000.
fn winsorized_qk(qk: f64) -> f64 {
    if !qk.is_finite() { return f64::NAN; }
    let a = 1.1_f64;
    let b = 1000.0_f64;
    let g = 1.0 / (1.0 + (b * (qk - a)).exp());
    1.1 * g + qk * (1.0 - g)
}

#[derive(Debug, Clone, Copy)]
pub struct RkDiag {
    pub qs: f64,
    pub qk_raw: f64,
    pub qk: f64,
    pub denom: f64,
    pub n: usize,
}

pub fn robust_kurtosis_diag(sorted: &[f64]) -> Option<RkDiag> {
    if sorted.len() < 20 { return None; }
    let qs = bowley_qs(sorted);
    let qk_raw = moors_qk(sorted);
    if !qs.is_finite() || !qk_raw.is_finite() { return None; }
    let qk = winsorized_qk(qk_raw);
    Some(RkDiag { qs, qk_raw, qk, denom: C5 * qk - C2, n: sorted.len() })
}

/// Робастный эксцесс RK (Liu et al., 2026, ур. 12).
///
/// RK(Q) = 3 + (c4·QK* − c1)/(c5·QK* − c2)
///           + [(c6·QK* − c3)/(c5·QK* − c2)] · 36·QS²/φ²_{0.75}
///
/// Требует: sorted_returns — отсортированный по возрастанию ряд.
pub fn robust_kurtosis(sorted_returns: &[f64]) -> Option<f64> {
    if sorted_returns.len() < 20 { return None; }

    let qs = bowley_qs(sorted_returns);
    let qk_raw = moors_qk(sorted_returns);
    if !qs.is_finite() || !qk_raw.is_finite() { return None; }

    let qk = winsorized_qk(qk_raw);
    let denom = C5 * qk - C2;
    if denom.abs() < 1e-12 { return None; }

    let phi_075 = z(0.75);
    let term2 = (C4 * qk - C1) / denom;
    let term3 = ((C6 * qk - C3) / denom) * (36.0 * qs * qs) / (phi_075 * phi_075);
    let rk = 3.0 + term2 + term3;

    if rk.is_finite() { Some(rk) } else { None }
}

// =====================================================================
// 2. Model Architecture Search (Miyahara et al., 2024, стр. 10–11)
// =====================================================================

/// Кандидатные модели для MAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasModel {
    Linear,
    Poly2,
    Poly5,
    Exponential,
    StretchedExponential,
    PowerLaw,
}

impl MasModel {
    /// Список кандидатных моделей для MAS.
    ///
    /// Ограничен двумя физически осмысленными формами убывания эксцесса:
    ///   - PowerLaw:     K(h) ~ C · h^(−λ)
    ///   - StretchedExp: K(h) ~ exp(−α · h^(−β) + γ)
    ///
    /// Exponential и Linear исключены как физически неверные: при агрегации
    /// независимых приращений эксцесс обязан убывать (ЦПТ), поэтому
    /// экспоненциальный рост K ~ a·exp(b·h) с b > 0 невозможен.
    /// Poly2 и Poly5 исключены как склонные к переобучению на 9 точках.
    pub fn all() -> &'static [MasModel] {
        &[
            MasModel::StretchedExponential,
            MasModel::PowerLaw,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            MasModel::Linear => "Linear",
            MasModel::Poly2 => "Poly2",
            MasModel::Poly5 => "Poly5",
            MasModel::Exponential => "Exponential",
            MasModel::StretchedExponential => "StretchedExp",
            MasModel::PowerLaw => "PowerLaw",
        }
    }

    /// Предсказание по x (горизонт h).
    /// Для большинства моделей x интерпретируется в лог-шкале, где это нужно.
    pub fn predict(&self, x: f64, coefs: &[f64]) -> f64 {
        match self {
            MasModel::Linear => coefs[0] + coefs[1] * x,
            MasModel::Poly2 => coefs[0] + coefs[1] * x + coefs[2] * x * x,
            MasModel::Poly5 => {
                let mut y = coefs[0];
                let mut xp = 1.0;
                for k in 1..=5 {
                    xp *= x;
                    y += coefs[k] * xp;
                }
                y
            }
            MasModel::Exponential => coefs[0] * (coefs[1] * x).exp(),
            MasModel::StretchedExponential => {
                // y = exp(α · x^{−β} + γ), где coefs = [α, β, γ]
                coefs[2] + coefs[0] * x.powf(-coefs[1])
            }
            MasModel::PowerLaw => {
                // y = log C − λ · log x, где coefs = [log C, λ]
                coefs[0] - coefs[1] * x.ln()
            }
        }
    }
}

/// Результат MAS для одного тикера.
#[derive(Debug, Clone)]
pub struct MasResult {
    pub best_model: MasModel,
    pub best_mse: f64,
    pub mse_by_model: Vec<(MasModel, f64)>,
    /// Параметры лучшей модели (интерпретация зависит от типа).
    pub best_params: Vec<f64>,
}

/// Подгонка одной из кандидатных моделей методом наименьших квадратов.
/// Возвращает параметры. Для простоты используется линейная регрессия
/// в подходящем преобразовании или прямой градиентный спуск.
fn fit_model(
    model: MasModel,
    xs: &[f64],
    ys: &[f64],
) -> Option<Vec<f64>> {
    match model {
        MasModel::Linear | MasModel::Poly2 | MasModel::Poly5 => {
            fit_polynomial_least_squares(model, xs, ys)
        }
        MasModel::Exponential => {
            // log y = log a + b·x
            let ly: Vec<f64> = ys.iter().map(|&v| if v > 0.0 { v.ln() } else { f64::NAN }).collect();
            if ly.iter().any(|v| !v.is_finite()) { return None; }
            let (intercept, slope) = simple_lr(xs, &ly);
            Some(vec![intercept.exp(), slope])
        }
        MasModel::StretchedExponential => {
            // y = exp(α·x^{−β} + γ) ⇒ log y = α·x^{−β} + γ
            // Подгоняем как log y = a·x^b + c через brute force по β.
            let ly: Vec<f64> = ys.iter().map(|&v| if v > 0.0 { v.ln() } else { f64::NAN }).collect();
            if ly.iter().any(|v| !v.is_finite()) { return None; }
            // Грид по β, для каждого β линейная регрессия по (x^{-β}, 1).
            let beta_grid: Vec<f64> = (1..=40).map(|i| 0.01 * i as f64).collect();
            let mut best: Option<(f64, Vec<f64>)> = None;
            for &beta in &beta_grid {
                let xb: Vec<f64> = xs.iter().map(|&x| x.powf(-beta)).collect();
                if xb.iter().any(|v| !v.is_finite()) { continue; }
                let (intercept, slope) = simple_lr(&xb, &ly);
                let alpha = slope;
                let gamma = intercept;
                // MSE в исходной шкале
                let mse: f64 = xs.iter().zip(ys.iter())
                    .map(|(&x, &y)| {
                        let pred = (alpha * x.powf(-beta) + gamma).exp();
                        (pred - y).powi(2)
                    })
                    .sum::<f64>() / xs.len() as f64;
                if best.as_ref().map_or(true, |(m, _)| mse < *m) {
                    best = Some((mse, vec![alpha, beta, gamma]));
                }
            }
            best.map(|(_, p)| p)
        }
        MasModel::PowerLaw => {
            // log y = log C − λ · log x
            let ly: Vec<f64> = ys.iter().map(|&v| if v > 0.0 { v.ln() } else { f64::NAN }).collect();
            let lx: Vec<f64> = xs.iter().map(|&v| if v > 0.0 { v.ln() } else { f64::NAN }).collect();
            if ly.iter().chain(lx.iter()).any(|v| !v.is_finite()) { return None; }
            let (intercept, slope) = simple_lr(&lx, &ly);
            Some(vec![intercept, -slope]) // log C, λ
        }
    }
}

/// Простая линейная регрессия y = a + b·x. Возвращает (a, b).
fn simple_lr(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0; let mut den = 0.0;
    for i in 0..x.len() {
        num += (x[i] - mx) * (y[i] - my);
        den += (x[i] - mx).powi(2);
    }
    let b = if den.abs() > 1e-15 { num / den } else { 0.0 };
    (my - b * mx, b)
}

/// Полиномиальная МНК-подгонка через нормальные уравнения.
fn fit_polynomial_least_squares(
    model: MasModel,
    x: &[f64],
    y: &[f64],
) -> Option<Vec<f64>> {
    let degree = match model {
        MasModel::Linear => 1,
        MasModel::Poly2 => 2,
        MasModel::Poly5 => 5,
        _ => return None,
    };
    let p = degree + 1;
    let mut a = vec![vec![0.0; p]; p];
    let mut b = vec![0.0; p];
    for i in 0..p {
        for j in 0..p {
            a[i][j] = x.iter().map(|&v| v.powi((i + j) as i32)).sum();
        }
        b[i] = x.iter().zip(y.iter()).map(|(&vx, &vy)| vx.powi(i as i32) * vy).sum();
    }
    solve_linear(&a, &b)
}

/// Решение СЛАУ методом Гаусса с частичным выбором ведущего элемента.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if m[row][col].abs() > m[max_row][col].abs() { max_row = row; }
        }
        if m[max_row][col].abs() < 1e-12 { return None; }
        m.swap(col, max_row);
        rhs.swap(col, max_row);
        let pivot = m[col][col];
        for j in col..n { m[col][j] /= pivot; }
        rhs[col] /= pivot;
        for row in 0..n {
            if row != col && m[row][col].abs() > 1e-15 {
                let f = m[row][col];
                for j in col..n { m[row][j] -= f * m[col][j]; }
                rhs[row] -= f * rhs[col];
            }
        }
    }
    Some(rhs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasSkip { TooFewHorizons, BadKurtosis, AllFitsFailed }

pub fn mas_skip_reason(horizons: &[f64], kurtoses: &[f64]) -> Option<MasSkip> {
    if horizons.len() != kurtoses.len() || horizons.len() < 8 { return Some(MasSkip::TooFewHorizons); }
    if kurtoses.iter().any(|k| !(k.is_finite() && *k > 0.0)) { return Some(MasSkip::BadKurtosis); }
    None
}

/// 4-фолдовая кросс-валидация: для каждой модели считаем MSE на отложенной выборке.
/// Miyahara et al. (2024), стр. 10–11.
pub fn model_architecture_search(
    horizons: &[f64],
    kurtoses: &[f64],
) -> Option<MasResult> {
    if horizons.len() < 8 || horizons.len() != kurtoses.len() { return None; }

    let n = horizons.len();
    let fold_size = (n / 4).max(1);

    let mut mse_by_model: Vec<(MasModel, f64)> = Vec::new();

    for &model in MasModel::all() {
        let mut sum_mse = 0.0;
        let mut count = 0;

        for fold in 0..4 {
            let test_start = fold * fold_size;
            let test_end = (test_start + fold_size).min(n);

            let mut train_x = Vec::new();
            let mut train_y = Vec::new();
            let mut test_x = Vec::new();
            let mut test_y = Vec::new();

            for i in 0..n {
                if i >= test_start && i < test_end {
                    test_x.push(horizons[i]);
                    test_y.push(kurtoses[i]);
                } else {
                    train_x.push(horizons[i]);
                    train_y.push(kurtoses[i]);
                }
            }
            if train_x.len() < 3 { continue; }

            let coefs = match fit_model(model, &train_x, &train_y) {
                Some(c) => c,
                None => continue,
            };

            let mut fold_mse = 0.0;
            for (i, &x) in test_x.iter().enumerate() {
                let pred = model.predict(x, &coefs);
                if !pred.is_finite() { continue; }
                fold_mse += (pred - test_y[i]).powi(2);
            }
            if !test_x.is_empty() {
                sum_mse += fold_mse / test_x.len() as f64;
                count += 1;
            }
        }
        if count > 0 {
            mse_by_model.push((model, sum_mse / count as f64));
        }
    }

    if mse_by_model.is_empty() { return None; }
    mse_by_model.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (best_model, best_mse) = mse_by_model[0];

    // Финальная подгонка лучшей модели на всех данных
    let best_params = fit_model(best_model, horizons, kurtoses).unwrap_or_default();

    Some(MasResult {
        best_model,
        best_mse,
        mse_by_model,
        best_params,
    })
}

// =====================================================================
// 3. KL-дивергенция (Miyahara et al., 2024, стр. 7)
// =====================================================================

/// KL-дивергенция между двумя дискретными распределениями на общей сетке.
/// Используется для определения режима инвариантности.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    if p.len() != q.len() || p.is_empty() { return f64::NAN; }
    let eps = 1e-12;
    let mut d = 0.0;
    let sp: f64 = p.iter().sum();
    let sq: f64 = q.iter().sum();
    if sp <= 0.0 || sq <= 0.0 { return f64::NAN; }
    for i in 0..p.len() {
        let pi = (p[i] / sp).max(eps);
        let qi = (q[i] / sq).max(eps);
        d += pi * (pi / qi).ln();
    }
    d
}

// =====================================================================
// 4. Обёртка уровня I
// =====================================================================

/// Все результаты уровня I по одному тикеру.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Level1Result {
    pub ticker: String,
    /// (h, K, RK, α) по горизонтам.
    pub by_h: Vec<Level1Point>,
    /// MAS: тип функции убывания эксцесса.
    pub mas_model: Option<String>,
    pub mas_beta: Option<f64>,
    pub mas_c: Option<f64>,
    /// KL-дивергенция между распределением при h=1 и при остальных h.
    pub kl_by_h: Vec<(usize, f64)>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Level1Point {
    pub h: usize,
    pub kurt: f64,
    pub robust_kurt: f64,
    pub alpha: f64,
    pub rk_denom: f64,
    pub n_eff: f64,
    pub rk_nonoverlap: f64,
}

/// Прогон уровня I по ценам одного тикера.
pub fn run_level1(
    ticker: &str,
    closes: &[f64],
    horizons: &[usize],
) -> Level1Result {
    let mut by_h: Vec<Level1Point> = Vec::with_capacity(horizons.len());
    let mut log_h: Vec<f64> = Vec::new();
    let mut log_kurt: Vec<f64> = Vec::new();

    for &h in horizons {
        if closes.len() <= h { continue; }
        let returns: Vec<f64> = (0..closes.len() - h)
            .filter_map(|t| {
                let p0 = closes[t];
                let p1 = closes[t + h];
                if p0 > 0.0 && p1 > 0.0 { Some((p1 / p0).ln()) } else { None }
            })
            .collect();
        if returns.len() < 100 { continue; }

        let kurt = empirical_kurtosis(&returns).unwrap_or(f64::NAN);

        // Робастный эксцесс (Liu et al., 2026) калиброван для
        // СТАНДАРТИЗИРОВАННОГО распределения. При агрегации доходностей
        // масштаб σ(R_{t,h}) растёт как √h, поэтому без нормировки RK
        // отражает масштаб, а не форму распределения.
        //
        // Нормируем ряд на выборочное σ перед вычислением квантилей.
        let std_ret: Vec<f64> = {
            let n_f = returns.len() as f64;
            let mean = returns.iter().sum::<f64>() / n_f;
            let var = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / (n_f - 1.0).max(1.0);
            let sigma = var.sqrt();
            if sigma < 1e-15 {
                returns.clone()
            } else {
                returns.iter().map(|&r| (r - mean) / sigma).collect()
            }
        };

        let mut sorted = std_ret;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rk = robust_kurtosis(&sorted).unwrap_or(f64::NAN);

        // Хвостовой индекс α считаем по СЫРЫМ доходностям —
        // Хилл оценивает абсолютный хвост, стандартизация его исказит.
        let mut sorted_raw = returns.clone();
        sorted_raw.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let alpha = crate::equations::tail_index(&sorted_raw).unwrap_or(f64::NAN);

        let diag = robust_kurtosis_diag(&sorted);
        let rk_denom = diag.map(|d| d.denom).unwrap_or(f64::NAN);
        let n_eff = returns.len() as f64 / (h as f64).max(1.0);
        let rk_nonoverlap = {
            let sub: Vec<f64> = returns.iter().step_by(h.max(1)).copied().collect();
            if sub.len() >= 100 {
                let n_f = sub.len() as f64;
                let m_f = sub.iter().sum::<f64>() / n_f;
                let sd = (sub.iter().map(|r| (r - m_f).powi(2)).sum::<f64>() / (n_f - 1.0)).sqrt();
                if sd > 1e-15 {
                    let mut s: Vec<f64> = sub.iter().map(|r| (r - m_f) / sd).collect();
                    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    robust_kurtosis(&s).unwrap_or(f64::NAN)
                } else { f64::NAN }
            } else { f64::NAN }
        };
        by_h.push(Level1Point {
            h, kurt, robust_kurt: rk, alpha,
            rk_denom, n_eff, rk_nonoverlap,
        });

        if kurt > 0.0 {
            log_h.push((h as f64).ln());
            log_kurt.push(kurt.ln());
        }
    }

    // β из степенного закона (стандартная оценка)
    let horizons_f: Vec<f64> = by_h.iter().map(|p| p.h as f64).collect();
    let kurt_f: Vec<f64> = by_h.iter().map(|p| p.kurt).collect();
    let (beta, c) = estimate_beta(&horizons_f, &kurt_f).unwrap_or((f64::NAN, f64::NAN));

    // MAS
    let mas = model_architecture_search(&horizons_f, &kurt_f);

    // KL-дивергенция относительно базового горизонта h=1.
    // Строим нормированные гистограммы стандартизованных доходностей
    // для каждого h и сравниваем с h=1.
    let mut kl_by_h: Vec<(usize, f64)> = Vec::new();
    let n_bins = 50usize;
    let mut base_hist: Option<Vec<f64>> = None;

    for &h in horizons {
        if closes.len() <= h { continue; }
        let returns: Vec<f64> = (0..closes.len() - h)
            .filter_map(|t| {
                let p0 = closes[t];
                let p1 = closes[t + h];
                if p0 > 0.0 && p1 > 0.0 { Some((p1 / p0).ln()) } else { None }
            })
            .collect();
        if returns.len() < 100 { continue; }

        let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
        let var_r = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
            / (returns.len() - 1).max(1) as f64;
        let sd = var_r.sqrt();
        if sd < 1e-15 { continue; }

        let lo = -5.0_f64;
        let hi = 5.0_f64;
        let bw = (hi - lo) / n_bins as f64;
        let mut hist = vec![0.0_f64; n_bins];
        for &r in &returns {
            let z = (r - mean_r) / sd;
            let idx = ((z - lo) / bw).floor() as isize;
            if idx >= 0 && (idx as usize) < n_bins {
                hist[idx as usize] += 1.0;
            }
        }
        let total: f64 = hist.iter().sum();
        if total > 0.0 {
            for v in hist.iter_mut() { *v /= total; }
        }

        if h == 1 {
            base_hist = Some(hist);
        } else if let Some(ref base) = base_hist {
            let kl = kl_divergence(base, &hist);
            if kl.is_finite() {
                kl_by_h.push((h, kl));
            }
        }
    }

    Level1Result {
        ticker: ticker.to_string(),
        by_h,
        mas_model: mas.as_ref().map(|m| m.best_model.name().to_string()),
        mas_beta: Some(beta).filter(|v| v.is_finite()),
        mas_c: Some(c).filter(|v| v.is_finite()),
        kl_by_h,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winsorized_qk_basic() {
        // Для QK < 1.1 винзоризация почти не меняет значение
        let qk = 1.0;
        let qk_star = winsorized_qk(qk);
        assert!((qk_star - qk).abs() < 0.01);
    }

    #[test]
    fn test_robust_kurtosis_normal() {
        // Для нормального распределения RK ≈ 3
        // (тест на синтетических квантилях нормального распределения)
        let n = 10000;
        let mut v: Vec<f64> = (0..n)
            .map(|i| {
                let p = (i as f64 + 0.5) / n as f64;
                z(p)
            })
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rk = robust_kurtosis(&v).unwrap();
        assert!((rk - 3.0).abs() < 0.3, "RK для нормального = {}", rk);
    }

    #[test]
    fn test_mas_power_law() {
        // Синтетика: K(h) = 10 · h^{-0.8}
        let horizons: Vec<f64> = vec![1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 1440.0];
        let kurtoses: Vec<f64> = horizons.iter().map(|&h| 10.0 * h.powf(-0.8)).collect();
        let result = model_architecture_search(&horizons, &kurtoses).unwrap();
        assert_eq!(result.best_model, MasModel::PowerLaw);
    }
}