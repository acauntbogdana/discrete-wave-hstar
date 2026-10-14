//! Уровень IV. Факторный анализ.
//!
//! Здесь осталась логит-модель для скачков (исходная статья, ур. 18–19).
//! Тест «ускорение эксцесса как рычаг» (SDF + регрессия цены риска,
//! DeMiguel, Martín-Utrera & Upfal, 2024, ур. 13, 17) переписан и вынесен в
//! `leverage.rs` (подмодуль `factors::leverage`): прежняя реализация смешивала
//! временные интервалы регрессора и отклика и давала ~37% ложных срабатываний
//! на данных без эффекта.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[path = "leverage.rs"]
pub mod leverage;

#[path = "escape.rs"]
pub mod escape;

#[path = "adaptive.rs"]
pub mod adaptive;

// =====================================================================
// 4. Логит-модель (перенос из старого equations.rs)
// =====================================================================

fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }

fn linear_predictor(x: ArrayView1<f64>, beta: ArrayView1<f64>) -> f64 {
    x.dot(&beta)
}

pub fn fit_logistic(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
) -> (Array1<f64>, Array2<f64>, Array2<f64>) {
    let n = x.nrows();
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    let max_iter = 100;
    let tol = 1e-8;
    let eps = 1e-12;

    for _ in 0..max_iter {
        let mut z = Array1::<f64>::zeros(n);
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            let eta = linear_predictor(x.row(i), beta.view());
            let prob = sigmoid(eta);
            w[i] = (prob * (1.0 - prob)).max(eps);
            z[i] = eta + (y[i] - prob) / w[i];
        }
        // IRLS через нормальные уравнения
        let mut xtwx = Array2::<f64>::zeros((p, p));
        let mut xtwz = Array1::<f64>::zeros(p);
        for i in 0..n {
            let wi = w[i];
            for j in 0..p {
                xtwz[j] += x[[i, j]] * wi * z[i];
                for k in 0..p {
                    xtwx[[j, k]] += x[[i, j]] * wi * x[[i, k]];
                }
            }
        }
        // Решение системы
        let xtwx_m = nalgebra::DMatrix::from_row_slice(p, p, xtwx.as_slice().unwrap());
        let xtwz_v = nalgebra::DVector::from_row_slice(xtwz.as_slice().unwrap());
        let beta_new = match xtwx_m.lu().solve(&xtwz_v) {
            Some(v) => Array1::from_shape_vec(p, v.iter().copied().collect()).unwrap(),
            None => break,
        };
        let diff = (&beta_new - &beta).mapv(f64::abs).sum();
        beta = beta_new;
        if diff < tol { break; }
    }

    // Гессиан и скоры
    let mut h = Array2::<f64>::zeros((p, p));
    let mut scores = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        let eta = linear_predictor(x.row(i), beta.view());
        let prob = sigmoid(eta);
        let wi = prob * (1.0 - prob);
        let residual = y[i] - prob;
        for j in 0..p {
            scores[[i, j]] = x[[i, j]] * residual;
            for k in 0..p {
                h[[j, k]] += x[[i, j]] * wi * x[[i, k]];
            }
        }
    }
    (beta, h, scores)
}

pub fn predict_proba(x: ArrayView2<f64>, beta: ArrayView1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let mut p = Array1::<f64>::zeros(n);
    for i in 0..n {
        p[i] = sigmoid(linear_predictor(x.row(i), beta));
    }
    p
}

pub fn auc_score(y: ArrayView1<f64>, p: ArrayView1<f64>) -> f64 {
    let n = y.len();
    let n_pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return f64::NAN; }
    let mut pairs: Vec<(f64, f64)> = (0..n).map(|i| (p[i], y[i])).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut tp = 0.0; let mut auc = 0.0;
    for (_, yi) in &pairs {
        if *yi > 0.5 { tp += 1.0; } else { auc += tp; }
    }
    auc / (n_pos * n_neg)
}

/// Псевдо-R² МакФаддена.
pub fn pseudo_r2_mcfadden(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    beta: ArrayView1<f64>,
) -> f64 {
    let n = y.len() as f64;
    let mut ll_model = 0.0;
    let eps = 1e-12;
    for i in 0..x.nrows() {
        let eta = linear_predictor(x.row(i), beta);
        let p = sigmoid(eta).clamp(eps, 1.0 - eps);
        ll_model += y[i] * p.ln() + (1.0 - y[i]) * (1.0 - p).ln();
    }
    let p_bar = (y.sum() / n).clamp(eps, 1.0 - eps);
    let ll_null = n * (p_bar * p_bar.ln() + (1.0 - p_bar) * (1.0 - p_bar).ln());
    if ll_null.abs() < 1e-12 { return f64::NAN; }
    1.0 - ll_model / ll_null
}