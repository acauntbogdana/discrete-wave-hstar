//! Уровень VIII. Двухслойная адаптивная модель.
//!
//! Внутренний слой: AR(1) с рекурсивным МНК и забыванием λ = 1 − 1/H*(μ).
//! Внешний слой: логит предсказания смены режима по мета-признакам
//!              (rolling std, skew, kurt, Φ-доля).
//! Композиция: ŷ = ŷ_inner · (1 − P̂_regime_change).
//!
//! Проверки: MSE inner vs two-layer, Sharpe sign-стратегии, AUC логита
//! vs persistence. Forward-walk: обучение на train, тест на test.

use crate::factors::escape::JumpRecord;

// =====================================================================
// Вспомогательные
// =====================================================================

fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = (-z).exp();
        1.0 / (1.0 + e)
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_gauss(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if m[r][col].abs() > m[piv][col].abs() { piv = r; }
        }
        if m[piv][col].abs() < 1e-12 { return None; }
        m.swap(col, piv);
        rhs.swap(col, piv);
        let p = m[col][col];
        for j in col..n { m[col][j] /= p; }
        rhs[col] /= p;
        for r in 0..n {
            if r != col && m[r][col].abs() > 1e-15 {
                let f = m[r][col];
                for j in col..n { m[r][j] -= f * m[col][j]; }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    Some(rhs)
}

fn mean(x: &[f64]) -> f64 {
    let v: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if v.is_empty() { return f64::NAN; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std(x: &[f64]) -> f64 {
    let v: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if v.len() < 2 { return f64::NAN; }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn skew(x: &[f64]) -> f64 {
    let v: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = v.len() as f64;
    if n < 3.0 { return f64::NAN; }
    let m = v.iter().sum::<f64>() / n;
    let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m3 = v.iter().map(|x| (x - m).powi(3)).sum::<f64>() / n;
    if m2 < 1e-15 { return 0.0; }
    m3 / m2.powf(1.5)
}

fn kurt(x: &[f64]) -> f64 {
    let v: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = v.len() as f64;
    if n < 4.0 { return f64::NAN; }
    let m = v.iter().sum::<f64>() / n;
    let m2 = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
    let m4 = v.iter().map(|x| (x - m).powi(4)).sum::<f64>() / n;
    if m2 < 1e-15 { return 0.0; }
    m4 / (m2 * m2) - 3.0
}

// =====================================================================
// Внутренний слой: адаптивный AR(1) с забыванием
// =====================================================================

/// Рекурсивный МНК с забыванием λ:
///   θ_t = θ_{t-1} + P_t x_t (y_t − x_t' θ_{t-1})
///   P_t = (1/λ)(P_{t-1} − P_{t-1} x_t x_t' P_{t-1} / (λ + x_t' P_{t-1} x_t))
/// Регрессор x_t = [1, r_{t-1}], отклик y_t = r_t.
pub struct AdaptiveAr1 {
    theta: [f64; 2],
    p: [[f64; 2]; 2],
    lambda: f64,
    pub n: usize,
}

impl AdaptiveAr1 {
    pub fn new(lambda: f64) -> Self {
        Self {
            theta: [0.0, 0.0],
            p: [[1e8, 0.0], [0.0, 1e8]],
            lambda: lambda.clamp(0.90, 0.9999),
            n: 0,
        }
    }

    pub fn update(&mut self, x: f64, y: f64) {
        let xt = [1.0, x];
        // P x
        let px = [
            self.p[0][0] * xt[0] + self.p[0][1] * xt[1],
            self.p[1][0] * xt[0] + self.p[1][1] * xt[1],
        ];
        let xtpx = xt[0] * px[0] + xt[1] * px[1];
        let denom = self.lambda + xtpx;
        if denom.abs() < 1e-15 { return; }
        let inv_d = 1.0 / denom;

        let pxxp = [
            [px[0]*px[0], px[0]*px[1]],
            [px[1]*px[0], px[1]*px[1]],
        ];
        let inv_lam = 1.0 / self.lambda;
        for i in 0..2 {
            for j in 0..2 {
                self.p[i][j] = inv_lam * (self.p[i][j] - pxxp[i][j] * inv_d);
            }
        }

        let pred = xt[0] * self.theta[0] + xt[1] * self.theta[1];
        let e = y - pred;
        self.theta[0] += px[0] * e;
        self.theta[1] += px[1] * e;
        self.n += 1;
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.theta[0] + self.theta[1] * x
    }
}

// =====================================================================
// Внешний слой: логит смены режима
// =====================================================================

pub struct RegimeLogit {
    beta: Vec<f64>,
}

impl RegimeLogit {
    pub fn fit(x: &[Vec<f64>], y: &[f64]) -> Self {
        let n = x.len();
        if n < 50 { return Self { beta: vec![0.0] }; }
        let p = x[0].len();
        let mut beta = vec![0.0; p];
        for _ in 0..25 {
            let mut w = vec![0.0; n];
            let mut z = vec![0.0; n];
            for i in 0..n {
                let eta = dot(&x[i], &beta);
                let prob = sigmoid(eta).clamp(1e-10, 1.0 - 1e-10);
                w[i] = (prob * (1.0 - prob)).max(1e-10);
                z[i] = eta + (y[i] - prob) / w[i];
            }
            let mut xtwx = vec![vec![0.0; p]; p];
            let mut xtwz = vec![0.0; p];
            for i in 0..n {
                for j in 0..p {
                    xtwz[j] += x[i][j] * w[i] * z[i];
                    for k in 0..p {
                        xtwx[j][k] += x[i][j] * w[i] * x[i][k];
                    }
                }
            }
            let beta_new = match solve_gauss(&xtwx, &xtwz) {
                Some(b) => b,
                None => break,
            };
            let diff: f64 = beta_new.iter().zip(beta.iter()).map(|(a, b)| (a - b).abs()).sum();
            beta = beta_new;
            if diff < 1e-7 { break; }
        }
        Self { beta }
    }

    pub fn predict_proba(&self, x: &[f64]) -> f64 {
        sigmoid(dot(x, &self.beta))
    }
}

// =====================================================================
// Мета-признаки
// =====================================================================

/// Признаки для логита: std, |skew|, kurt, phi_frac.
/// Всё в скользящем окне W.
fn make_features(
    returns: &[f64],
    jumps: &[JumpRecord],
    t: usize,
    w: usize,
    day_len: usize,
) -> Option<[f64; 4]> {
    if t < w { return None; }
    let win = &returns[t - w..t];
    let s = std(win);
    let sk = skew(win);
    let ku = kurt(win);
    if !s.is_finite() || !ku.is_finite() { return None; }

    // Доля скачков с Φ>1 в окне [t−day_len, t)
    let lo = t.saturating_sub(day_len);
    let mut n_jumps = 0usize;
    let mut n_passed = 0usize;
    for jr in jumps {
        if jr.idx >= lo && jr.idx < t {
            n_jumps += 1;
            if jr.passed { n_passed += 1; }
        }
    }
    let phi_frac = if n_jumps > 0 { n_passed as f64 / n_jumps as f64 } else { 0.0 };

    Some([s, sk.abs(), ku, phi_frac])
}

// =====================================================================
// Результат
// =====================================================================

#[derive(Debug, Clone, Default)]
pub struct AdaptiveResult {
    pub n_train: usize,
    pub n_test: usize,
    pub mse_inner: f64,
    pub mse_two_layer: f64,
    pub sharpe_inner: f64,
    pub sharpe_two_layer: f64,
    pub auc_regime: f64,
    pub auc_persistence: f64,
    pub forgetting_lambda: f64,
}

fn auc(y: &[f64], p: &[f64]) -> f64 {
    let n = y.len();
    let n_pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
    let n_neg = n as f64 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return f64::NAN; }
    let mut pairs: Vec<(f64, f64)> = (0..n).map(|i| (p[i], y[i])).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut tp = 0.0;
    let mut a = 0.0;
    for (_, yi) in &pairs {
        if *yi > 0.5 { tp += 1.0; } else { a += tp; }
    }
    a / (n_pos * n_neg)
}

// =====================================================================
// Forward-walk
// =====================================================================

pub fn run_adaptive(
    log_returns: &[f64],
    jumps: &[JumpRecord],
    h_star_mu: usize,
    window: usize,
    day_len: usize,
) -> Option<AdaptiveResult> {
    let n = log_returns.len();
    if n < 5000 { return None; }

    // λ из H*(μ): эффективная память ≈ H*, λ = 1 − 1/H*.
    let hs = h_star_mu.max(20).min(500);
    let lambda = 1.0 - 1.0 / hs as f64;

    // Train/test: 60/40.
    let split = (n * 6) / 10;
    let n_test = n - split;
    if n_test < 500 { return None; }

    // Метка смены режима: |log(σ²_{t+60}/σ²_t)| > log(1.5).
    let log_thr = 1.5_f64.ln();
    let regime_label = |t: usize| -> f64 {
        if t + 2 * window > n { return f64::NAN; }
        let v_now: f64 = log_returns[t..t + window].iter().map(|x| x * x).sum::<f64>() / window as f64;
        let v_next: f64 = log_returns[t + window..t + 2 * window].iter().map(|x| x * x).sum::<f64>() / window as f64;
        if v_now <= 1e-15 || v_next <= 1e-15 { return f64::NAN; }
        let ratio = (v_next / v_now).ln();
        if ratio.abs() > log_thr { 1.0 } else { 0.0 }
    };

    // Forward-walk.
    let mut ar = AdaptiveAr1::new(lambda);
    // Прогреваем AR на train.
    for t in 1..split {
        ar.update(log_returns[t - 1], log_returns[t]);
    }

    // Логит обучен на train.
    let mut train_x: Vec<Vec<f64>> = Vec::new();
    let mut train_y: Vec<f64> = Vec::new();
    let step = 5;
    for t in (window..split).step_by(step) {
        if let Some(f) = make_features(log_returns, jumps, t, window, day_len) {
            let lab = regime_label(t);
            if lab.is_finite() {
                train_x.push(f.to_vec());
                train_y.push(lab);
            }
        }
    }
    if train_x.len() < 50 { return None; }
    let mut logit = RegimeLogit::fit(&train_x, &train_y);

    // Тест.
    let mut errs_inner: Vec<f64> = Vec::new();
    let mut errs_two: Vec<f64> = Vec::new();
    let mut ret_sign_inner: Vec<f64> = Vec::new();
    let mut ret_sign_two: Vec<f64> = Vec::new();
    let mut test_labels: Vec<f64> = Vec::new();
    let mut test_probs: Vec<f64> = Vec::new();
    let mut pers_labels: Vec<f64> = Vec::new();
    let mut pers_probs: Vec<f64> = Vec::new();

    let retrain_every = day_len * 5;
    let mut last_retrain = split;

    for t in split..(n - 1) {
        // Переобучение логита каждые 5 дней на скользящем окне.
        if t - last_retrain >= retrain_every && t > window + 500 {
            let lo = t.saturating_sub(15000);
            let mut tx: Vec<Vec<f64>> = Vec::new();
            let mut ty: Vec<f64> = Vec::new();
            for s in (lo.max(window)..t).step_by(step) {
                if let Some(f) = make_features(log_returns, jumps, s, window, day_len) {
                    let lab = regime_label(s);
                    if lab.is_finite() { tx.push(f.to_vec()); ty.push(lab); }
                }
            }
            if tx.len() >= 50 { logit = RegimeLogit::fit(&tx, &ty); }
            last_retrain = t;
        }

        // Обновление AR.
        ar.update(log_returns[t - 1], log_returns[t]);

        // Прогноз.
        let pred_inner = ar.predict(log_returns[t]);

        let pred_two = if let Some(f) = make_features(log_returns, jumps, t, window, day_len) {
            let p = logit.predict_proba(&f);
            pred_inner * (1.0 - p)
        } else {
            pred_inner
        };

        let y = log_returns[t + 1];
        errs_inner.push((y - pred_inner).powi(2));
        errs_two.push((y - pred_two).powi(2));

        // Sign strategy: длинная если pred > 0, короткая иначе.
        ret_sign_inner.push(pred_inner.signum() * y);
        ret_sign_two.push(pred_two.signum() * y);

        // Логит-тест для AUC.
        if let Some(f) = make_features(log_returns, jumps, t, window, day_len) {
            let lab = regime_label(t);
            if lab.is_finite() {
                test_labels.push(lab);
                test_probs.push(logit.predict_proba(&f));
                // Persistence baseline: P = 1 если метка вчера = 1.
                let prev_lab = if t > split { test_labels[test_labels.len().saturating_sub(2)] } else { 0.5 };
                pers_labels.push(lab);
                pers_probs.push(prev_lab);
            }
        }
    }

    if errs_inner.len() < 500 { return None; }

    let mse_inner = errs_inner.iter().sum::<f64>() / errs_inner.len() as f64;
    let mse_two = errs_two.iter().sum::<f64>() / errs_two.len() as f64;

    let sharpe = |r: &[f64]| -> f64 {
        let m = mean(r);
        let s = std(r);
        if s < 1e-15 { 0.0 } else { m / s }
    };

    let auc_regime = auc(&test_labels, &test_probs);
    let auc_pers = auc(&pers_labels, &pers_probs);

    Some(AdaptiveResult {
        n_train: split,
        n_test: n_test,
        mse_inner,
        mse_two_layer: mse_two,
        sharpe_inner: sharpe(&ret_sign_inner),
        sharpe_two_layer: sharpe(&ret_sign_two),
        auc_regime,
        auc_persistence: auc_pers,
        forgetting_lambda: lambda,
    })
}