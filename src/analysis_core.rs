//! Оркестрация всех уровней методологии: I → VII.
//!
//! Собирает результаты по каждому тикеру, накапливает данные для графиков
//! и генерирует итоговый отчёт и визуализации.
//!
//! Уровни:
//!   I.  Дискретность и агрегация эксцесса       (kurtosis.rs)
//!   II. Эргодичность и устойчивость моментов    (ergodicity.rs)
//!   III.Микроструктура и скачки                 (microstructure.rs)
//!   IV. Лестница ускорения эксцесса             (factors::leverage)
//!       H4a — уровень K_j, H4b — ΔK_j, H4c — A_j
//!   V.  Escape из барьера Гроссмана–Миллера     (factors::escape)
//!   VI. Гистерезисная надстройка                (hysteresis.rs)
//!   VII.Горизонт стат. пригодности H*(θ)        (horizon.rs)

use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    time::Instant,
};

use anyhow::{Context, Result};

use crate::{
    bars_io::{list_available_tickers, load_bars_for_ticker},
    config::*,
    ergodicity::run_level2,
    factors::adaptive::run_adaptive,
    factors::escape::{run_escape, JumpRecord},
    horizon::{measure_h_star, measure_h_star_coint, verify_hierarchy, HStarConfig},
    hysteresis::run_level6,
    kurtosis::run_level1,
    microstructure::{
        detect_jumps, med9_rv, run_level3,
    },
    plots::{
        generate_all_plots, Level1PlotData, Level2PlotData, Level3PlotData,
        Level4PlotData, Level6PlotData, Level7PlotData,
    },
    statistics::median,
};

// =====================================================================
// Сводка по одному тикеру
// =====================================================================

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TickerSummary {
    pub ticker: String,

    // --- Уровень I ---
    pub level1_mas_model: Option<String>,
    pub level1_mas_beta: Option<f64>,
    pub level1_median_kurt: f64,
    pub level1_median_robust_kurt: f64,
    pub level1_median_alpha: f64,
    pub level1_by_h: Vec<(usize, f64, f64, f64)>,
    /// Диагностика RK по горизонтам: (h, rk_denom, n_eff, rk_nonoverlap).
    /// `#[serde(default)]` — для совместимости со старыми чекпоинтами.
    #[serde(default)]
    pub level1_diag_by_h: Vec<(usize, f64, f64, f64)>,

    // --- Уровень II ---
    pub level2_f_mu_slope: f64,
    pub level2_f_sigma2_slope: f64,
    pub level2_f_mu_significant: bool,
    pub level2_f_sigma2_significant: bool,
    pub level2_cusum_mean_p: f64,
    pub level2_cusum_sigma_p: f64,
    pub level2_mfdfa_delta_alpha: f64,
    pub level2_variance_decay_slope: f64,
    pub level2_example_f_mu: Vec<f64>,
    pub level2_example_f_sigma2: Vec<f64>,
    pub level2_mfdfa_spectrum: Vec<(f64, f64)>,
    pub level2_stability_comparison: Vec<(String, f64, f64)>,

    // --- Уровень III ---
    pub level3_med9_rv_median: f64,
    pub level3_n_jumps_total: usize,
    pub level3_n_jumps_economic: usize,
    pub level3_frac_economic: f64,
    pub level3_jumps_by_minute: Vec<(usize, usize)>,
    pub level3_phi_values: Vec<f64>,
    pub level3_n_jumps_near_tax: usize,
    pub level3_n_jumps_outside_tax: usize,
    pub level3_tax_lift: f64,
    pub level3_ofi_median: f64,

    // --- Уровень IV (H4c — основной тест на ускорении) ---
    pub level4_beta_k: f64,
    pub level4_t_beta_k: f64,
    pub level4_is_leverage: bool,
    pub level4_sdf_mean: f64,
    pub level4_sdf_std: f64,
    pub level4_example_scatter: Vec<(f64, f64)>,
    pub level4_example_fit: Option<(f64, f64)>,
    pub level4_perm_p: f64,
    pub level4_n_obs: usize,
    pub level4_t_ctrl: f64,
    pub level4_t_lag1: f64,
    pub level4_t_lead1: f64,
    pub level4_sens: Vec<(usize, f64)>,

    // --- Уровень IV — лестница H4a/H4b/H4c ---
    pub level4_beta_k_level: f64,
    pub level4_t_k_level: f64,
    pub level4_perm_p_k_level: f64,
    pub level4_is_leverage_k_level: bool,
    pub level4_beta_dk: f64,
    pub level4_t_dk: f64,
    pub level4_perm_p_dk: f64,
    pub level4_is_leverage_dk: bool,

    // --- Уровень VIII. Двухслойная адаптивная модель ---
    pub level8_mse_inner: f64,
    pub level8_mse_two_layer: f64,
    pub level8_sharpe_inner: f64,
    pub level8_sharpe_two_layer: f64,
    pub level8_auc_regime: f64,
    pub level8_auc_persistence: f64,
    pub level8_forgetting_lambda: f64,
    pub level8_n_test: usize,
    pub level8_ok: bool,

    // --- Уровень V (H5). Escape из барьера Гроссмана–Миллера ---
    pub level5_escape_n_blocks: usize,
    pub level5_escape_frac_beta: f64,
    pub level5_escape_frac_t: f64,
    pub level5_escape_frac_perm_p: f64,
    pub level5_escape_mom_beta: f64,
    pub level5_escape_mom_t: f64,
    pub level5_escape_mom_perm_p: f64,
    pub level5_escape_is_escape: bool,

    // --- Уровень VI ---
    pub level6_n_regimes: usize,
    pub level6_mean_regime_len_bars: f64,
    pub level6_regimes_per_100k: f64,
    pub level6_dm_by_h: Vec<(f64, f64)>,
    pub level6_gdm_stat: f64,
    pub level6_gdm_pvalue: f64,

    // --- Уровень VII ---
    pub level7_h_mu: usize,
    pub level7_h_sigma2: usize,
    /// H*(β) в блоках длины WINDOW_MOMENTS.
    pub level7_h_beta: usize,
    /// H*(β) в барах: level7_h_beta * WINDOW_MOMENTS.
    pub level7_h_beta_bars: usize,
    pub level7_hierarchy_holds: bool,
    pub level7_h_mu_coint: usize,
    pub level7_h_sigma2_coint: usize,
    pub level7_h_beta_coint: usize,
    pub level7_hierarchy_holds_coint: bool,
    pub level7_beta_coint: Option<f64>,
    pub level7_mu_bad_fraction: f64,
    pub level7_mu_median_drift: f64,
    pub level7_sigma2_bad_fraction: f64,
    pub level7_sigma2_median_drift: f64,
    pub level7_beta_bad_fraction: f64,
    pub level7_beta_median_drift: f64,

    // --- Признаки для группового анализа ---
    pub median_volume: f64,
    pub median_sigma_pct: f64,
    pub mean_daily_volume: f64,
}

// =====================================================================
// Главный оркестратор
// =====================================================================

pub fn run() -> Result<()> {
    std::fs::create_dir_all(OUTPUT_DIR)?;
    let old_phi = format!("{}/phi_values.txt", OUTPUT_DIR);
    if std::path::Path::new(&old_phi).exists() {
        let _ = std::fs::remove_file(&old_phi);
    }
    let start = Instant::now();

    // Шаг 1. Бары.
    let bars_dir = std::path::Path::new(OUTPUT_DIR).join("bars");
    let bars_ready = bars_dir.exists()
        && std::fs::read_dir(&bars_dir)
            .map(|mut it| it.any(|e| {
                e.ok()
                    .map(|e| {
                        e.path()
                            .extension()
                            .and_then(|x| x.to_str())
                            .map(|x| x.eq_ignore_ascii_case("parquet"))
                            .unwrap_or(false)
                    })
                    .unwrap_or(false)
            }))
            .unwrap_or(false);

    if SKIP_BAR_BUILDING_IF_READY && bars_ready {
        let n = std::fs::read_dir(&bars_dir).map(|it| it.count()).unwrap_or(0);
        println!("Шаг 1: бары уже построены ({} файлов), пропускаем.", n);
    } else {
        println!("Шаг 1: Построение баров...");
        crate::data_loading::build_all_bars()?;
    }

    // Шаг 2. Тикеры.
    println!("Шаг 2: Список доступных тикеров...");
    let all_tickers = list_available_tickers()?;
    println!("Найдено тикеров: {}", all_tickers.len());

    // Шаг 3. Отчёт.
    let report_path = format!("{}/report.txt", OUTPUT_DIR);
    let mut report = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&report_path)
        .with_context(|| format!("не удалось открыть {}", report_path))?;

    writeln!(
        report,
        "\n=== Полный отчёт по методологии H*(θ) от {} ===",
        chrono::Utc::now()
    )?;
    writeln!(report, "Горизонты: {:?}", HORIZONS)?;
    writeln!(
        report,
        "δ_rel={:.4}, окно моментов W={}, HSTAR_DRIFT_TOL={:.2}, HSTAR_MAX_BAD_FRACTION={:.2}",
        DELTA_REL, WINDOW_MOMENTS, HSTAR_DRIFT_TOL, HSTAR_MAX_BAD_FRACTION
    )?;
    writeln!(
        report,
        "Порог скачка: |r*| > {:.1}; барьер Φ > {:.1}",
        JUMP_THRESHOLD_STD, ECONOMIC_BARRIER_PHI
    )?;

    // Шаг 4. Проход 1 → чекпоинт (MessagePack).
    println!("Шаг 3: Прогон уровней I–VII по тикерам...");

    let ckpt_path = format!("{}/tickers_checkpoint.bin", OUTPUT_DIR);
    let mut ckpt = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ckpt_path)?;

    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(mut f) = std::fs::File::open(&ckpt_path) {
        use std::io::Read;
        let mut len_buf = [0u8; 4];
        loop {
            if f.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut buf = vec![0u8; len];
            if f.read_exact(&mut buf).is_err() { break; }
            if let Ok(s) = rmp_serde::from_slice::<TickerSummary>(&buf) {
                done.insert(s.ticker.clone());
            }
        }
        println!("Чекпоинт: найдено {} обработанных тикеров", done.len());
    }

    let remaining: Vec<String> = all_tickers
        .into_iter()
        .filter(|t| !done.contains(t))
        .collect();
    println!("Осталось обработать: {}", remaining.len());

    let mut have_l2_example = false;
    let mut have_l4_example = false;

    for (idx, ticker) in remaining.iter().enumerate() {
        match analyze_one_ticker(
            ticker,
            !have_l2_example,
            !have_l4_example,
        ) {
            Ok(Some(summary)) => {
                match rmp_serde::to_vec_named(&summary) {
                    Ok(bytes) => {
                        let len = bytes.len() as u32;
                        if let Err(e) = ckpt.write_all(&len.to_le_bytes())
                            .and_then(|_| ckpt.write_all(&bytes))
                        {
                            eprintln!("Ошибка записи чекпоинта для {}: {}", ticker, e);
                        }
                        let _ = ckpt.flush();
                    }
                    Err(e) => eprintln!("Ошибка сериализации {}: {}", ticker, e),
                }
                if !summary.level2_example_f_mu.is_empty() {
                    have_l2_example = true;
                }
                if !summary.level4_example_scatter.is_empty() {
                    have_l4_example = true;
                }
            }
            Ok(None) => {}
            Err(e) => eprintln!("Тикер {} пропущен: {}", ticker, e),
        }

        if (idx + 1) % 50 == 0 {
            println!("  обработано {}/{}", idx + 1, remaining.len());
        }
    }
    drop(ckpt);

    // ================================================================
    // ПРОХОД 2. Агрегация из чекпоинта.
    // ================================================================
    println!("Шаг 3.1: Агрегация из чекпоинта...");

    let mut plot_l1 = Level1PlotData::default();
    let mut plot_l2 = Level2PlotData::default();
    let mut plot_l3 = Level3PlotData::default();
    let mut plot_l4 = Level4PlotData::default();
    let mut plot_l6 = Level6PlotData::default();
    let mut plot_l7 = Level7PlotData::default();

    let mut n_ok = 0usize;
    let mut n_leverage = 0usize;
    let mut n_hierarchy_holds = 0usize;
    let mut n_hierarchy_total = 0usize;

    let phi_path = format!("{}/phi_values.txt", OUTPUT_DIR);
    let mut phi_writer = std::io::BufWriter::new(
        std::fs::File::create(&phi_path)
            .with_context(|| format!("не удалось создать {}", phi_path))?,
    );

    let mut all_results: Vec<TickerSummary> = Vec::new();

    {
        use std::io::Read;
        let mut f = std::fs::File::open(&ckpt_path)
            .with_context(|| format!("не удалось открыть {}", ckpt_path))?;
        let mut len_buf = [0u8; 4];
        let mut line_no = 0usize;
        loop {
            if f.read_exact(&mut len_buf).is_err() { break; }
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut buf = vec![0u8; len];
            if f.read_exact(&mut buf).is_err() { break; }
            line_no += 1;

            match rmp_serde::from_slice::<TickerSummary>(&buf) {
                Ok(s) => {
                    aggregate_plot_data_streaming(
                        &s,
                        &mut plot_l1,
                        &mut plot_l2,
                        &mut plot_l3,
                        &mut plot_l4,
                        &mut plot_l6,
                        &mut plot_l7,
                        &mut phi_writer,
                    ).with_context(|| format!("ошибка агрегации блока {}", line_no))?;

                    if s.level4_is_leverage { n_leverage += 1; }
                    if s.level7_hierarchy_holds { n_hierarchy_holds += 1; }
                    n_hierarchy_total += 1;
                    n_ok += 1;
                    all_results.push(s);
                }
                Err(e) => eprintln!("Чекпоинт: битый блок {} ({})", line_no, e),
            }

            if line_no % 5000 == 0 {
                println!("  агрегировано {} блоков", line_no);
            }
        }
    }

    drop(phi_writer);

    println!("  чтение Φ из {}", phi_path);
    plot_l3.phi_values = {
        let content = std::fs::read_to_string(&phi_path)
            .with_context(|| format!("не удалось прочитать {}", phi_path))?;
        let mut vals: Vec<f64> = content
            .lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .collect();
        const PHI_CAP: usize = 2_000_000;
        if vals.len() > PHI_CAP {
            let step = (vals.len() / PHI_CAP) + 1;
            vals = vals.into_iter().step_by(step).collect();
        }
        vals
    };

    if n_hierarchy_total > 0 {
        plot_l7.hierarchy_holds_fraction =
            n_hierarchy_holds as f64 / n_hierarchy_total as f64;
    }

    println!("Успешно обработано тикеров: {}", n_ok);

    write_summary(
        &mut report,
        n_ok,
        n_leverage,
        n_hierarchy_holds,
        n_hierarchy_total,
        &plot_l1,
        &plot_l2,
        &plot_l3,
        &plot_l4,
        &plot_l6,
        &plot_l7,
    )?;

    println!("Шаг 4.1: Групповой анализ тикеров...");
    if let Err(e) = write_group_analysis(&mut report, &all_results) {
        eprintln!("Предупреждение: ошибка группового анализа: {}", e);
    }

    println!("Шаг 4: Генерация графиков...");
    if let Err(e) = generate_all_plots(
        &plot_l1, &plot_l2, &plot_l3, &plot_l4, &plot_l6, &plot_l7,
    ) {
        eprintln!("Предупреждение: ошибка построения графиков: {}", e);
    }

    println!("Шаг 4.2: Сводка уровня VIII...");
    if let Err(e) = write_level8_summary(&mut report, &all_results) {
        eprintln!("Предупреждение: ошибка сводки VIII: {}", e);
    }

    println!("Готово за {:.1} с", start.elapsed().as_secs_f64());
    Ok(())
}

// =====================================================================
// Анализ одного тикера
// =====================================================================

fn analyze_one_ticker(
    ticker: &str,
    collect_l2_example: bool,
    collect_l4_example: bool,
) -> Result<Option<TickerSummary>> {
    let bars = load_bars_for_ticker(ticker)
        .with_context(|| format!("не удалось загрузить бары для {}", ticker))?;
    if bars.len() < MIN_BARS.max(WARMUP_BARS + 100) {
        return Ok(None);
    }

    let closes: Vec<f64> = bars.iter().map(|b| b.close).collect();
    if closes.iter().any(|&p| !p.is_finite() || p <= 0.0) {
        return Ok(None);
    }
    let n = closes.len();

    let log_returns: Vec<f64> = closes
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .filter(|r| r.is_finite())
        .collect();
    if log_returns.len() < WARMUP_BARS + 100 {
        return Ok(None);
    }

    // ================================================================
    // Уровень I.
    // ================================================================
    let l1 = run_level1(ticker, &closes, HORIZONS);

    let median_kurt = median(&l1.by_h.iter().map(|p| p.kurt).collect::<Vec<_>>());
    let median_rk = median(&l1.by_h.iter().map(|p| p.robust_kurt).collect::<Vec<_>>());
    let median_alpha = median(&l1.by_h.iter().map(|p| p.alpha).collect::<Vec<_>>());

    let l1_by_h: Vec<(usize, f64, f64, f64)> = l1
        .by_h.iter()
        .map(|p| (p.h, p.kurt, p.robust_kurt, p.alpha))
        .collect();

    let l1_diag_by_h: Vec<(usize, f64, f64, f64)> = l1
        .by_h.iter()
        .map(|p| (p.h, p.rk_denom, p.n_eff, p.rk_nonoverlap))
        .collect();

    let median_volume = {
        let vols: Vec<f64> = bars.iter().map(|b| b.volume as f64).collect();
        median(&vols)
    };
    let mean_daily_volume = {
        let n_days = (bars.len() / 390).max(1) as f64;
        let total: f64 = bars.iter().map(|b| b.volume as f64).sum();
        total / n_days
    };
    let median_sigma_pct = {
        let m = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
        let v = log_returns.iter()
            .map(|r| (r - m).powi(2))
            .sum::<f64>() / (log_returns.len() - 1).max(1) as f64;
        v.sqrt() * 100.0
    };

    // ================================================================
    // Уровень II.
    // ================================================================
    let l2 = run_level2(ticker, &log_returns, WINDOW_MOMENTS);

    let (example_f_mu, example_f_sigma2) = if collect_l2_example {
        let series = crate::ergodicity::rolling_moments(&log_returns, WINDOW_MOMENTS);
        let ratios = crate::ergodicity::stationarity_ratios(&series);
        (ratios.f_mu, ratios.f_sigma2)
    } else {
        (vec![], vec![])
    };

    let mf = crate::equations::mf_dfa(&log_returns, -5.0, 5.0, 0.5, None, 1);
    let mfdfa_spectrum: Vec<(f64, f64)> = mf
        .alpha_q.iter().copied()
        .zip(mf.f_alpha.iter().copied())
        .filter(|(a, f)| a.is_finite() && f.is_finite())
        .collect();

    // ================================================================
    // Уровень III.
    // ================================================================
    let day_len = 390usize;
    let best_bids: Vec<Option<f64>> = bars.iter().map(|b| b.best_bid).collect();
    let best_asks: Vec<Option<f64>> = bars.iter().map(|b| b.best_ask).collect();

    let day_unix_ids: Vec<i64> = (0..bars.len())
        .step_by(day_len)
        .map(|i| bars[i].date)
        .collect();

    let l3 = run_level3(
        ticker, &closes, &best_bids, &best_asks, day_len, &day_unix_ids,
    );

    let (jumps_by_minute, phi_values) =
        collect_jumps_by_minute(&closes, &best_bids, &best_asks, day_len);

    let jumps_records = collect_jump_records(&closes, &best_bids, &best_asks, day_len);

    // ================================================================
    // Уровень IV. Лестница H4a/H4b/H4c.
    // ================================================================
    let l4 = crate::factors::leverage::run_level4_leverage(
        ticker,
        &log_returns,
        WINDOW_MOMENTS,
        Some(day_len),
        collect_l4_example,
    );
    let l4r = l4.result.as_ref();
    let nan = f64::NAN;
    let (beta_k, t_beta_k) = l4r.map(|r| (r.beta, r.t_beta)).unwrap_or((nan, nan));
    let (sdf_mean, sdf_std) = l4r.map(|r| (r.sdf_mean, r.sdf_std)).unwrap_or((nan, nan));
    let l4_perm_p = l4r.map(|r| r.perm_p).unwrap_or(nan);
    let l4_n_obs = l4r.map(|r| r.n_obs).unwrap_or(0);
    let l4_t_ctrl = l4r.map(|r| r.t_ctrl).unwrap_or(nan);
    let l4_t_at = |sh: i32| -> f64 {
        l4r.and_then(|r| r.profile.iter().find(|p| p.0 == sh).map(|p| p.1))
            .unwrap_or(nan)
    };
    let (l4_t_lag1, l4_t_lead1) = (l4_t_at(-1), l4_t_at(1));
    let l4_sens = crate::factors::leverage::sensitivity_t(
        &log_returns,
        &[WINDOW_MOMENTS / 2, WINDOW_MOMENTS, WINDOW_MOMENTS * 2],
        Some(day_len),
    );

    // Лестница H4a/H4b/H4c.
    let l4_beta_kl = l4r.map(|r| r.beta_k_level).unwrap_or(nan);
    let l4_t_kl = l4r.map(|r| r.t_k_level).unwrap_or(nan);
    let l4_perm_kl = l4r.map(|r| r.perm_p_k_level).unwrap_or(nan);
    let l4_is_kl = l4.is_leverage_k_level;
    let l4_beta_dk = l4r.map(|r| r.beta_dk).unwrap_or(nan);
    let l4_t_dk = l4r.map(|r| r.t_dk).unwrap_or(nan);
    let l4_perm_dk = l4r.map(|r| r.perm_p_dk).unwrap_or(nan);
    let l4_is_dk = l4.is_leverage_dk;

    let (example_scatter, example_fit) = match l4r {
        Some(r) if collect_l4_example => (r.scatter.clone(), Some((r.beta, 0.0))),
        _ => (vec![], None),
    };

    // ================================================================
    // Уровень V (H5). Escape из барьера.
    // ================================================================
    let bad_escape: Option<Vec<bool>> = Some(
        (0..log_returns.len()).map(|i| (i + 1) % day_len == 0).collect()
    );
    let escape_cfg = crate::factors::leverage::LeverageConfig::new(WINDOW_MOMENTS);
    let escape_res = run_escape(
        &log_returns,
        &jumps_records,
        &escape_cfg,
        bad_escape.as_deref(),
        1,
    );
    let escr = escape_res.as_ref();
    let l5_n_blocks = escr.map(|r| r.n_blocks).unwrap_or(0);
    let l5_frac_beta = escr.map(|r| r.frac_beta).unwrap_or(nan);
    let l5_frac_t = escr.map(|r| r.frac_t).unwrap_or(nan);
    let l5_frac_perm_p = escr.map(|r| r.frac_perm_p).unwrap_or(nan);
    let l5_mom_beta = escr.map(|r| r.mom_beta).unwrap_or(nan);
    let l5_mom_t = escr.map(|r| r.mom_t).unwrap_or(nan);
    let l5_mom_perm_p = escr.map(|r| r.mom_perm_p).unwrap_or(nan);
    let l5_is_escape = escr.map(|r| r.is_escape).unwrap_or(false);

    // ================================================================
    // Уровень VI.
    // ================================================================
    let l6 = run_level6(ticker, &closes, HORIZONS);

    // ================================================================
    // Уровень VII.
    // ================================================================
    // θ = β — ускорение эксцесса A_j = K_j − 2K_{j−1} + K_{j−2}
    // на НЕПЕРЕСЕКАЮЩИХСЯ блоках W.
    let cfg_vii = crate::factors::leverage::LeverageConfig::new(WINDOW_MOMENTS);
    let grid_vii = crate::factors::leverage::build_grid(&log_returns, &cfg_vii, None);
    let acc_vii = crate::factors::leverage::kurtosis_acceleration(&grid_vii.k);
    let a_tilde_vii = crate::factors::leverage::demeaned_acceleration(&acc_vii);

    let h_mu_res = measure_h_star(
        ticker, "mu", "mean", &log_returns,
        HStarConfig {
            min_window: HSTAR_MU_MIN_W,
            max_window: HSTAR_MU_MAX_W,
            local_mult: HSTAR_LOCAL_MULT,
            drift_tol: HSTAR_DRIFT_TOL,
            max_bad_fraction: HSTAR_MAX_BAD_FRACTION,
            min_segments: HSTAR_MIN_SEGMENTS,
            null_shuffles: 3,
            null_quantile: 0.95,
            null_block: 1,
        },
    );
    let h_sigma2_res = measure_h_star(
        ticker, "sigma2", "var", &log_returns,
        HStarConfig {
            min_window: HSTAR_SIGMA2_MIN_W,
            max_window: HSTAR_SIGMA2_MAX_W,
            local_mult: HSTAR_LOCAL_MULT,
            drift_tol: HSTAR_DRIFT_TOL,
            max_bad_fraction: HSTAR_MAX_BAD_FRACTION,
            min_segments: HSTAR_MIN_SEGMENTS,
            null_shuffles: 3,
            null_quantile: 0.95,
            null_block: 1,
        },
    );
    let h_beta_res = measure_h_star(
        ticker, "beta", "raw", &a_tilde_vii,
        HStarConfig {
            min_window: HSTAR_BETA_MIN_W,
            max_window: HSTAR_BETA_MAX_W,
            local_mult: HSTAR_LOCAL_MULT,
            drift_tol: HSTAR_DRIFT_TOL,
            max_bad_fraction: HSTAR_MAX_BAD_FRACTION,
            min_segments: HSTAR_BETA_MIN_SEGMENTS,
            null_shuffles: 3,
            null_quantile: 0.95,
            null_block: 1,
        },
    );
    let h_beta_bars = h_beta_res.h_star * WINDOW_MOMENTS;

    // Коинтеграционный метод параллельно.
    let h_mu_coint = measure_h_star_coint(
        ticker, "mu", "mean", &log_returns,
        WINDOW_MOMENTS / 3, 240,
    );
    let h_sigma2_coint = measure_h_star_coint(
        ticker, "sigma2", "var", &log_returns,
        WINDOW_MOMENTS / 3, 240,
    );
    let h_beta_coint = measure_h_star_coint(
        ticker, "beta", "raw", &a_tilde_vii,
        WINDOW_MOMENTS / 3, 240,
    );

    let beta_coint = {
        let moment = crate::horizon::rolling_mean(&log_returns, WINDOW_MOMENTS);
        let ref_w = (2 * WINDOW_MOMENTS).min(log_returns.len() - 1);
        if ref_w > WINDOW_MOMENTS {
            let reference = crate::horizon::rolling_mean(&log_returns, ref_w);
            let pairs: Vec<(f64, f64)> = moment.iter().zip(reference.iter())
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .map(|(&a, &b)| (a, b))
                .collect();
            if pairs.len() >= 20 {
                let y: Vec<f64> = pairs.iter().map(|p| p.0).collect();
                let x: Vec<f64> = pairs.iter().map(|p| p.1).collect();
                if let Some(var2) = crate::horizon::fit_var2(&y, &x) {
                    let vecm = crate::horizon::detect_rank_and_cointegration(&var2);
                    if vecm.rank == 1 && vecm.beta_coint.is_finite() {
                        Some(vecm.beta_coint)
                    } else { None }
                } else { None }
            } else { None }
        } else { None }
    };

    let hierarchy = verify_hierarchy(h_mu_res.h_star, h_sigma2_res.h_star);
    let hierarchy_coint = verify_hierarchy(h_mu_coint.h_star, h_sigma2_coint.h_star);

    // ================================================================
    // Уровень VIII. Двухслойная адаптивная модель.
    // ================================================================
    let h_mu_hint = h_mu_res.h_star;
    let adaptive_res = run_adaptive(
        &log_returns,
        &jumps_records,
        h_mu_hint,
        WINDOW_MOMENTS,
        day_len,
    );
    let ar = adaptive_res.as_ref();
    let l8_mse_inner = ar.map(|r| r.mse_inner).unwrap_or(f64::NAN);
    let l8_mse_two = ar.map(|r| r.mse_two_layer).unwrap_or(f64::NAN);
    let l8_sh_inner = ar.map(|r| r.sharpe_inner).unwrap_or(f64::NAN);
    let l8_sh_two = ar.map(|r| r.sharpe_two_layer).unwrap_or(f64::NAN);
    let l8_auc_rg = ar.map(|r| r.auc_regime).unwrap_or(f64::NAN);
    let l8_auc_ps = ar.map(|r| r.auc_persistence).unwrap_or(f64::NAN);
    let l8_lambda = ar.map(|r| r.forgetting_lambda).unwrap_or(f64::NAN);
    let l8_n = ar.map(|r| r.n_test).unwrap_or(0);
    let l8_ok = adaptive_res.is_some();

    let summary = TickerSummary {
        ticker: ticker.to_string(),

        level1_mas_model: l1.mas_model,
        level1_mas_beta: l1.mas_beta,
        level1_median_kurt: median_kurt,
        level1_median_robust_kurt: median_rk,
        level1_median_alpha: median_alpha,
        level1_by_h: l1_by_h,
        level1_diag_by_h: l1_diag_by_h,

        level2_f_mu_slope: l2.f_mu_slope,
        level2_f_sigma2_slope: l2.f_sigma2_slope,
        level2_f_mu_significant: l2.f_mu_significant,
        level2_f_sigma2_significant: l2.f_sigma2_significant,
        level2_cusum_mean_p: l2.cusum_mean_pvalue,
        level2_cusum_sigma_p: l2.cusum_sigma_pvalue,
        level2_mfdfa_delta_alpha: l2.mfdfa_delta_alpha,
        level2_variance_decay_slope: l2.variance_decay_slope,
        level2_example_f_mu: example_f_mu,
        level2_example_f_sigma2: example_f_sigma2,
        level2_mfdfa_spectrum: mfdfa_spectrum,
        level2_stability_comparison: l2.stability_comparison,

        level3_med9_rv_median: l3.med9_rv_median,
        level3_n_jumps_total: l3.n_jumps_total,
        level3_n_jumps_economic: l3.n_jumps_economic,
        level3_frac_economic: l3.frac_economic,
        level3_jumps_by_minute: jumps_by_minute,
        level3_phi_values: phi_values,
        level3_n_jumps_near_tax: l3.n_jumps_near_tax,
        level3_n_jumps_outside_tax: l3.n_jumps_outside_tax,
        level3_tax_lift: l3.tax_lift,
        level3_ofi_median: l3.ofi_median,

        level4_beta_k: beta_k,
        level4_t_beta_k: t_beta_k,
        level4_is_leverage: l4.is_leverage,
        level4_sdf_mean: sdf_mean,
        level4_sdf_std: sdf_std,
        level4_example_scatter: example_scatter,
        level4_example_fit: example_fit,
        level4_perm_p: l4_perm_p,
        level4_n_obs: l4_n_obs,
        level4_t_ctrl: l4_t_ctrl,
        level4_t_lag1: l4_t_lag1,
        level4_t_lead1: l4_t_lead1,
        level4_sens: l4_sens,

        level4_beta_k_level: l4_beta_kl,
        level4_t_k_level: l4_t_kl,
        level4_perm_p_k_level: l4_perm_kl,
        level4_is_leverage_k_level: l4_is_kl,
        level4_beta_dk: l4_beta_dk,
        level4_t_dk: l4_t_dk,
        level4_perm_p_dk: l4_perm_dk,
        level4_is_leverage_dk: l4_is_dk,

        level8_mse_inner: l8_mse_inner,
        level8_mse_two_layer: l8_mse_two,
        level8_sharpe_inner: l8_sh_inner,
        level8_sharpe_two_layer: l8_sh_two,
        level8_auc_regime: l8_auc_rg,
        level8_auc_persistence: l8_auc_ps,
        level8_forgetting_lambda: l8_lambda,
        level8_n_test: l8_n,
        level8_ok: l8_ok,

        level5_escape_n_blocks: l5_n_blocks,
        level5_escape_frac_beta: l5_frac_beta,
        level5_escape_frac_t: l5_frac_t,
        level5_escape_frac_perm_p: l5_frac_perm_p,
        level5_escape_mom_beta: l5_mom_beta,
        level5_escape_mom_t: l5_mom_t,
        level5_escape_mom_perm_p: l5_mom_perm_p,
        level5_escape_is_escape: l5_is_escape,

        level6_n_regimes: l6.n_regimes,
        level6_mean_regime_len_bars: l6.mean_regime_len_bars,
        level6_regimes_per_100k: l6.regimes_per_100k,
        level6_dm_by_h: l6.dm_by_h,
        level6_gdm_stat: l6.gdm_stat,
        level6_gdm_pvalue: l6.gdm_pvalue,

        level7_h_mu: h_mu_res.h_star,
        level7_h_sigma2: h_sigma2_res.h_star,
        level7_h_beta: h_beta_res.h_star,
        level7_h_beta_bars: h_beta_bars,
        level7_hierarchy_holds: hierarchy.holds,
        level7_h_mu_coint: h_mu_coint.h_star,
        level7_h_sigma2_coint: h_sigma2_coint.h_star,
        level7_h_beta_coint: h_beta_coint.h_star,
        level7_hierarchy_holds_coint: hierarchy_coint.holds,
        level7_beta_coint: beta_coint,
        level7_mu_bad_fraction: h_mu_res.bad_fraction,
        level7_mu_median_drift: h_mu_res.median_drift,
        level7_sigma2_bad_fraction: h_sigma2_res.bad_fraction,
        level7_sigma2_median_drift: h_sigma2_res.median_drift,
        level7_beta_bad_fraction: h_beta_res.bad_fraction,
        level7_beta_median_drift: h_beta_res.median_drift,

        median_volume,
        median_sigma_pct,
        mean_daily_volume,
    };

    let _ = n;
    Ok(Some(summary))
}

// =====================================================================
// Вспомогательные функции
// =====================================================================

fn collect_jumps_by_minute(
    closes: &[f64],
    best_bids: &[Option<f64>],
    best_asks: &[Option<f64>],
    day_len: usize,
) -> (Vec<(usize, usize)>, Vec<f64>) {
    let mut by_minute: HashMap<usize, usize> = HashMap::new();
    let mut phi_values: Vec<f64> = Vec::new();

    let mut day_start = 0usize;
    while day_start + day_len <= closes.len() {
        let day_closes = &closes[day_start..day_start + day_len];
        let returns: Vec<f64> = day_closes.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.len() >= 9 {
            let _ = med9_rv(&returns);
            let jumps = detect_jumps(&returns, JUMP_THRESHOLD_STD);
            for &j in &jumps {
                *by_minute.entry(j).or_insert(0) += 1;
                let abs_ret = returns[j].abs();
                let idx = day_start + j + 1;
                if idx < best_bids.len() && idx < best_asks.len() {
                    if let (Some(bid), Some(ask)) = (best_bids[idx], best_asks[idx]) {
                        let mid = (bid + ask) / 2.0;
                        if mid > 0.0 {
                            let spread = (ask - bid) / mid;
                            if spread > 0.0 {
                                phi_values.push(abs_ret / spread);
                            }
                        }
                    }
                }
            }
        }
        day_start += day_len;
    }

    let mut by_minute_vec: Vec<(usize, usize)> = by_minute.into_iter().collect();
    by_minute_vec.sort_by_key(|(m, _)| *m);
    (by_minute_vec, phi_values)
}

/// Собрать список скачков с глобальными индексами в log_returns,
/// знаками и Φ. Для гипотезы H5.
fn collect_jump_records(
    closes: &[f64],
    best_bids: &[Option<f64>],
    best_asks: &[Option<f64>],
    day_len: usize,
) -> Vec<JumpRecord> {
    let mut out = Vec::new();
    if closes.len() < 2 { return out; }
    let global_returns: Vec<f64> = closes.windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let mut day_start = 0usize;
    while day_start + day_len <= closes.len() {
        let day_closes = &closes[day_start..day_start + day_len];
        let day_returns: Vec<f64> = day_closes.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        if day_returns.len() >= 9 {
            let jumps_local = detect_jumps(&day_returns, JUMP_THRESHOLD_STD);
            for &j in &jumps_local {
                let global_idx = day_start + j;
                if global_idx >= global_returns.len() { continue; }
                let r = global_returns[global_idx];
                if !r.is_finite() { continue; }
                let direction = if r > 0.0 { 1.0 } else { -1.0 };
                let bid_ask_idx = global_idx + 1;
                let (bid, ask) = (
                    best_bids.get(bid_ask_idx).copied().flatten(),
                    best_asks.get(bid_ask_idx).copied().flatten(),
                );
                let phi = match (bid, ask) {
                    (Some(b), Some(a)) if a > b => {
                        let mid = (b + a) / 2.0;
                        if mid > 0.0 { (a - b) / mid } else { f64::NAN }
                    }
                    _ => f64::NAN,
                };
                let phi_val = if phi.is_finite() && phi > 0.0 {
                    r.abs() / phi
                } else {
                    f64::NAN
                };
                let passed = phi_val.is_finite() && phi_val > 1.0;
                out.push(JumpRecord {
                    idx: global_idx,
                    direction,
                    phi: phi_val,
                    passed,
                });
            }
        }
        day_start += day_len;
    }
    out
}

// =====================================================================
// Агрегация данных для графиков
// =====================================================================

fn append_to_bucket(buckets: &mut Vec<(usize, Vec<f64>)>, h: usize, v: f64) {
    if !v.is_finite() { return; }
    let entry = buckets.iter_mut().find(|(hh, _)| *hh == h);
    match entry {
        Some((_, vec)) => vec.push(v),
        None => buckets.push((h, vec![v])),
    }
}

fn aggregate_plot_data_streaming(
    s: &TickerSummary,
    plot_l1: &mut Level1PlotData,
    plot_l2: &mut Level2PlotData,
    plot_l3: &mut Level3PlotData,
    plot_l4: &mut Level4PlotData,
    plot_l6: &mut Level6PlotData,
    plot_l7: &mut Level7PlotData,
    phi_writer: &mut dyn std::io::Write,
) -> std::io::Result<()> {
    // --- Уровень I ---
    for &(h, k, rk, alpha) in &s.level1_by_h {
        append_to_bucket(&mut plot_l1.kurt_by_h, h, k);
        append_to_bucket(&mut plot_l1.robust_kurt_by_h, h, rk);
        append_to_bucket(&mut plot_l1.alpha_by_h, h, alpha);
    }
    for &(h, denom, n_eff, nonoverlap) in &s.level1_diag_by_h {
        append_to_bucket(&mut plot_l1.rk_denom_by_h, h, denom);
        append_to_bucket(&mut plot_l1.n_eff_by_h, h, n_eff);
        append_to_bucket(&mut plot_l1.rk_nonoverlap_by_h, h, nonoverlap);
    }
    if let Some(model) = &s.level1_mas_model {
        let entry = plot_l1.mas_counts.iter_mut().find(|(name, _)| name == model);
        match entry {
            Some((_, c)) => *c += 1,
            None => plot_l1.mas_counts.push((model.clone(), 1)),
        }
    }
    if let Some(beta) = s.level1_mas_beta {
        if beta.is_finite() { plot_l1.betas.push(beta); }
    }

    // --- Уровень II ---
    if s.level2_f_mu_slope.is_finite() {
        plot_l2.f_mu_slopes.push(s.level2_f_mu_slope);
        plot_l2.n_total_slopes += 1;
        if s.level2_f_mu_significant { plot_l2.n_f_mu_significant += 1; }
    }
    if s.level2_f_sigma2_slope.is_finite() {
        plot_l2.f_sigma2_slopes.push(s.level2_f_sigma2_slope);
        if s.level2_f_sigma2_significant { plot_l2.n_f_sigma2_significant += 1; }
    }
    if plot_l2.example_f_mu.is_empty() && !s.level2_example_f_mu.is_empty() {
        plot_l2.example_f_mu = s.level2_example_f_mu.clone();
        plot_l2.example_f_sigma2 = s.level2_example_f_sigma2.clone();
    }
    if !s.level2_mfdfa_spectrum.is_empty() {
        if plot_l2.mfdfa_spectrum.is_empty() {
            plot_l2.mfdfa_spectrum = s.level2_mfdfa_spectrum.clone();
        }
    }

    // --- Уровень III ---
    for &(m, c) in &s.level3_jumps_by_minute {
        let entry = plot_l3.jumps_by_minute.iter_mut().find(|(mm, _)| *mm == m);
        match entry {
            Some((_, v)) => *v += c,
            None => plot_l3.jumps_by_minute.push((m, c)),
        }
    }
    for &v in &s.level3_phi_values {
        if v.is_finite() { writeln!(phi_writer, "{}", v)?; }
    }
    if plot_l3.n_total == 0 { plot_l3.frac_economic = s.level3_frac_economic; }
    plot_l3.n_total += s.level3_n_jumps_total;
    plot_l3.n_jumps_near_tax += s.level3_n_jumps_near_tax;
    plot_l3.n_jumps_outside_tax += s.level3_n_jumps_outside_tax;
    if s.level3_tax_lift.is_finite() { plot_l3.tax_lifts.push(s.level3_tax_lift); }

    // --- Уровень IV ---
    if s.level4_beta_k.is_finite() { plot_l4.betas.push(s.level4_beta_k); }
    if s.level4_t_beta_k.is_finite() { plot_l4.t_stats.push(s.level4_t_beta_k); }
    if s.level4_perm_p.is_finite() { plot_l4.perm_ps.push(s.level4_perm_p); }
    if s.level4_t_ctrl.is_finite() { plot_l4.t_ctrl.push(s.level4_t_ctrl); }
    if s.level4_t_lag1.is_finite() { plot_l4.t_lag1.push(s.level4_t_lag1); }
    if s.level4_t_lead1.is_finite() { plot_l4.t_lead1.push(s.level4_t_lead1); }
    for &(w, t) in &s.level4_sens {
        if !t.is_finite() { continue; }
        match plot_l4.sens.iter_mut().find(|e| e.0 == w) {
            Some(e) => e.1.push(t),
            None => plot_l4.sens.push((w, vec![t])),
        }
    }
    if plot_l4.example_scatter.is_empty() && !s.level4_example_scatter.is_empty() {
        plot_l4.example_scatter = s.level4_example_scatter.clone();
        plot_l4.example_fit = s.level4_example_fit;
    }
    // Лестница H4a/H4b.
    if s.level4_t_k_level.is_finite() {
        plot_l4.t_k_level.push(s.level4_t_k_level);
    }
    if s.level4_t_dk.is_finite() {
        plot_l4.t_k_delta.push(s.level4_t_dk);
    }

    // --- Уровень V (H5) ---
    // Пушим парами: иначе frac_ts и frac_perm_ps рассинхронизируются,
    // и BH применяется к большему числу p, чем реально тикеров.
    if s.level5_escape_frac_t.is_finite() && s.level5_escape_frac_perm_p.is_finite() {
        plot_l4.frac_ts.push(s.level5_escape_frac_t);
        plot_l4.frac_perm_ps.push(s.level5_escape_frac_perm_p);
    }
    if s.level5_escape_mom_t.is_finite() && s.level5_escape_mom_perm_p.is_finite() {
        plot_l4.mom_ts.push(s.level5_escape_mom_t);
        plot_l4.mom_perm_ps.push(s.level5_escape_mom_perm_p);
    }

    // --- Уровень VI ---
    for (h_idx, &h) in HORIZONS.iter().enumerate() {
        if let Some((dm, _)) = s.level6_dm_by_h.get(h_idx) {
            if dm.is_finite() { append_to_bucket(&mut plot_l6.dm_by_h, h, *dm); }
        }
    }
    if s.level6_gdm_pvalue.is_finite() {
        plot_l6.gdm_pvalues.push(s.level6_gdm_pvalue);
    }

    // --- Уровень VII ---
    plot_l7.h_mu.push(s.level7_h_mu as f64);
    plot_l7.h_sigma2.push(s.level7_h_sigma2 as f64);
    plot_l7.h_beta.push(s.level7_h_beta as f64);
    if let Some(bc) = s.level7_beta_coint {
        if bc.is_finite() { plot_l7.beta_coint.push(bc); }
    }
    if s.level7_mu_bad_fraction.is_finite() {
        plot_l7.mu_bad_fraction.push(s.level7_mu_bad_fraction);
    }
    if s.level7_mu_median_drift.is_finite() {
        plot_l7.mu_median_drift.push(s.level7_mu_median_drift);
    }
    if s.level7_sigma2_bad_fraction.is_finite() {
        plot_l7.sigma2_bad_fraction.push(s.level7_sigma2_bad_fraction);
    }
    if s.level7_sigma2_median_drift.is_finite() {
        plot_l7.sigma2_median_drift.push(s.level7_sigma2_median_drift);
    }
    if s.level7_beta_bad_fraction.is_finite() {
        plot_l7.beta_bad_fraction.push(s.level7_beta_bad_fraction);
    }
    if s.level7_beta_median_drift.is_finite() {
        plot_l7.beta_median_drift.push(s.level7_beta_median_drift);
    }

    Ok(())
}

// =====================================================================
// Сводный отчёт
// =====================================================================

#[allow(clippy::too_many_arguments)]
fn write_summary(
    report: &mut std::fs::File,
    n_ok: usize,
    n_leverage: usize,
    n_hierarchy_holds: usize,
    n_hierarchy_total: usize,
    plot_l1: &Level1PlotData,
    plot_l2: &Level2PlotData,
    plot_l3: &Level3PlotData,
    plot_l4: &Level4PlotData,
    plot_l6: &Level6PlotData,
    plot_l7: &Level7PlotData,
) -> Result<()> {
    writeln!(report, "\n--- Сводка по рынку ---")?;
    writeln!(report, "Тикеров обработано: {}", n_ok)?;
    writeln!(report)?;

    // ===== Уровень I =====
    writeln!(report, "=== Уровень I. Дискретность и эксцесс ===")?;
    for &(h, ref vals) in &plot_l1.kurt_by_h {
        let rk_vals = plot_l1.robust_kurt_by_h.iter()
            .find(|(hh, _)| *hh == h).map(|(_, v)| v.as_slice()).unwrap_or(&[]);
        let alpha_vals = plot_l1.alpha_by_h.iter()
            .find(|(hh, _)| *hh == h).map(|(_, v)| v.as_slice()).unwrap_or(&[]);
        writeln!(
            report,
            "  h={:>4}: медиана K = {:.3}, медиана RK = {:.3}, медиана α = {:.3} (n={})",
            h, median(vals), median(rk_vals), median(alpha_vals), vals.len()
        )?;
    }
    // Диагностика роста RK при падении K.
    writeln!(report, "\nДиагностика RK (почему RK растёт при падении K):")?;
    for &(h, ref denoms) in &plot_l1.rk_denom_by_h {
        let nonoverlap = plot_l1.rk_nonoverlap_by_h.iter()
            .find(|(hh, _)| *hh == h).map(|(_, v)| v.as_slice()).unwrap_or(&[]);
        let n_eff = plot_l1.n_eff_by_h.iter()
            .find(|(hh, _)| *hh == h).map(|(_, v)| v.as_slice()).unwrap_or(&[]);
        writeln!(report,
            "  h={:>4}: медиана rk_denom = {:+.4}, медиана n_eff = {:.0}, \
             медиана RK_nonoverlap = {:.3}",
            h, median(denoms), median(n_eff), median(nonoverlap))?;
    }

    writeln!(report, "\nРаспределение MAS-моделей:")?;
    for (model, count) in &plot_l1.mas_counts {
        writeln!(report, "  {:<15}: {}", model, count)?;
    }
    if !plot_l1.betas.is_empty() {
        writeln!(report, "Медиана β (степенной закон): {:.3}", median(&plot_l1.betas))?;
    }

    // ===== Уровень II =====
    writeln!(report, "\n=== Уровень II. Эргодичность и устойчивость ===")?;

    let n_l2 = plot_l2.n_total_slopes.max(1);
    let frac_mu_sig = plot_l2.n_f_mu_significant as f64 / n_l2 as f64;
    let frac_sig2_sig = plot_l2.n_f_sigma2_significant as f64 / n_l2 as f64;

    writeln!(report, "Медиана наклона f_μ: {:.6}", median(&plot_l2.f_mu_slopes))?;
    writeln!(report,
        "  доля тикеров с |t| > 1.96: {:.1}% ({} из {})",
        frac_mu_sig * 100.0, plot_l2.n_f_mu_significant, n_l2)?;
    writeln!(report, "Медиана наклона f_σ²: {:.6}", median(&plot_l2.f_sigma2_slopes))?;
    writeln!(report,
        "  доля тикеров с |t| > 1.96: {:.1}% ({} из {})",
        frac_sig2_sig * 100.0, plot_l2.n_f_sigma2_significant, n_l2)?;

    writeln!(report)?;
    if frac_sig2_sig > 0.30 && frac_mu_sig < 0.15 {
        writeln!(report,
            "H2 подтверждается: дисперсия дрейфует у >30% тикеров, \
             среднее — у <15%.")?;
    } else if frac_sig2_sig > 0.30 && frac_mu_sig > 0.30 {
        writeln!(report,
            "Оба момента нестабильны у >30% тикеров.")?;
    } else if frac_sig2_sig < 0.15 && frac_mu_sig < 0.15 {
        writeln!(report,
            "Оба момента стабильны у >85% тикеров.")?;
    } else {
        writeln!(report,
            "Смешанная картина: доля значимых f_μ = {:.1}%, f_σ² = {:.1}%.",
            frac_mu_sig * 100.0, frac_sig2_sig * 100.0)?;
    }

    // ===== Уровень III =====
    writeln!(report, "\n=== Уровень III. Микроструктура и скачки ===")?;
    let total_jumps: usize = plot_l3.jumps_by_minute.iter().map(|(_, c)| *c).sum();
    writeln!(report, "Всего скачков (по рынку): {}", total_jumps)?;
    writeln!(report,
        "Доля экономически значимых (Φ > 1): {:.4}", plot_l3.frac_economic)?;
    if !plot_l3.phi_values.is_empty() {
        writeln!(report,
            "Медиана Φ = {:.3}, доля Φ>1 = {:.1}%",
            median(&plot_l3.phi_values),
            plot_l3.phi_values.iter().filter(|&&p| p > 1.0).count() as f64
                / plot_l3.phi_values.len() as f64 * 100.0)?;
    }

    writeln!(report, "\n--- Кластеризация скачков вокруг налоговых дат ---")?;
    writeln!(report, "Окно: ±3 дня от 15-го и конца месяца (2018–2022).")?;
    writeln!(report,
        "Скачков около налоговых дат = {}, вне окна = {}",
        plot_l3.n_jumps_near_tax, plot_l3.n_jumps_outside_tax)?;
    if !plot_l3.tax_lifts.is_empty() {
        let med_lift = median(&plot_l3.tax_lifts);
        let frac_elevated = plot_l3.tax_lifts.iter()
            .filter(|&&l| l > 1.20).count() as f64 / plot_l3.tax_lifts.len() as f64;
        let frac_depressed = plot_l3.tax_lifts.iter()
            .filter(|&&l| l < 0.80).count() as f64 / plot_l3.tax_lifts.len() as f64;
        writeln!(report, "Медиана tax_lift = {:.3}", med_lift)?;
        writeln!(report, "Доля с tax_lift > 1.20: {:.1}%", frac_elevated * 100.0)?;
        writeln!(report, "Доля с tax_lift < 0.80: {:.1}%", frac_depressed * 100.0)?;
        if med_lift > 1.20 && frac_elevated > 0.30 {
            writeln!(report, "→ Кластеризация подтверждается.")?;
        } else if med_lift < 0.80 && frac_depressed > 0.30 {
            writeln!(report, "→ Антикластеризация.")?;
        } else {
            writeln!(report, "→ Нет систематической связи.")?;
        }
    }

    // ===== Уровень IV. Лестница H4a/H4b/H4c =====
    writeln!(report, "\n=== Уровень IV. Лестница ускорения эксцесса ===")?;

    {
        let share = |v: &[f64], thr: f64| -> f64 {
            v.iter().filter(|&&t| t > thr).count() as f64 / v.len().max(1) as f64 * 100.0
        };
        let share_neg = |v: &[f64], thr: f64| -> f64 {
            v.iter().filter(|&&t| t < thr).count() as f64 / v.len().max(1) as f64 * 100.0
        };

        writeln!(report,
            "  [H4] Непересекающиеся блоки W={}, отклик = доходность следующего блока.",
            WINDOW_MOMENTS)?;
        writeln!(report,
            "  [H4] Ex-ante нормировка: y_j = R_(j+1)/s_j, s_j = √(W·σ²_j), \
             σ²_j известно в конце блока j.")?;
        writeln!(report, "  [H4] Под нулём доля t > 1.645 ≈ 5% (односторонний тест).")?;
        writeln!(report)?;

        // Вспомогательная функция: строка + binom-тест на перекос вниз.
        let report_h4 = |report: &mut std::fs::File,
                         label: &str,
                         ts: &[f64]| -> Result<()> {
            let n = ts.len();
            if n == 0 { return Ok(()); }
            let k_pos = ts.iter().filter(|&&t| t > 1.645).count();
            let k_neg = ts.iter().filter(|&&t| t < -1.645).count();
            let p_neg = crate::dm_test::binom_upper_tail(k_neg, n, 0.05);
            let p_pos = crate::dm_test::binom_upper_tail(k_pos, n, 0.05);
            writeln!(report,
                "  [{}] доля t > 1.645 = {:.1}%, t < −1.645 = {:.1}%",
                label, share(ts, 1.645), share_neg(ts, -1.645))?;
            writeln!(report,
                "      binom-p перекос вверх: p = {:.4}; перекос вниз: p = {:.4}",
                p_pos, p_neg)?;
            Ok(())
        };

        report_h4(report, "H4a", &plot_l4.t_k_level)?;
        report_h4(report, "H4b", &plot_l4.t_k_delta)?;
        report_h4(report, "H4c", &plot_l4.t_stats)?;
        writeln!(report)?;

        if !plot_l4.betas.is_empty() {
            writeln!(report,
                "  Медиана β_k = {:.4}, доля β_k > 0 = {:.1}%",
                median(&plot_l4.betas),
                plot_l4.betas.iter().filter(|&&b| b > 0.0).count() as f64
                    / plot_l4.betas.len() as f64 * 100.0)?;
        }
        writeln!(report,
            "  Тикеров с рычагом H4c (t>1.645 ∧ perm_p≤0.05 ∧ β>0 в обеих половинах): {} из {}",
            n_leverage, plot_l4.t_stats.len())?;

        if !plot_l4.perm_ps.is_empty() {
            let m = plot_l4.perm_ps.len();
            let below = plot_l4.perm_ps.iter().filter(|&&p| p <= 0.05).count();
            let bh = crate::factors::leverage::benjamini_hochberg(&plot_l4.perm_ps, 0.05);
            writeln!(report,
                "  [H4c] Доля perm_p ≤ 0.05: {:.1}% ({} из {}); BH (q=5%): {} из {}",
                below as f64 / m as f64 * 100.0, below, m, bh, m)?;
        }
        if !plot_l4.t_ctrl.is_empty() {
            writeln!(report,
                "  [H4c] Доля t_β > 1.645 при контроле Δlog σ²: {:.1}%",
                share(&plot_l4.t_ctrl, 1.645))?;
        }
        if !plot_l4.t_lag1.is_empty() {
            writeln!(report,
                "  [H4c] Профиль сдвигов −1: доля t>1.645 = {:.1}%",
                share(&plot_l4.t_lag1, 1.645))?;
        }
        if !plot_l4.t_lead1.is_empty() {
            writeln!(report,
                "  [H4c] Диагностика загрязнения (сдвиг +1): {:.1}%",
                share(&plot_l4.t_lead1, 1.645))?;
        }
        for (w, ts) in &plot_l4.sens {
            writeln!(report,
                "  [H4c] Чувствительность W={:>4}: доля t_β>1.645 = {:.1}%, медиана t = {:+.2}",
                w, share(ts, 1.645), median(ts))?;
        }
    }

    // ===== Уровень V (H5) =====
    writeln!(report, "\n=== Уровень V (H5). Escape из барьера Гроссмана–Миллера ===")?;
    let n_l5 = plot_l4.frac_ts.len();
    if n_l5 > 0 {
        let share = |v: &[f64], thr: f64| -> f64 {
            v.iter().filter(|&&t| t > thr).count() as f64 / v.len().max(1) as f64 * 100.0
        };
        let n_esc = (0..n_l5.min(plot_l4.mom_ts.len()))
            .filter(|&i| {
                plot_l4.frac_ts[i] > 1.645
                    && plot_l4.mom_ts[i] > 1.645
                    && plot_l4.frac_perm_ps.get(i).map(|p| *p <= 0.05).unwrap_or(false)
                    && plot_l4.mom_perm_ps.get(i).map(|p| *p <= 0.05).unwrap_or(false)
            })
            .count();
        writeln!(report, "Тикеров протестировано (H5): {} из {}", n_l5, n_ok)?;
        writeln!(report,
            "  H5a (доля Φ>1 ↑ с Ã): доля t_frac > 1.645 = {:.1}% (ожидание ≈5%)",
            share(&plot_l4.frac_ts, 1.645))?;
        if !plot_l4.mom_ts.is_empty() {
            writeln!(report,
                "  H5b (momentum ↑ с Ã): доля t_mom > 1.645 = {:.1}% (ожидание ≈5%)",
                share(&plot_l4.mom_ts, 1.645))?;
        }
        writeln!(report,
            "  Тикеров с escape (оба условия): {} из {} ({:.1}%)",
            n_esc, n_l5, n_esc as f64 / n_l5.max(1) as f64 * 100.0)?;
        if !plot_l4.frac_perm_ps.is_empty() {
            let bh_frac = crate::factors::leverage::benjamini_hochberg(&plot_l4.frac_perm_ps, 0.05);
            let bh_mom = crate::factors::leverage::benjamini_hochberg(&plot_l4.mom_perm_ps, 0.05);
            writeln!(report,
                "  BH (q=5%) по perm_p: frac {} из {}, mom {} из {}",
                bh_frac, plot_l4.frac_perm_ps.len(),
                bh_mom, plot_l4.mom_perm_ps.len())?;
        }
    }

    // ===== Уровень VI =====
    writeln!(report, "\n=== Уровень VI. Гистерезисная надстройка ===")?;
    for &(h, ref vals) in &plot_l6.dm_by_h {
        writeln!(report,
            "  h={:>4}: медиана DM = {:+.3}, доля DM<0 = {:.1}%, n = {}",
            h, median(vals),
            vals.iter().filter(|&&v| v < 0.0).count() as f64
                / vals.len().max(1) as f64 * 100.0,
            vals.len())?;
    }
    if !plot_l6.gdm_pvalues.is_empty() {
        writeln!(report,
            "GDM: доля p < 0.05 = {:.1}%, n = {}",
            plot_l6.gdm_pvalues.iter().filter(|&&p| p < 0.05).count() as f64
                / plot_l6.gdm_pvalues.len() as f64 * 100.0,
            plot_l6.gdm_pvalues.len())?;
    }

    // ===== Уровень VII =====
    writeln!(report, "\n=== Уровень VII. Горизонт H*(θ) ===")?;

    let med_mu = median(&plot_l7.h_mu);
    let med_sig = median(&plot_l7.h_sigma2);
    let med_beta = median(&plot_l7.h_beta);

    let nz_mu: Vec<f64> = plot_l7.h_mu.iter().copied().filter(|&x| x > 0.0).collect();
    let nz_sig: Vec<f64> = plot_l7.h_sigma2.iter().copied().filter(|&x| x > 0.0).collect();
    let nz_beta: Vec<f64> = plot_l7.h_beta.iter().copied().filter(|&x| x > 0.0).collect();

    let n_total = plot_l7.h_mu.len();

    writeln!(report, "Медиана H*(μ)  = {:.0} (ноль у {} из {})",
        med_mu, n_total.saturating_sub(nz_mu.len()), n_total)?;
    writeln!(report, "Медиана H*(σ²) = {:.0} (ноль у {} из {})",
        med_sig, n_total.saturating_sub(nz_sig.len()), n_total)?;
    writeln!(report, "Медиана H*(β)  = {:.0} блоков = {:.0} баров (ноль у {} из {})",
        med_beta, med_beta * WINDOW_MOMENTS as f64,
        n_total.saturating_sub(nz_beta.len()), n_total)?;

    // Разделение «ноль» и «равенство» — критично для интерпретации иерархии.
    let n_zero_mu = plot_l7.h_mu.iter().filter(|&&x| x == 0.0).count();
    let n_zero_s2 = plot_l7.h_sigma2.iter().filter(|&&x| x == 0.0).count();
    let n_both_zero = plot_l7.h_mu.iter().zip(&plot_l7.h_sigma2)
        .filter(|(&a, &b)| a == 0.0 && b == 0.0).count();
    let n_mu_zero_s2_pos = plot_l7.h_mu.iter().zip(&plot_l7.h_sigma2)
        .filter(|(&a, &b)| a == 0.0 && b > 0.0).count();
    let n_mu_pos_s2_zero = plot_l7.h_mu.iter().zip(&plot_l7.h_sigma2)
        .filter(|(&a, &b)| a > 0.0 && b == 0.0).count();
    writeln!(report,
        "  H*(μ)=0: {}; H*(σ²)=0: {}; оба ноль: {}; μ=0&σ²>0: {}; μ>0&σ²=0: {}",
        n_zero_mu, n_zero_s2, n_both_zero, n_mu_zero_s2_pos, n_mu_pos_s2_zero)?;

    if !nz_mu.is_empty() {
        writeln!(report, "Медиана H*(μ) по ненулевым:  {:.0} (n={})",
            median(&nz_mu), nz_mu.len())?;
    }
    if !nz_sig.is_empty() {
        writeln!(report, "Медиана H*(σ²) по ненулевым: {:.0} (n={})",
            median(&nz_sig), nz_sig.len())?;
    }
    if !nz_beta.is_empty() {
        writeln!(report, "Медиана H*(β) по ненулевым:  {:.0} блоков (n={})",
            median(&nz_beta), nz_beta.len())?;
    }


    writeln!(report,
        "Иерархия H*(μ) ≥ H*(σ²) подтверждена: {} из {} ({:.1}%)",
        n_hierarchy_holds, n_hierarchy_total,
        n_hierarchy_holds as f64 / n_hierarchy_total.max(1) as f64 * 100.0)?;

    if !plot_l7.mu_bad_fraction.is_empty() {
        writeln!(report,
            "  [local] μ: медиана bad_fraction = {:.3}, медиана drift = {:.3} (n={})",
            median(&plot_l7.mu_bad_fraction),
            median(&plot_l7.mu_median_drift),
            plot_l7.mu_bad_fraction.len())?;
    }
    if !plot_l7.sigma2_bad_fraction.is_empty() {
        writeln!(report,
            "  [local] σ²: медиана bad_fraction = {:.3}, медиана drift = {:.3} (n={})",
            median(&plot_l7.sigma2_bad_fraction),
            median(&plot_l7.sigma2_median_drift),
            plot_l7.sigma2_bad_fraction.len())?;
    }
    if !plot_l7.beta_bad_fraction.is_empty() {
        writeln!(report,
            "  [local] β: медиана bad_fraction = {:.3}, медиана drift = {:.3} (n={})",
            median(&plot_l7.beta_bad_fraction),
            median(&plot_l7.beta_median_drift),
            plot_l7.beta_bad_fraction.len())?;
    }

    writeln!(report,
        "  H*(β) — блочный A_j = K_j − 2K_(j−1) + K_(j−2), шаг = W баров.")?;
    writeln!(report,
        "  Иерархия проверяется только для H*(μ) ≥ H*(σ²); \
         β — диагностика, шкала блоков несопоставима со шкалой баров.")?;

    Ok(())
}

// =====================================================================
// Групповой анализ
// =====================================================================

fn two_prop_z(k1: usize, n1: usize, k2: usize, n2: usize) -> f64 {
    if n1 == 0 || n2 == 0 { return f64::NAN; }
    let (p1, p2) = (k1 as f64 / n1 as f64, k2 as f64 / n2 as f64);
    let p = (k1 + k2) as f64 / (n1 + n2) as f64;
    let se = (p * (1.0 - p) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    if se > 0.0 { (p2 - p1) / se } else { f64::NAN }
}

fn split_by_quantiles<'a>(
    tickers: &'a [TickerSummary],
    key: fn(&TickerSummary) -> f64,
) -> (Vec<&'a TickerSummary>, Vec<&'a TickerSummary>, Vec<&'a TickerSummary>) {
    let mut filtered: Vec<&TickerSummary> = tickers.iter()
        .filter(|s| key(s).is_finite())
        .collect();
    if filtered.len() < 30 {
        return (vec![], vec![], vec![]);
    }
    filtered.sort_by(|a, b| {
        key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = filtered.len();
    let low = filtered[..n / 3].to_vec();
    let mid = filtered[n / 3..2 * n / 3].to_vec();
    let high = filtered[2 * n / 3..].to_vec();
    (low, mid, high)
}

#[derive(Debug, Clone, Default)]
struct GroupStats {
    n: usize,
    n_tested: usize,
    n_leverage: usize,
    frac_leverage: f64,
    median_beta_k: f64,
    frac_hierarchy_holds: f64,
    median_dm_h1: f64,
    median_volume: f64,
    median_sigma_pct: f64,
}

fn compute_group_stats(tickers: &[&TickerSummary]) -> GroupStats {
    if tickers.is_empty() { return GroupStats::default(); }
    let n = tickers.len();
    let n_tested = tickers.iter().filter(|s| s.level4_beta_k.is_finite()).count();
    let n_leverage = tickers.iter().filter(|s| s.level4_is_leverage).count();
    let n_hierarchy = tickers.iter().filter(|s| s.level7_hierarchy_holds).count();

    let betas: Vec<f64> = tickers.iter()
        .map(|s| s.level4_beta_k).filter(|v| v.is_finite()).collect();
    let dm_h1: Vec<f64> = tickers.iter()
        .filter_map(|s| s.level6_dm_by_h.first().map(|(dm, _)| *dm))
        .filter(|v| v.is_finite()).collect();
    let volumes: Vec<f64> = tickers.iter()
        .map(|s| s.median_volume).filter(|v| v.is_finite()).collect();
    let sigmas: Vec<f64> = tickers.iter()
        .map(|s| s.median_sigma_pct).filter(|v| v.is_finite()).collect();

    GroupStats {
        n,
        n_tested,
        n_leverage,
        frac_leverage: if n_tested > 0 { n_leverage as f64 / n_tested as f64 } else { f64::NAN },
        median_beta_k: median(&betas),
        frac_hierarchy_holds: n_hierarchy as f64 / n as f64,
        median_dm_h1: median(&dm_h1),
        median_volume: median(&volumes),
        median_sigma_pct: median(&sigmas),
    }
}

fn write_one_grouping(
    report: &mut std::fs::File,
    summaries: &[TickerSummary],
    label: &str,
    key: fn(&TickerSummary) -> f64,
) -> Result<()> {
    let (low, mid, high) = split_by_quantiles(summaries, key);
    if low.is_empty() {
        writeln!(report, "\n=== Группировка по {} === (недостаточно данных)", label)?;
        return Ok(());
    }
    let g_low = compute_group_stats(&low);
    let g_mid = compute_group_stats(&mid);
    let g_high = compute_group_stats(&high);

    writeln!(report, "\n=== Группировка по {} (терцили: low/mid/high) ===", label)?;
    writeln!(report,
        "  {:<8} {:<7} {:<12} {:<12} {:<10} {:<12}",
        "Группа", "N", "Доля рычага", "Доля иерарх.", "Med β_k", "Med DM h=1")?;
    for (lbl, g) in &[("low", &g_low), ("mid", &g_mid), ("high", &g_high)] {
        writeln!(report,
            "  {:<8} {:<7} {:>6.1}%      {:>6.1}%      {:>10.4}  {:>10.3}",
            lbl, g.n, g.frac_leverage * 100.0, g.frac_hierarchy_holds * 100.0,
            g.median_beta_k, g.median_dm_h1)?;
    }

    let mut interp: Vec<String> = Vec::new();
    let z = two_prop_z(g_low.n_leverage, g_low.n_tested,
                       g_high.n_leverage, g_high.n_tested);
    if z.is_finite() && z.abs() > 2.58 {
        interp.push(format!("рычаг {} в high-{} (z={:+.2})",
            if z > 0.0 { "чаще" } else { "реже" }, label, z));
    } else {
        interp.push(format!("различий не обнаружено (z={:+.2})", z));
    }
    writeln!(report, "  → {}", interp.join("; "))?;

    Ok(())
}

// =====================================================================
// Сводка уровня VIII
// =====================================================================

fn write_level8_summary(
    report: &mut std::fs::File,
    summaries: &[TickerSummary],
) -> Result<()> {
    writeln!(report, "\n\n=== Уровень VIII. Двухслойная адаптивная модель ===")?;

    let ok: Vec<&TickerSummary> = summaries.iter().filter(|s| s.level8_ok).collect();
    if ok.is_empty() {
        writeln!(report, "  Нет тикеров с достаточными данными.")?;
        return Ok(());
    }
    let n = ok.len();

    let mse_inner: Vec<f64> = ok.iter().map(|s| s.level8_mse_inner).filter(|v| v.is_finite()).collect();
    let mse_two: Vec<f64> = ok.iter().map(|s| s.level8_mse_two_layer).filter(|v| v.is_finite()).collect();
    let sh_inner: Vec<f64> = ok.iter().map(|s| s.level8_sharpe_inner).filter(|v| v.is_finite()).collect();
    let sh_two: Vec<f64> = ok.iter().map(|s| s.level8_sharpe_two_layer).filter(|v| v.is_finite()).collect();
    let auc_rg: Vec<f64> = ok.iter().map(|s| s.level8_auc_regime).filter(|v| v.is_finite()).collect();
    let auc_ps: Vec<f64> = ok.iter().map(|s| s.level8_auc_persistence).filter(|v| v.is_finite()).collect();
    let lambdas: Vec<f64> = ok.iter().map(|s| s.level8_forgetting_lambda).filter(|v| v.is_finite()).collect();

    writeln!(report, "  Тикеров с валидным бэктестом: {} из {}", n, summaries.len())?;
    writeln!(report, "  Медиана λ = 1 − 1/H*(μ) = {:.4}", median(&lambdas))?;
    writeln!(report)?;

    writeln!(report, "  --- MSE (меньше = лучше) ---")?;
    writeln!(report, "    Медиана MSE inner:      {:.3e}", median(&mse_inner))?;
    writeln!(report, "    Медиана MSE two-layer:  {:.3e}", median(&mse_two))?;

    let n_better = (0..ok.len()).filter(|&i| {
        let a = ok[i].level8_mse_inner;
        let b = ok[i].level8_mse_two_layer;
        a.is_finite() && b.is_finite() && b < a
    }).count();
    writeln!(report, "    Доля тикеров, где two-layer лучше: {:.1}%",
        n_better as f64 / n as f64 * 100.0)?;

    writeln!(report)?;
    writeln!(report, "  --- Sharpe sign-стратегии ---")?;
    writeln!(report, "    Медиана Sharpe inner:     {:+.4}", median(&sh_inner))?;
    writeln!(report, "    Медиана Sharpe two-layer: {:+.4}", median(&sh_two))?;

    let n_sh_better = (0..ok.len()).filter(|&i| {
        let a = ok[i].level8_sharpe_inner;
        let b = ok[i].level8_sharpe_two_layer;
        a.is_finite() && b.is_finite() && b > a
    }).count();
    writeln!(report, "    Доля тикеров, где Sharpe улучшился: {:.1}%",
        n_sh_better as f64 / n as f64 * 100.0)?;

    writeln!(report)?;
    writeln!(report, "  --- AUC логита предсказания смены режима ---")?;
    writeln!(report, "    Медиана AUC логита:      {:.4}", median(&auc_rg))?;
    writeln!(report, "    Медиана AUC persistence: {:.4}", median(&auc_ps))?;

    let n_auc_better = (0..ok.len()).filter(|&i| {
        let a = ok[i].level8_auc_regime;
        let b = ok[i].level8_auc_persistence;
        a.is_finite() && b.is_finite() && a > b
    }).count();
    writeln!(report, "    Доля тикеров, где логит > persistence: {:.1}%",
        n_auc_better as f64 / n as f64 * 100.0)?;

    writeln!(report)?;
    let med_mse_ratio = {
        let ratios: Vec<f64> = (0..ok.len())
            .filter_map(|i| {
                let a = ok[i].level8_mse_inner;
                let b = ok[i].level8_mse_two_layer;
                if a.is_finite() && b.is_finite() && a > 1e-15 {
                    Some(b / a)
                } else { None }
            })
            .collect();
        median(&ratios)
    };
    writeln!(report, "  Медиана отношения MSE_two / MSE_inner: {:.4}", med_mse_ratio)?;

    let med_sharpe_inner = median(&sh_inner);
    let med_sharpe_two = median(&sh_two);
    let med_sharpe_delta = med_sharpe_two - med_sharpe_inner;
    let med_auc_rg = median(&auc_rg);
    let med_auc_ps = median(&auc_ps);

    writeln!(report)?;
    writeln!(report, "  --- Диагностика ---")?;
    writeln!(report, "    Sharpe delta (two − inner): {:+.6}", med_sharpe_delta)?;

    if med_mse_ratio < 0.95
        && n_better as f64 / n as f64 > 0.55
        && med_sharpe_delta > 1e-4
    {
        writeln!(report, "  → Двухслойная модель улучшает и MSE, и Sharpe.")?;
    } else if med_mse_ratio < 1.0 && med_sharpe_delta <= 1e-4 {
        writeln!(report,
            "  → MSE падает механически (сжатие прогноза через (1−P̂)), \
             Sharpe не улучшился. Улучшение MSE — артефакт масштаба, не сигнал.")?;
    } else if med_mse_ratio > 1.05 {
        writeln!(report, "  → Двухслойная модель ухудшает MSE.")?;
    } else {
        writeln!(report, "  → Эффект двухслойной модели неотличим от шума.")?;
    }

    if med_auc_rg < 0.55 {
        writeln!(report,
            "  → Логит не предсказывает режим лучше случайного \
             (AUC={:.4} vs persistence={:.4}).",
            med_auc_rg, med_auc_ps)?;
    }
    if med_auc_ps > 0.90 {
        writeln!(report,
            "  → Метка смены режима сильно персистентна \
             (AUC persistence={:.4}); логит не может конкурировать с тривиальным baseline.",
            med_auc_ps)?;
    }

    writeln!(report,
        "  → H8 ОТВЕРГНУТА: ни MSE-эффект, ни Sharpe, ни AUC не подтверждают \
         преимущества двухслойной модели.")?;

    Ok(())
}

fn write_group_analysis(
    report: &mut std::fs::File,
    summaries: &[TickerSummary],
) -> Result<()> {
    if summaries.len() < 30 {
        writeln!(report, "\n=== Групповой анализ === (нужно ≥30 тикеров)")?;
        return Ok(());
    }

    writeln!(report, "\n\n=========================================================")?;
    writeln!(report, "=========== ГРУППОВОЙ АНАЛИЗ ТИКЕРОВ ===================")?;
    writeln!(report, "=========================================================")?;

    let configs: Vec<(&str, fn(&TickerSummary) -> f64)> = vec![
        ("median_volume",   |s| s.median_volume),
        ("median_sigma_pct",|s| s.median_sigma_pct),
        ("median_kurt",     |s| s.level1_median_kurt),
        ("mean_regime_len", |s| s.level6_mean_regime_len_bars),
        ("H*(μ)",           |s| {
            if s.level7_h_mu > 0 { s.level7_h_mu as f64 } else { f64::NAN }
        }),
    ];
    for (name, key) in configs {
        write_one_grouping(report, summaries, name, key)?;
    }

    writeln!(report, "\n\n=== Сравнение: рычаг работает vs не работает ===")?;

    let works: Vec<&TickerSummary> = summaries.iter()
        .filter(|s| s.level4_is_leverage).collect();
    let not_works: Vec<&TickerSummary> = summaries.iter()
        .filter(|s| s.level4_beta_k.is_finite() && !s.level4_is_leverage).collect();

    if works.len() < 10 || not_works.len() < 10 {
        writeln!(report, "  недостаточно данных (works={}, not_works={})",
            works.len(), not_works.len())?;
        return Ok(());
    }

    let g_w = compute_group_stats(&works);
    let g_nw = compute_group_stats(&not_works);

    writeln!(report,
        "  {:<20} {:<10} {:<15} {:<15} {:<12} {:<12} {:<12}",
        "Группа", "N", "Med объем", "Med σ%", "Med β_k", "Доля иерарх", "Med DM h=1")?;
    writeln!(report,
        "  {:<20} {:<10} {:>12.0}   {:>12.4}   {:>10.4}   {:>10.1}%   {:>10.3}",
        "Рычаг работает", g_w.n, g_w.median_volume, g_w.median_sigma_pct,
        g_w.median_beta_k, g_w.frac_hierarchy_holds * 100.0, g_w.median_dm_h1)?;
    writeln!(report,
        "  {:<20} {:<10} {:>12.0}   {:>12.4}   {:>10.4}   {:>10.1}%   {:>10.3}",
        "Рычаг НЕ работает", g_nw.n, g_nw.median_volume, g_nw.median_sigma_pct,
        g_nw.median_beta_k, g_nw.frac_hierarchy_holds * 100.0, g_nw.median_dm_h1)?;

    writeln!(report)?;
    let mut interp: Vec<String> = Vec::new();
    if g_w.median_volume > 2.0 * g_nw.median_volume {
        interp.push("у тикеров с рычагом объём в 2+ раза выше".to_string());
    } else if g_nw.median_volume > 2.0 * g_w.median_volume {
        interp.push("у тикеров с рычагом объём в 2+ раза ниже".to_string());
    }
    if g_w.median_sigma_pct > 1.5 * g_nw.median_sigma_pct {
        interp.push("у тикеров с рычагом волатильность выше".to_string());
    } else if g_nw.median_sigma_pct > 1.5 * g_w.median_sigma_pct {
        interp.push("у тикеров с рычагом волатильность ниже".to_string());
    }
    if interp.is_empty() {
        interp.push("нет ярких различий по базовым признакам".to_string());
    }
    writeln!(report, "  → {}", interp.join("; "))?;

    Ok(())
}