//! Графики результатов по всем уровням методологии.
//!
//! Каждый график реализован через generic-функцию `draw_X<DB: DrawingBackend>`
//! и вызывается дважды через макрос `render_both!` — один раз для PNG,
//! один раз для SVG. Это обходит ограничение Rust, запрещающее одной
//! closure быть полиморфной над двумя типами backend.

use anyhow::Result;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::FontStyle;

use crate::config::OUTPUT_DIR;
use crate::statistics::median;

const IMG_WIDTH: u32 = 1400;
const IMG_HEIGHT: u32 = 1100;
const FONT_SIZE: f64 = 46.0;
const LABEL_SIZE: f64 = 34.0;
const LEGEND_SIZE: f64 = 28.0;
const NOTE_SIZE: f64 = 30.0;
const MAIN_LINE_WIDTH: u32 = 4;
const IQR_LINE_WIDTH: u32 = 2;
const MARKER_SIZE: i32 = 6;
const MIN_CROSS_SECTIONAL_N: usize = 10;

#[derive(Clone, Copy)]
enum Lang {
    Ru,
    En,
}

impl Lang {
    fn pick<'a>(self, ru: &'a str, en: &'a str) -> &'a str {
        match self {
            Lang::Ru => ru,
            Lang::En => en,
        }
    }
}

// =====================================================================
// Макрос для рендеринга в PNG + SVG
// =====================================================================

macro_rules! render_both {
    ($name:expr, $draw_fn:ident $(, $arg:expr)*) => {{
        for &(suffix, lang) in &[("en", Lang::En), ("ru", Lang::Ru)] {
            let png = format!("{}/{}_{}.png", OUTPUT_DIR, $name, suffix);
            {
                let root = BitMapBackend::new(&png, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                $draw_fn(&root, lang $(, $arg)*)?;
            }
            let svg = format!("{}/{}_{}.svg", OUTPUT_DIR, $name, suffix);
            {
                let root = SVGBackend::new(&svg, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
                $draw_fn(&root, lang $(, $arg)*)?;
            }
        }
        Ok::<(), anyhow::Error>(())
    }};
}

// =====================================================================
// Вспомогательные функции
// =====================================================================

fn quantile_sorted(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return Some(sorted[lo]);
    }
    let w = pos - lo as f64;
    Some(sorted[lo] + w * (sorted[hi] - sorted[lo]))
}

#[derive(Clone, Copy, Debug)]
struct Summary {
    median: f64,
    q25: f64,
    q75: f64,
}

fn summary(values: &[f64]) -> Option<Summary> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if sorted.len() < MIN_CROSS_SECTIONAL_N {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(Summary {
        median: median(&sorted),
        q25: quantile_sorted(&sorted, 0.25)?,
        q75: quantile_sorted(&sorted, 0.75)?,
    })
}

fn positive_range(values: &[f64]) -> Option<(f64, f64)> {
    let valid: Vec<f64> = values
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x > 0.0)
        .collect();
    if valid.is_empty() {
        return None;
    }
    let lo = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some(((lo * 0.85).max(1e-12), hi * 1.15))
}

fn linear_range(values: &[f64]) -> Option<(f64, f64)> {
    let valid: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.is_empty() {
        return None;
    }
    let lo = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let pad = (hi - lo).abs().max(1e-9) * 0.10;
    Some((lo - pad, hi + pad))
}

/// Вспомогательная функция: нарисовать boxplot в произвольный backend.
fn draw_box<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<plotters::coord::types::RangedCoordf64,
                                              plotters::coord::types::RangedCoordf64>>,
    x_center: f64,
    s: &Summary,
    color: &RGBColor,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let half = 0.2;
    chart.draw_series(std::iter::once(Rectangle::new(
        [(x_center - half, s.q25), (x_center + half, s.q75)],
        ShapeStyle::from(&color.mix(0.3)).filled(),
    )))?;
    chart.draw_series(LineSeries::new(
        vec![
            (x_center - half, s.median),
            (x_center + half, s.median),
        ],
        ShapeStyle::from(color).stroke_width(MAIN_LINE_WIDTH),
    ))?;
    Ok(())
}

// =====================================================================
// Структуры данных для графиков
// =====================================================================

#[derive(Debug, Clone, Default)]
pub struct Level1PlotData {
    pub kurt_by_h: Vec<(usize, Vec<f64>)>,
    pub robust_kurt_by_h: Vec<(usize, Vec<f64>)>,
    pub alpha_by_h: Vec<(usize, Vec<f64>)>,
    pub mas_counts: Vec<(String, usize)>,
    pub betas: Vec<f64>,
    /// Диагностика роста RK: знаменатель RK, n_eff, RK на непересекающихся окнах.
    /// Позволяет понять, почему медиана RK растёт при падении K.
    pub rk_denom_by_h: Vec<(usize, Vec<f64>)>,
    pub n_eff_by_h: Vec<(usize, Vec<f64>)>,
    pub rk_nonoverlap_by_h: Vec<(usize, Vec<f64>)>,
}

#[derive(Debug, Clone, Default)]
pub struct Level2PlotData {
    pub example_f_mu: Vec<f64>,
    pub example_f_sigma2: Vec<f64>,
    pub f_mu_slopes: Vec<f64>,
    pub f_sigma2_slopes: Vec<f64>,
    pub mfdfa_spectrum: Vec<(f64, f64)>,
    /// Счётчик: сколько тикеров имеют значимый наклон f_μ.
    pub n_f_mu_significant: usize,
    /// Счётчик: сколько тикеров имеют значимый наклон f_σ².
    pub n_f_sigma2_significant: usize,
    /// Общее число тикеров с оценками наклонов.
    pub n_total_slopes: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Level3PlotData {
    pub jumps_by_minute: Vec<(usize, usize)>,
    pub phi_values: Vec<f64>,
    pub frac_economic: f64,
    pub n_total: usize,
    pub n_jumps_near_tax: usize,
    pub n_jumps_outside_tax: usize,
    pub tax_lifts: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct Level4PlotData {
    pub betas: Vec<f64>,
    pub t_stats: Vec<f64>,
    /// v2: диагностика и плацебо.
    pub perm_ps: Vec<f64>,
    pub t_ctrl: Vec<f64>,
    pub t_lag1: Vec<f64>,
    pub t_lead1: Vec<f64>,
    pub sens: Vec<(usize, Vec<f64>)>,
    pub example_scatter: Vec<(f64, f64)>,
    pub example_fit: Option<(f64, f64)>,
    /// H5: t_frac по тикерам.
    pub frac_ts: Vec<f64>,
    pub frac_perm_ps: Vec<f64>,
    /// H4a: t по уровню K_j.
    pub t_k_level: Vec<f64>,
    /// H4b: t по первой разности ΔK_j.
    pub t_k_delta: Vec<f64>,
    /// H5: t_mom по тикерам.
    pub mom_ts: Vec<f64>,
    pub mom_perm_ps: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct Level6PlotData {
    pub dm_by_h: Vec<(usize, Vec<f64>)>,
    pub gdm_pvalues: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct Level7PlotData {
    pub h_mu: Vec<f64>,
    pub h_sigma2: Vec<f64>,
    pub h_beta: Vec<f64>,
    pub hierarchy_holds_fraction: f64,
    pub beta_coint: Vec<f64>,
    // Метрики локального метода для отчёта.
    pub mu_bad_fraction: Vec<f64>,
    pub mu_median_drift: Vec<f64>,
    pub sigma2_bad_fraction: Vec<f64>,
    pub sigma2_median_drift: Vec<f64>,
    pub beta_bad_fraction: Vec<f64>,
    pub beta_median_drift: Vec<f64>,
}

// =====================================================================
// 1. Уровень I: эксцесс по горизонтам
// =====================================================================

fn draw_kurtosis_by_h<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    xs: &[f64],
    ys_kurt: &[f64],
    ys_rk: &[f64],
    q25: &[f64],
    q75: &[f64],
    rk_q25: &[f64],
    rk_q75: &[f64],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Медианный эксцесс и робастный эксцесс по горизонтам",
                "Median excess kurtosis and robust kurtosis by horizon",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d((x_min..x_max).log_scale(), (y_min..y_max).log_scale())?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Горизонт h (мин)", "Horizon h (min)"))
        .y_desc(lang.pick("Эксцесс", "Kurtosis"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    let q25_pts: Vec<(f64, f64)> = xs.iter().copied().zip(q25.iter().copied()).collect();
    let q75_pts: Vec<(f64, f64)> = xs.iter().copied().zip(q75.iter().copied()).collect();
    chart
        .draw_series(LineSeries::new(
            q25_pts,
            ShapeStyle::from(&BLUE.mix(0.35)).stroke_width(IQR_LINE_WIDTH),
        ))?
        .label(lang.pick("K Q25-Q75", "K Q25-Q75"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&BLUE.mix(0.35)).stroke_width(IQR_LINE_WIDTH),
            )
        });
    chart.draw_series(LineSeries::new(
        q75_pts,
        ShapeStyle::from(&BLUE.mix(0.35)).stroke_width(IQR_LINE_WIDTH),
    ))?;

    let rk_q25_pts: Vec<(f64, f64)> = xs.iter().copied().zip(rk_q25.iter().copied()).collect();
    let rk_q75_pts: Vec<(f64, f64)> = xs.iter().copied().zip(rk_q75.iter().copied()).collect();
    chart
        .draw_series(LineSeries::new(
            rk_q25_pts,
            ShapeStyle::from(&RED.mix(0.35)).stroke_width(IQR_LINE_WIDTH),
        ))?
        .label(lang.pick("RK Q25-Q75", "RK Q25-Q75"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&RED.mix(0.35)).stroke_width(IQR_LINE_WIDTH),
            )
        });
    chart.draw_series(LineSeries::new(
        rk_q75_pts,
        ShapeStyle::from(&RED.mix(0.35)).stroke_width(IQR_LINE_WIDTH),
    ))?;

    let kurt_pts: Vec<(f64, f64)> = xs.iter().copied().zip(ys_kurt.iter().copied()).collect();
    chart
        .draw_series(LineSeries::new(
            kurt_pts.clone(),
            ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
        ))?
        .label(lang.pick("Медиана K(h)", "Median K(h)"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
            )
        });

    let rk_pts: Vec<(f64, f64)> = xs.iter().copied().zip(ys_rk.iter().copied()).collect();
    chart
        .draw_series(LineSeries::new(
            rk_pts.clone(),
            ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
        ))?
        .label(lang.pick("Медиана RK(h)", "Median RK(h)"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
            )
        });

    chart.draw_series(kurt_pts.iter().map(|&(x, y)| {
        Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLUE).filled())
    }))?;
    chart.draw_series(rk_pts.iter().map(|&(x, y)| {
        TriangleMarker::new((x, y), MARKER_SIZE, ShapeStyle::from(&RED).filled())
    }))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(FontDesc::new(FontFamily::SansSerif, LEGEND_SIZE, FontStyle::Normal))
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_kurtosis_by_h(data: &Level1PlotData) -> Result<()> {
    let summaries: Vec<(usize, Summary, Summary)> = data
        .kurt_by_h
        .iter()
        .filter_map(|(h, vals)| {
            let s = summary(vals)?;
            let rk_summary = data
                .robust_kurt_by_h
                .iter()
                .find(|(hh, _)| hh == h)
                .and_then(|(_, rk_vals)| summary(rk_vals))?;
            Some((*h, s, rk_summary))
        })
        .collect();

    if summaries.len() < 2 {
        return Ok(());
    }

    let xs: Vec<f64> = summaries.iter().map(|(h, _, _)| *h as f64).collect();
    let ys_kurt: Vec<f64> = summaries.iter().map(|(_, s, _)| s.median).collect();
    let ys_rk: Vec<f64> = summaries.iter().map(|(_, _, rk)| rk.median).collect();
    let q25: Vec<f64> = summaries.iter().map(|(_, s, _)| s.q25).collect();
    let q75: Vec<f64> = summaries.iter().map(|(_, s, _)| s.q75).collect();
    let rk_q25: Vec<f64> = summaries.iter().map(|(_, _, rk)| rk.q25).collect();
    let rk_q75: Vec<f64> = summaries.iter().map(|(_, _, rk)| rk.q75).collect();

    let mut all_y = ys_kurt.clone();
    all_y.extend(ys_rk.iter().copied());
    all_y.extend(q25.iter().copied());
    all_y.extend(q75.iter().copied());
    all_y.extend(rk_q25.iter().copied());
    all_y.extend(rk_q75.iter().copied());

    let (x_min, x_max) = match positive_range(&xs) {
        Some(r) => r,
        None => return Ok(()),
    };
    let (y_min, y_max) = match positive_range(&all_y) {
        Some(r) => r,
        None => return Ok(()),
    };

    render_both!(
        "kurtosis_by_h",
        draw_kurtosis_by_h,
        &xs,
        &ys_kurt,
        &ys_rk,
        &q25,
        &q75,
        &rk_q25,
        &rk_q75,
        x_min,
        x_max,
        y_min,
        y_max
    )
}

fn draw_tail_index_by_h<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    xs: &[f64],
    med: &[f64],
    q25: &[f64],
    q75: &[f64],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Медианный хвостовой индекс α(h)",
                "Median tail index α(h)",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Горизонт h (мин)", "Horizon h (min)"))
        .y_desc("α")
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    let iqr_style = ShapeStyle::from(&BLUE.mix(0.35)).stroke_width(IQR_LINE_WIDTH);
    chart.draw_series(LineSeries::new(
        xs.iter().copied().zip(q25.iter().copied()).collect::<Vec<_>>(),
        iqr_style.clone(),
    ))?;
    chart.draw_series(LineSeries::new(
        xs.iter().copied().zip(q75.iter().copied()).collect::<Vec<_>>(),
        iqr_style,
    ))?;
    chart
        .draw_series(LineSeries::new(
            xs.iter().copied().zip(med.iter().copied()).collect::<Vec<_>>(),
            ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
        ))?
        .label(lang.pick("Медиана α(h)", "Median α(h)"))
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
            )
        });

    let pts: Vec<(f64, f64)> = xs.iter().copied().zip(med.iter().copied()).collect();
    chart.draw_series(pts.iter().map(|&(x, y)| {
        Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLUE).filled())
    }))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(FontDesc::new(FontFamily::SansSerif, LEGEND_SIZE, FontStyle::Normal))
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_tail_index_by_h(data: &Level1PlotData) -> Result<()> {
    let summaries: Vec<(usize, Summary)> = data
        .alpha_by_h
        .iter()
        .filter_map(|(h, vals)| summary(vals).map(|s| (*h, s)))
        .collect();
    if summaries.len() < 2 {
        return Ok(());
    }
    let xs: Vec<f64> = summaries.iter().map(|(h, _)| *h as f64).collect();
    let med: Vec<f64> = summaries.iter().map(|(_, s)| s.median).collect();
    let q25: Vec<f64> = summaries.iter().map(|(_, s)| s.q25).collect();
    let q75: Vec<f64> = summaries.iter().map(|(_, s)| s.q75).collect();

    let (x_min, x_max) = match linear_range(&xs) {
        Some(r) => r,
        None => return Ok(()),
    };
    let mut all_y = med.clone();
    all_y.extend(q25.iter().copied());
    all_y.extend(q75.iter().copied());
    let (y_min, y_max) = match positive_range(&all_y) {
        Some(r) => r,
        None => return Ok(()),
    };

    render_both!(
        "tail_index_by_h",
        draw_tail_index_by_h,
        &xs,
        &med,
        &q25,
        &q75,
        x_min,
        x_max,
        y_min,
        y_max
    )
}

fn draw_mas_distribution<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    labels: &[String],
    counts: &[f64],
    max_c: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let n = labels.len();

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "MAS: распределение выбранных моделей",
                "MAS: distribution of selected models",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(120)
        .y_label_area_size(120)
        .build_cartesian_2d(-0.5_f64..(n as f64 - 0.5), 0.0_f64..(max_c * 1.15))?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Модель", "Model"))
        .y_desc(lang.pick("Число тикеров", "Number of tickers"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .x_labels(n)
        .x_label_formatter(&|x| {
            let i = x.round() as isize;
            if i < 0 || (i as usize) >= labels.len() {
                String::new()
            } else {
                labels[i as usize].clone()
            }
        })
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = i as f64 - 0.4;
        let x1 = i as f64 + 0.4;
        Rectangle::new(
            [(x0, 0.0), (x1, c)],
            ShapeStyle::from(&BLUE.mix(0.6)).filled(),
        )
    }))?;

    root.present()?;
    Ok(())
}

pub fn plot_mas_model_distribution(data: &Level1PlotData) -> Result<()> {
    if data.mas_counts.is_empty() {
        return Ok(());
    }
    let labels: Vec<String> = data.mas_counts.iter().map(|(n, _)| n.clone()).collect();
    let counts: Vec<f64> = data.mas_counts.iter().map(|(_, c)| *c as f64).collect();
    let max_c = counts.iter().copied().fold(0.0_f64, f64::max).max(1.0);

    render_both!(
        "mas_model_distribution",
        draw_mas_distribution,
        &labels,
        &counts,
        max_c
    )
}

fn draw_beta_distribution<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    counts: &[usize],
    lo: f64,
    hi: f64,
    bw: f64,
    max_c: f64,
    med_beta: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Распределение β (степенной закон)",
                "Distribution of β (power law)",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(lo..hi, 0.0_f64..(max_c * 1.15))?;

    chart
        .configure_mesh()
        .x_desc("β")
        .y_desc(lang.pick("Число тикеров", "Number of tickers"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = lo + i as f64 * bw;
        let x1 = lo + (i + 1) as f64 * bw;
        Rectangle::new(
            [(x0, 0.0), (x1, c as f64)],
            ShapeStyle::from(&BLUE.mix(0.6)).filled(),
        )
    }))?;

    chart.draw_series(LineSeries::new(
        vec![(med_beta, 0.0), (med_beta, max_c * 1.1)],
        ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
    ))?;
    chart.draw_series(LineSeries::new(
        vec![(1.0, 0.0), (1.0, max_c * 1.1)],
        ShapeStyle::from(&GREEN).stroke_width(IQR_LINE_WIDTH),
    ))?;

    root.present()?;
    Ok(())
}


pub fn plot_beta_distribution(data: &Level1PlotData) -> Result<()> {
    let betas: Vec<f64> = data.betas.iter().copied().filter(|v| v.is_finite()).collect();
    if betas.len() < MIN_CROSS_SECTIONAL_N {
        return Ok(());
    }

    let bins = 30usize;
    let lo = betas.iter().copied().fold(f64::INFINITY, f64::min).max(-5.0);
    let hi = betas.iter().copied().fold(f64::NEG_INFINITY, f64::max).min(5.0);
    if !(hi > lo) {
        return Ok(());
    }
    let bw = (hi - lo) / bins as f64;

    let mut counts = vec![0usize; bins];
    for &b in &betas {
        let idx = ((b - lo) / bw).floor() as isize;
        if idx >= 0 && (idx as usize) < bins {
            counts[idx as usize] += 1;
        }
    }
    let max_c = *counts.iter().max().unwrap_or(&1) as f64;
    let med_beta = median(&betas);

    render_both!(
        "beta_distribution",
        draw_beta_distribution,
        &counts,
        lo,
        hi,
        bw,
        max_c,
        med_beta
    )
}

// =====================================================================
// 2. Уровень II: f_μ, f_σ², MF-DFA
// =====================================================================

fn draw_stationarity_ratios<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    n: usize,
    f_mu: &[f64],
    f_sigma2: &[f64],
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "f_μ(t) и f_σ²(t) для примера тикера",
                "f_μ(t) and f_σ²(t) for an example ticker",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(0.0_f64..(n as f64), y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("t (номер окна)", "t (window index)"))
        .y_desc(lang.pick("Отношение к эталону", "Ratio to reference"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(LineSeries::new(
        vec![(0.0, 1.0), (n as f64, 1.0)],
        ShapeStyle::from(&GREEN.mix(0.5)).stroke_width(IQR_LINE_WIDTH),
    ))?;

    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();

    let f_mu_pts: Vec<(f64, f64)> = t.iter().copied().zip(f_mu.iter().copied()).collect();
    chart
        .draw_series(LineSeries::new(
            f_mu_pts,
            ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
        ))?
        .label("f_μ(t)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
            )
        });

    let f_s_pts: Vec<(f64, f64)> = t.iter().copied().zip(f_sigma2.iter().copied()).collect();
    chart
        .draw_series(LineSeries::new(
            f_s_pts,
            ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
        ))?
        .label("f_σ²(t)")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
            )
        });

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(FontDesc::new(FontFamily::SansSerif, LEGEND_SIZE, FontStyle::Normal))
        .draw()?;

    root.present()?;
    Ok(())
}

pub fn plot_stationarity_ratios(data: &Level2PlotData) -> Result<()> {
    if data.example_f_mu.is_empty() {
        return Ok(());
    }
    let n = data.example_f_mu.len();
    let f_mu = &data.example_f_mu;
    let f_sigma2 = &data.example_f_sigma2;

    let mut all = f_mu.clone();
    all.extend(f_sigma2.iter().copied());
    let (mut y_min, mut y_max) = match linear_range(&all) {
        Some(r) => r,
        None => return Ok(()),
    };
    y_min = y_min.min(0.5);
    y_max = y_max.max(1.5);

    render_both!(
        "stationarity_ratios",
        draw_stationarity_ratios,
        n,
        f_mu,
        f_sigma2,
        y_min,
        y_max
    )
}

fn draw_f_slopes_boxplot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    s_mu: &Summary,
    s_sigma2: &Summary,
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Наклоны f_μ и f_σ² по тикерам",
                "Slopes of f_μ and f_σ² across tickers",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(0.0_f64..2.0_f64, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Момент", "Moment"))
        .y_desc(lang.pick("Наклон f по t", "Slope of f vs t"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .x_labels(2)
        .x_label_formatter(&|x| {
            if (x - 0.5).abs() < 0.1 {
                "f_μ".to_string()
            } else if (x - 1.5).abs() < 0.1 {
                "f_σ²".to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (2.0, 0.0)],
        ShapeStyle::from(&RED).stroke_width(IQR_LINE_WIDTH),
    ))?;

    draw_box(&mut chart, 0.5, s_mu, &BLUE)?;
    draw_box(&mut chart, 1.5, s_sigma2, &GREEN)?;

    root.present()?;
    Ok(())
}

pub fn plot_f_slopes_boxplot(data: &Level2PlotData) -> Result<()> {
    let s_mu = summary(&data.f_mu_slopes);
    let s_sigma2 = summary(&data.f_sigma2_slopes);
    let (Some(s_mu), Some(s_sigma2)) = (s_mu, s_sigma2) else {
        return Ok(());
    };

    let mut all = vec![s_mu.q25, s_mu.q75, s_sigma2.q25, s_sigma2.q75];
    all.push(0.0);
    let (y_min, y_max) = match linear_range(&all) {
        Some(r) => r,
        None => return Ok(()),
    };

    render_both!(
        "f_slopes_boxplot",
        draw_f_slopes_boxplot,
        &s_mu,
        &s_sigma2,
        y_min,
        y_max
    )
}

fn draw_mfdfa_spectrum<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    pts: &[(f64, f64)],
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Медианный мультифрактальный спектр",
                "Median multifractal spectrum",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("α")
        .y_desc("f(α)")
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(LineSeries::new(
        pts.iter().copied(),
        ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
    ))?;
    chart.draw_series(pts.iter().map(|&(x, y)| {
        Circle::new((x, y), MARKER_SIZE, ShapeStyle::from(&BLUE).filled())
    }))?;

    root.present()?;
    Ok(())
}

pub fn plot_mfdfa_spectrum(data: &Level2PlotData) -> Result<()> {
    if data.mfdfa_spectrum.len() < 3 {
        return Ok(());
    }
    let xs: Vec<f64> = data.mfdfa_spectrum.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = data.mfdfa_spectrum.iter().map(|p| p.1).collect();
    let (x_min, x_max) = match linear_range(&xs) {
        Some(r) => r,
        None => return Ok(()),
    };
    let (y_min, y_max) = match linear_range(&ys) {
        Some(r) => r,
        None => return Ok(()),
    };

    render_both!(
        "mfdfa_spectrum",
        draw_mfdfa_spectrum,
        &data.mfdfa_spectrum,
        x_min,
        x_max,
        y_min,
        y_max
    )
}

// =====================================================================
// 3. Уровень III: скачки
// =====================================================================

fn draw_jumps_by_minute<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    dense: &[f64],
    max_c: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let max_m = dense.len().saturating_sub(1);

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Распределение скачков по времени дня",
                "Distribution of jumps by time of day",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(180)
        .build_cartesian_2d(0.0_f64..(max_m as f64 + 1.0), 0.0_f64..(max_c * 1.15))?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Минута дня (от 09:30)", "Minute of day (from 09:30)"))
        .y_desc(lang.pick("Число скачков", "Number of jumps"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    let pts: Vec<(f64, f64)> = dense.iter().enumerate().map(|(i, &c)| (i as f64, c)).collect();
    chart.draw_series(LineSeries::new(
        pts,
        ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
    ))?;

    root.present()?;
    Ok(())
}

pub fn plot_jumps_by_minute(data: &Level3PlotData) -> Result<()> {
    if data.jumps_by_minute.is_empty() {
        return Ok(());
    }
    let max_m = data.jumps_by_minute.iter().map(|(m, _)| *m).max().unwrap_or(390);
    let max_c = data.jumps_by_minute.iter().map(|(_, c)| *c).max().unwrap_or(1);

    let mut dense = vec![0.0_f64; max_m + 1];
    for &(m, c) in &data.jumps_by_minute {
        if m <= max_m {
            dense[m] = c as f64;
        }
    }
    let max_c = max_c as f64;

    render_both!("jumps_by_minute", draw_jumps_by_minute, &dense, max_c)
}

fn draw_phi_distribution<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    counts: &[usize],
    lo: f64,
    hi: f64,
    bw: f64,
    max_c: f64,
    frac_economic: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let caption = match lang {
        Lang::Ru => format!("Распределение Φ (доля Φ>1 = {:.1}%)", frac_economic * 100.0),
        Lang::En => format!("Distribution of Φ (share Φ>1 = {:.1}%)", frac_economic * 100.0),
    };
    let mut chart = ChartBuilder::on(root)
        .caption(
            caption,
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(lo..hi, 0.0_f64..(max_c * 1.15))?;

    chart
        .configure_mesh()
        .x_desc("Φ = |r| / spread")
        .y_desc(lang.pick("Число скачков", "Number of jumps"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = lo + i as f64 * bw;
        let x1 = lo + (i + 1) as f64 * bw;
        Rectangle::new(
            [(x0, 0.0), (x1, c as f64)],
            ShapeStyle::from(&BLUE.mix(0.6)).filled(),
        )
    }))?;

    chart.draw_series(LineSeries::new(
        vec![(1.0, 0.0), (1.0, max_c * 1.1)],
        ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
    ))?;

    root.present()?;
    Ok(())
}

pub fn plot_phi_distribution(data: &Level3PlotData) -> Result<()> {
    let phi: Vec<f64> = data
        .phi_values
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    if phi.len() < MIN_CROSS_SECTIONAL_N {
        return Ok(());
    }

    let bins = 40usize;
    let lo = phi.iter().copied().fold(f64::INFINITY, f64::min).max(0.0);
    let hi = phi.iter().copied().fold(f64::NEG_INFINITY, f64::max).min(20.0);
    if !(hi > lo) {
        return Ok(());
    }
    let bw = (hi - lo) / bins as f64;

    let mut counts = vec![0usize; bins];
    for &p in &phi {
        let idx = ((p - lo) / bw).floor() as isize;
        if idx >= 0 && (idx as usize) < bins {
            counts[idx as usize] += 1;
        }
    }
    let max_c = *counts.iter().max().unwrap_or(&1) as f64;

    render_both!(
        "phi_distribution",
        draw_phi_distribution,
        &counts,
        lo,
        hi,
        bw,
        max_c,
        data.frac_economic
    )
}

// =====================================================================
// 4. Уровень IV: ускорение эксцесса как рычаг
// =====================================================================

fn draw_kurtosis_leverage_beta<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    counts: &[usize],
    lo: f64,
    hi: f64,
    bw: f64,
    max_c: f64,
    med_beta: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Распределение β_k (ускорение эксцесса как рычаг)",
                "Distribution of β_k (kurtosis acceleration as leverage)",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(lo..hi, 0.0_f64..(max_c * 1.15))?;

    chart
        .configure_mesh()
        .x_desc("β_k")
        .y_desc(lang.pick("Число тикеров", "Number of tickers"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = lo + i as f64 * bw;
        let x1 = lo + (i + 1) as f64 * bw;
        Rectangle::new(
            [(x0, 0.0), (x1, c as f64)],
            ShapeStyle::from(&BLUE.mix(0.6)).filled(),
        )
    }))?;

    chart.draw_series(LineSeries::new(
        vec![(med_beta, 0.0), (med_beta, max_c * 1.1)],
        ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
    ))?;
    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (0.0, max_c * 1.1)],
        ShapeStyle::from(&GREEN).stroke_width(IQR_LINE_WIDTH),
    ))?;

    root.present()?;
    Ok(())
}


pub fn plot_kurtosis_leverage_beta(data: &Level4PlotData) -> Result<()> {
    let betas: Vec<f64> = data.betas.iter().copied().filter(|v| v.is_finite()).collect();
    if betas.len() < MIN_CROSS_SECTIONAL_N {
        return Ok(());
    }

    let bins = 30usize;
    let lo = betas.iter().copied().fold(f64::INFINITY, f64::min).max(-5.0);
    let hi = betas.iter().copied().fold(f64::NEG_INFINITY, f64::max).min(5.0);
    if !(hi > lo) {
        return Ok(());
    }
    let bw = (hi - lo) / bins as f64;

    let mut counts = vec![0usize; bins];
    for &b in &betas {
        let idx = ((b - lo) / bw).floor() as isize;
        if idx >= 0 && (idx as usize) < bins {
            counts[idx as usize] += 1;
        }
    }
    let max_c = *counts.iter().max().unwrap_or(&1) as f64;
    let med_beta = median(&betas);

    render_both!(
        "kurtosis_leverage_beta",
        draw_kurtosis_leverage_beta,
        &counts,
        lo,
        hi,
        bw,
        max_c,
        med_beta
    )
}

fn draw_kurtosis_leverage_scatter<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    scatter: &[(f64, f64)],
    fit: Option<(f64, f64)>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Пример (частичная регрессия): s·Ã_K,j против R_{j+1}/s после учёта α",
                "Example (partial regression): s·Ã_K,j vs R_{j+1}/s after removing α",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(lang.pick(
            "s_j·Ã_K,j (остаток после s_j; Ã — стандартизованное ускорение эксцесса блока)",
            "s_j·Ã_K,j (residual after s_j; Ã = standardized block kurtosis acceleration)",
        ))
        .y_desc(lang.pick(
            "R_{j+1} / s_j (остаток после s_j)",
            "R_{j+1} / s_j (residual after s_j)",
        ))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(scatter.iter().map(|&(x, y)| {
        Circle::new((x, y), 3, ShapeStyle::from(&BLUE.mix(0.5)).filled())
    }))?;

    if let Some((slope, intercept)) = fit {
        let line = vec![
            (x_min, intercept + slope * x_min),
            (x_max, intercept + slope * x_max),
        ];
        chart.draw_series(LineSeries::new(
            line,
            ShapeStyle::from(&RED).stroke_width(MAIN_LINE_WIDTH),
        ))?;
    }

    root.present()?;
    Ok(())
}

pub fn plot_kurtosis_leverage_scatter(data: &Level4PlotData) -> Result<()> {
    if data.example_scatter.len() < MIN_CROSS_SECTIONAL_N {
        return Ok(());
    }
    let xs: Vec<f64> = data.example_scatter.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = data.example_scatter.iter().map(|p| p.1).collect();
    let (x_min, x_max) = match linear_range(&xs) {
        Some(r) => r,
        None => return Ok(()),
    };
    let (y_min, y_max) = match linear_range(&ys) {
        Some(r) => r,
        None => return Ok(()),
    };

    render_both!(
        "kurtosis_leverage_scatter",
        draw_kurtosis_leverage_scatter,
        &data.example_scatter,
        data.example_fit,
        x_min,
        x_max,
        y_min,
        y_max
    )
}

fn draw_leverage_tstats<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    counts: &[usize],
    lo: f64,
    hi: f64,
    bw: f64,
    max_c: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Распределение t-статистик β_k",
                "Distribution of t-statistics of β_k",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(lo..hi, 0.0_f64..(max_c * 1.15))?;

    chart
        .configure_mesh()
        .x_desc("t_k")
        .y_desc(lang.pick("Число тикеров", "Number of tickers"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(counts.iter().enumerate().map(|(i, &c)| {
        let x0 = lo + i as f64 * bw;
        let x1 = lo + (i + 1) as f64 * bw;
        Rectangle::new(
            [(x0, 0.0), (x1, c as f64)],
            ShapeStyle::from(&BLUE.mix(0.6)).filled(),
        )
    }))?;

    for &crit in &[-1.96_f64, 1.96_f64] {
        chart.draw_series(LineSeries::new(
            vec![(crit, 0.0), (crit, max_c * 1.1)],
            ShapeStyle::from(&RED).stroke_width(IQR_LINE_WIDTH),
        ))?;
    }

    root.present()?;
    Ok(())
}

pub fn plot_leverage_tstats(data: &Level4PlotData) -> Result<()> {
    let ts: Vec<f64> = data.t_stats.iter().copied().filter(|v| v.is_finite()).collect();
    if ts.len() < MIN_CROSS_SECTIONAL_N {
        return Ok(());
    }

    let bins = 40usize;
    let lo = ts.iter().copied().fold(f64::INFINITY, f64::min).max(-10.0);
    let hi = ts.iter().copied().fold(f64::NEG_INFINITY, f64::max).min(10.0);
    if !(hi > lo) {
        return Ok(());
    }
    let bw = (hi - lo) / bins as f64;

    let mut counts = vec![0usize; bins];
    for &t in &ts {
        let idx = ((t - lo) / bw).floor() as isize;
        if idx >= 0 && (idx as usize) < bins {
            counts[idx as usize] += 1;
        }
    }
    let max_c = *counts.iter().max().unwrap_or(&1) as f64;

    render_both!(
        "leverage_tstats",
        draw_leverage_tstats,
        &counts,
        lo,
        hi,
        bw,
        max_c
    )
}

// =====================================================================
// 5. Уровень VI: DM / GDM
// =====================================================================

fn draw_dm_boxplot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    summaries: &[(usize, Summary)],
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    let n = summaries.len();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "DM (DWT − EWMA) по горизонтам",
                "DM (DWT − EWMA) by horizon",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(-0.5_f64..(n as f64 - 0.5), y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Горизонт h", "Horizon h"))
        .y_desc("DM")
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .x_labels(n)
        .x_label_formatter(&|x| {
            let i = x.round() as isize;
            if i < 0 || (i as usize) >= summaries.len() {
                String::new()
            } else {
                summaries[i as usize].0.to_string()
            }
        })
        .draw()?;

    chart.draw_series(LineSeries::new(
        vec![(-0.5, 0.0), (n as f64 - 0.5, 0.0)],
        ShapeStyle::from(&RED).stroke_width(IQR_LINE_WIDTH),
    ))?;
    for &crit in &[-1.96_f64, 1.96_f64] {
        chart.draw_series(LineSeries::new(
            vec![(-0.5, crit), (n as f64 - 0.5, crit)],
            ShapeStyle::from(&GREEN.mix(0.6)).stroke_width(IQR_LINE_WIDTH),
        ))?;
    }

    for (i, (_, s)) in summaries.iter().enumerate() {
        draw_box(&mut chart, i as f64, s, &BLUE)?;
    }

    root.present()?;
    Ok(())
}

pub fn plot_dm_boxplot(data: &Level6PlotData) -> Result<()> {
    let summaries: Vec<(usize, Summary)> = data
        .dm_by_h
        .iter()
        .filter_map(|(h, vals)| summary(vals).map(|s| (*h, s)))
        .collect();
    if summaries.len() < 2 {
        return Ok(());
    }

    let mut all = Vec::new();
    for (_, s) in &summaries {
        all.push(s.q25);
        all.push(s.q75);
    }
    all.push(0.0);
    all.push(-2.0);
    all.push(2.0);
    let (y_min, y_max) = match linear_range(&all) {
        Some(r) => r,
        None => return Ok(()),
    };

    render_both!("dm_boxplot", draw_dm_boxplot, &summaries, y_min, y_max)
}

fn draw_gdm_pvalue_cdf<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    cdf: &[(f64, f64)],
    frac_05: f64,
    frac_01: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Эмпирическая CDF p-value GDM",
                "Empirical CDF of GDM p-values",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(0.0_f64..1.0_f64, 0.0_f64..1.0_f64)?;

    chart
        .configure_mesh()
        .x_desc("p-value")
        .y_desc(lang.pick("Доля тикеров", "Share of tickers"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    for &alpha in &[0.05_f64, 0.01_f64] {
        chart.draw_series(LineSeries::new(
            vec![(alpha, 0.0), (alpha, 1.0)],
            ShapeStyle::from(&RED.mix(0.5)).stroke_width(IQR_LINE_WIDTH),
        ))?;
    }
    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (1.0, 1.0)],
        ShapeStyle::from(&GREEN.mix(0.6)).stroke_width(IQR_LINE_WIDTH),
    ))?;
    chart.draw_series(LineSeries::new(
        cdf.iter().copied(),
        ShapeStyle::from(&BLUE).stroke_width(MAIN_LINE_WIDTH),
    ))?;

    let label05 = format!("{:.1}% p<0.05", frac_05 * 100.0);
    let label01 = format!("{:.1}% p<0.01", frac_01 * 100.0);
    chart.draw_series(std::iter::once(Text::new(
        label05,
        (0.06, (frac_05 + 0.04).min(0.92)),
        FontDesc::new(FontFamily::SansSerif, NOTE_SIZE, FontStyle::Normal).color(&RED),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        label01,
        (0.02, (frac_01 + 0.04).min(0.84)),
        FontDesc::new(FontFamily::SansSerif, NOTE_SIZE, FontStyle::Normal).color(&RED),
    )))?;

    root.present()?;
    Ok(())
}


pub fn plot_gdm_pvalue_cdf(data: &Level6PlotData) -> Result<()> {
    let mut pvals: Vec<f64> = data
        .gdm_pvalues
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if pvals.len() < MIN_CROSS_SECTIONAL_N {
        return Ok(());
    }
    pvals.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = pvals.len() as f64;
    let mut cdf: Vec<(f64, f64)> = Vec::with_capacity(pvals.len() + 2);
    cdf.push((0.0, 0.0));
    for (i, &p) in pvals.iter().enumerate() {
        cdf.push((p, i as f64 / n));
        cdf.push((p, (i as f64 + 1.0) / n));
    }
    cdf.push((1.0, 1.0));

    let frac_05 = pvals.iter().filter(|&&p| p < 0.05).count() as f64 / n;
    let frac_01 = pvals.iter().filter(|&&p| p < 0.01).count() as f64 / n;

    render_both!(
        "gdm_pvalue_cdf",
        draw_gdm_pvalue_cdf,
        &cdf,
        frac_05,
        frac_01
    )
}

// =====================================================================
// 6. Уровень VII: H*(θ) и иерархия
// =====================================================================

fn draw_hstar_boxplot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    lang: Lang,
    s_mu: &Summary,
    s_sigma2: &Summary,
    s_beta: &Summary,
    y_min: f64,
    y_max: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            lang.pick(
                "Горизонты статистической пригодности H*(θ)",
                "Statistical usability horizons H*(θ)",
            ),
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(140)
        .build_cartesian_2d(0.0_f64..3.0_f64, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc(lang.pick("Момент θ", "Moment θ"))
        .y_desc(lang.pick("H*(θ), баров", "H*(θ), bars"))
        .label_style(FontDesc::new(FontFamily::SansSerif, LABEL_SIZE, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .x_labels(3)
        .x_label_formatter(&|x| {
            if (x - 0.5).abs() < 0.1 {
                "μ".to_string()
            } else if (x - 1.5).abs() < 0.1 {
                "σ²".to_string()
            } else if (x - 2.5).abs() < 0.1 {
                "β".to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    draw_box(&mut chart, 0.5, s_mu, &BLUE)?;
    draw_box(&mut chart, 1.5, s_sigma2, &GREEN)?;
    draw_box(&mut chart, 2.5, s_beta, &RED)?;

    root.present()?;
    Ok(())
}

pub fn plot_hstar_boxplot(data: &Level7PlotData) -> Result<()> {
    // Для H* используем пониженный порог: 5 тикеров минимум.
    if data.h_mu.len() < 5 || data.h_sigma2.len() < 5 || data.h_beta.len() < 5 {
        return Ok(());
    }
    let s_mu = summary(&data.h_mu);
    let s_sigma2 = summary(&data.h_sigma2);
    let s_beta = summary(&data.h_beta);
    let (Some(s_mu), Some(s_sigma2), Some(s_beta)) = (s_mu, s_sigma2, s_beta) else {
        return Ok(());
    };

    let mut all = vec![
        s_mu.q25, s_mu.q75,
        s_sigma2.q25, s_sigma2.q75,
        s_beta.q25, s_beta.q75,
    ];
    let (mut y_min, y_max) = match linear_range(&all) {
        Some(r) => r,
        None => return Ok(()),
    };
    y_min = y_min.max(0.0);
    all.clear();

    render_both!(
        "hstar_boxplot",
        draw_hstar_boxplot,
        &s_mu,
        &s_sigma2,
        &s_beta,
        y_min,
        y_max
    )
}

// =====================================================================
// 7. Оркестрация
// =====================================================================

pub fn generate_all_plots(
    l1: &Level1PlotData,
    l2: &Level2PlotData,
    l3: &Level3PlotData,
    l4: &Level4PlotData,
    l6: &Level6PlotData,
    l7: &Level7PlotData,
) -> Result<()> {
    if let Err(e) = plot_kurtosis_by_h(l1) {
        eprintln!("Warning: kurtosis_by_h: {}", e);
    }
    if let Err(e) = plot_tail_index_by_h(l1) {
        eprintln!("Warning: tail_index_by_h: {}", e);
    }
    if let Err(e) = plot_mas_model_distribution(l1) {
        eprintln!("Warning: mas_model_distribution: {}", e);
    }
    if let Err(e) = plot_beta_distribution(l1) {
        eprintln!("Warning: beta_distribution: {}", e);
    }

    if let Err(e) = plot_stationarity_ratios(l2) {
        eprintln!("Warning: stationarity_ratios: {}", e);
    }
    if let Err(e) = plot_f_slopes_boxplot(l2) {
        eprintln!("Warning: f_slopes_boxplot: {}", e);
    }
    if let Err(e) = plot_mfdfa_spectrum(l2) {
        eprintln!("Warning: mfdfa_spectrum: {}", e);
    }

    if let Err(e) = plot_jumps_by_minute(l3) {
        eprintln!("Warning: jumps_by_minute: {}", e);
    }
    if let Err(e) = plot_phi_distribution(l3) {
        eprintln!("Warning: phi_distribution: {}", e);
    }

    if let Err(e) = plot_kurtosis_leverage_beta(l4) {
        eprintln!("Warning: kurtosis_leverage_beta: {}", e);
    }
    if let Err(e) = plot_kurtosis_leverage_scatter(l4) {
        eprintln!("Warning: kurtosis_leverage_scatter: {}", e);
    }
    if let Err(e) = plot_leverage_tstats(l4) {
        eprintln!("Warning: leverage_tstats: {}", e);
    }

    if let Err(e) = plot_dm_boxplot(l6) {
        eprintln!("Warning: dm_boxplot: {}", e);
    }
    if let Err(e) = plot_gdm_pvalue_cdf(l6) {
        eprintln!("Warning: gdm_pvalue_cdf: {}", e);
    }

    if let Err(e) = plot_hstar_boxplot(l7) {
        eprintln!("Warning: hstar_boxplot: {}", e);
    }


    Ok(())
}