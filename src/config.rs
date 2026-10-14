// Константы конфигурации.

use crate::equations::StsrvParams;

// Пути
pub const DEEP_DIR: &str     = r"Ваш путь:\IEX\DEEP";
pub const FUND_DIR: &str     = r"Ваш путь:\IEX\Fundamental\parquet_by_year";
pub const TICKERS_FILE: &str = r"Ваш путь:\IEX\Fundamental\expanded_tickers.parquet";
pub const MACRO_FILE: &str   = r"Ваш путь:\IEX\Macro\macro_indicators_2018_2022.parquet";
pub const OUTPUT_DIR: &str   = r"Ваш путь:\Users\Bogdan\Desktop\HSTAR";

// --- Данные ---
pub const BAR_INTERVAL_MIN: i64 = 1;
pub const MIN_VOLUME: u64 = 1_000_000;
pub const MIN_BARS: usize = 500;
pub const WARMUP_BARS: usize = 200;

// --- Управление пайплайном ---
/// Пропускать построение баров (шаг 1), если каталог bars/ уже
/// содержит parquet-файлы. Установите `false`, чтобы принудительно
/// пересобрать бары с нуля (медленно, часы работы).
pub const SKIP_BAR_BUILDING_IF_READY: bool = true;

// --- Уровень I ---
/// Горизонты агрегирования (в барах, т.е. минутах).
pub const HORIZONS: &[usize] = &[1, 5, 15, 30, 60, 120, 240, 480, 1440];
/// Число кандидатных моделей MAS (Miyahara et al., 2024).
pub const MAS_K_FOLDS: usize = 4;
pub const MAS_TRAIN_FRACTION: f64 = 0.75;

// --- Уровень II ---
/// Скользящее окно для f_μ(t), f_σ²(t) (Nerantzaki et al., 2023).
pub const WINDOW_MOMENTS: usize = 60;
/// Bootstrap для CUSUM-BWB.
pub const CUSUM_BOOTSTRAP_B: usize = 200;

// --- Уровень III ---
/// Med9: число соседних возвратов.
pub const MED_K: usize = 9;
/// c_9 из Chen et al. (2025), ур. (12).
pub const MED9_C: f64 = 1.82184;
/// Порог детекции скачка J_4.
pub const JUMP_THRESHOLD_STD: f64 = 4.0;
/// Уровни стакана для OFI (Cont et al., 2023).
pub const OFI_LEVELS: usize = 10;
/// Число подынтервалов в PC-IVP.
pub const PC_IVP_N: usize = 3;

pub const STSRV_PARAMS: StsrvParams = StsrvParams {
    k: 20, j: 5, m_preavg: 5,
    threshold: 0.001, max_tau_step: 1.0 / (252.0 * 390.0), n_blocks: 390,
};

// --- Уровень IV ---
pub const VAR_LEVEL: f64 = 0.05;
pub const EWMA_LAMBDA: f64 = 0.94;

// --- Уровень V ---
pub const JUMP_THRESHOLD_MULT: f64 = 3.0;
pub const ECONOMIC_BARRIER_PHI: f64 = 1.0;

// --- Уровень VI ---
pub const DELTA_REL: f64 = 0.005;
pub const BOOTSTRAP_B: usize = 1000;

/// Ширина полосы для QS-ядра (Andrews 1991).
/// QS1 = 1.3221 · N^(1/5), QS2 = 1.80 · N^(1/5).
pub const QS_KERNEL_COEF: f64 = 1.80;

// --- Уровень VII ---
/// Максимальный допустимый локальный drift момента на горизонте L,
/// выраженный в единицах robust scale (1.4826·MAD). Безразмерный,
/// применим и к μ, и к σ², и к β. 0.50 означает: за один локальный
/// сегмент длины L момент не должен сдвигаться более чем на половину
/// своего robust-стандартного масштаба.
pub const HSTAR_DRIFT_TOL: f64 = 0.50;

/// Максимальная доля локальных сегментов, в которых drift превышает
/// HSTAR_DRIFT_TOL. Если доля "плохих" сегментов ≤ этого порога,
/// момент считается стабильным на данном W.
pub const HSTAR_MAX_BAD_FRACTION: f64 = 0.20;

/// Минимальное число локальных сегментов для валидной оценки.
/// Если ряд слишком короткий, H* не определяется (возвращается 0).
pub const HSTAR_MIN_SEGMENTS: usize = 8;

/// Множитель для длины локального сегмента: L = max(60, HSTAR_LOCAL_MULT·W).
pub const HSTAR_LOCAL_MULT: usize = 2;

// Диапазоны W отдельно для каждого момента.
// β — ускорение эксцесса определено на трёх блоках вперёд, поэтому
// минимальное окно больше, чем для μ и σ².
pub const HSTAR_MU_MIN_W: usize = 20;
pub const HSTAR_MU_MAX_W: usize = 500;
pub const HSTAR_SIGMA2_MIN_W: usize = 20;
pub const HSTAR_SIGMA2_MAX_W: usize = 500;
pub const HSTAR_BETA_MIN_W: usize = 2;
pub const HSTAR_BETA_MAX_W: usize = 8;
/// Минимум сегментов для β меньше, чем для μ/σ²: при W=60 блоков мало.
pub const HSTAR_BETA_MIN_SEGMENTS: usize = 4;
/// Уровень значимости Johansen trace test.
pub const JOHANSEN_ALPHA: f64 = 0.05;
/// Критические значения trace-теста (Osterwald-Lenum, 1992).
/// H0: r ≤ 0 против H1: r > 0.
pub const TRACE_R0_CRIT_5PCT: f64 = 19.96;
/// H0: r ≤ 1 против H1: r > 1.
pub const TRACE_R1_CRIT_5PCT: f64 = 9.24;

/// Макропеременные панели уровня IV (см. комментарий в старом config.rs).
pub const SELECTED_MACRO_COLS: &[&str] = &[
    "vix_fred_ffill",
    "ted_spread_ffill",
    "fed_funds_rate_ffill",
    "treasury_10y_ffill",
    "economic_policy_uncertainty_ffill",
    "CPI_headline_ffill",
    "PCE_headline_ffill",
    "PPI_all_commodities_ffill",
    "unemployment_rate_ffill",
    "industrial_production_ffill",
    "nonfarm_payrolls_ffill",
    "umich_consumer_sentiment_ffill",
    "retail_sales_ffill",
    "dollar_index_broad_ffill",
    "real_effective_exchange_rate_ffill",
    "oil_wti_ffill",
    "gold_ffill",
    "copper_ffill",
    "m2_money_supply_ffill",
    "gdp_real_ffill",
    "default_spread_baa_aaa_ffill",
    "yield_curve_10y_2y_ffill",
    "real_rate_10y_tips_ffill",
];