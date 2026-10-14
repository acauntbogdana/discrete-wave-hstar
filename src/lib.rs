//! # discrete-wave-hstar
//!
//! Единая методология: дискретность → эргодичность → микроструктура →
//! факторы → экономический барьер → гистерезис → горизонт H*(θ).
//!
//! Уровень IV (ускорение эксцесса как рычаг) вынесен в подмодуль
//! `factors::leverage` — переработанный тест с непересекающимися
//! блоками, выравниванием «регрессор → строго будущий отклик»,
//! HAC по правильной матрице (x·e), плацебо-перестановкой,
//! профилем сдвигов и BH-поправкой.

pub mod config;
pub mod data_loading;      // построение баров из IEX DEEP
pub mod bars_io;           // чтение баров из Parquet
pub mod calendar;          // календарь NYSE
pub mod tax_calendar;      // налоговые даты IRS 2018–2022
pub mod statistics;        // median, mean, bootstrap
pub mod equations;         // эксцесс, Хилл, MF-DFA, CUSUM, S-TSRV
pub mod dwt;               // стек разворотов
pub mod var_models;        // EWMA, DWT-волатильность, VaR
pub mod dm_test;           // DM/GDM

pub mod kurtosis;          // робастный эксцесс + MAS
pub mod ergodicity;        // f_μ, f_σ², TSS, HAC-наклон
pub mod microstructure;    // Med9, OFI, PC-IVP
pub mod factors;           // логит + реэкспорт `factors::leverage`
pub mod hysteresis;        // обёртка уровня VI (реэкспорт)
pub mod horizon;           // VAR(2), VECM, Johansen, H*, measure_h_star_coint

// Объявлены, но пока не подключены к основному пайплайну.
// Активируются при подключении логит-модели уровня IV.
pub mod fundamental_data;
pub mod macro_data;

pub mod analysis_core;
pub mod plots;