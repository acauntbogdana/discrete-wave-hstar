//! Календарь ключевых налоговых дат IRS (2018–2022).
//!
//! Используется для проверки гипотезы: кластеризуются ли скачки
//! вокруг дат уплаты налогов.
//!
//! Список дат — рекуррентный, без учёта переносов на выходные:
//!   - 15-е число каждого месяца: сроки уплаты estimated tax,
//!     подачи деклараций, отчётности.
//!   - Последний день каждого месяца: payroll tax deposits,
//!     формы 720, 941, 730, 2290.
//!
//! Наиболее значимые для рынка:
//!   - Апрель 15 — Tax Day (декларации физлиц + Q1 estimated tax),
//!   - Январь 15, Июнь 15, Сентябрь 15 — quarterly estimated tax,
//!   - Март 15 — S-Corp и partnership returns,
//!   - Декабрь 15 — Q4 estimated tax корпораций.

use chrono::NaiveDate;

/// Возвращает последний день месяца (28–31).
pub fn last_day_of_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                29
            } else {
                28
            }
        }
        _ => 28,
    }
}

/// Возвращает список налоговых дат за 2018–2022.
///
/// Дата — 15-е каждого месяца И последний день каждого месяца.
/// Итого 24 даты в год × 5 лет = 120 дат.
pub fn tax_dates_2018_2022() -> Vec<NaiveDate> {
    let mut dates = Vec::with_capacity(120);
    for year in 2018..=2022 {
        for month in 1..=12 {
            // 15-е число
            if let Some(d) = NaiveDate::from_ymd_opt(year, month, 15) {
                dates.push(d);
            }
            // Последний день
            let last = last_day_of_month(year, month);
            if let Some(d) = NaiveDate::from_ymd_opt(year, month, last) {
                dates.push(d);
            }
        }
    }
    dates.sort();
    dates
}

/// Проверяет, попадает ли `date` в окно ±`window_days` от любой
/// налоговой даты.
pub fn is_near_tax_date(date: NaiveDate, window_days: i64, dates: &[NaiveDate]) -> bool {
    dates.iter().any(|&td| {
        let diff = (date - td).num_days().abs();
        diff <= window_days
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tax_dates_count() {
        let dates = tax_dates_2018_2022();
        assert_eq!(dates.len(), 120);
    }

    #[test]
    fn test_april_15_included() {
        let dates = tax_dates_2018_2022();
        let april_15_2020 = NaiveDate::from_ymd_opt(2020, 4, 15).unwrap();
        assert!(dates.contains(&april_15_2020));
    }

    #[test]
    fn test_near_tax_date() {
        let dates = vec![NaiveDate::from_ymd_opt(2020, 4, 15).unwrap()];
        let d1 = NaiveDate::from_ymd_opt(2020, 4, 13).unwrap();
        let d2 = NaiveDate::from_ymd_opt(2020, 4, 25).unwrap();
        assert!(is_near_tax_date(d1, 3, &dates));
        assert!(!is_near_tax_date(d2, 3, &dates));
    }
}