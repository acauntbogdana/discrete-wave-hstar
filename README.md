# discrete-wave-hstar

Rust-реализация методологии **горизонта статистической пригодности H\*(θ)** и проверки **неэргодичности моментов** на высокочастотных финансовых данных.

Сопровождает исследование *«Неэргодичность моментов и горизонт статистической пригодности на высокочастотных финансовых данных»*.

---

## Что делает

Из потока IEX DEEP строит минутные бары (OHLCV + best_bid/best_ask) и последовательно прогоняет **8 уровней** анализа:

| Уровень | Модуль | Содержание |
|---|---|---|
| I | `kurtosis.rs` | Эксцесс, робастный RK, Хилл, MAS-подбор формы убывания |
| II | `ergodicity.rs` | f_μ, f_σ², Theil–Sen, CUSUM-BWB, MF-DFA |
| III | `microstructure.rs` | Med9 RV, скачки J₄, OFI, Φ = \|r\|/spread, налоговые окна |
| IV | `factors/leverage.rs` | Лестница H4a/H4b/H4c — уровень, ΔK, ускорение K |
| V | `factors/escape.rs` | H5 — escape из барьера Гроссмана–Миллера |
| VI | `hysteresis.rs` | DWT-стек разворотов, DM/GDM тесты |
| VII | `horizon.rs` | VAR(2), VECM, Johansen, измерение H\*(μ), H\*(σ²), H\*(β) |
| VIII | `factors/adaptive.rs` | AR(1)+RLS с забыванием + логит смены режима |

На выходе: `report.txt`, графики в PNG+SVG на двух языках (EN/RU), чекпоинт по тикерам.

---

## Сборка и запуск

```bash
cargo build --release -j1 
cargo run --release
```

Рантайм однопоточный. Пути и пороги задаются в `src/config.rs`.

---

## Зависимости

`arrow`/`parquet` 53, `nalgebra` 0.32, `statrs` 0.16, `plotters` 0.3, `chrono`, `serde`/`rmp-serde`, `rand` 0.8, `fs2`, `env_logger`. Rust ≥ 1.75.

---
