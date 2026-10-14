use std::time::Instant;
use anyhow::Result;
use env_logger::Env;
use discrete_wave_hstar::analysis_core;

fn main() -> Result<()> {
    env_logger::Builder::from_env(Env::default().default_filter_or("warn")).init();

    let start = Instant::now();
    analysis_core::run()?;
    println!("Общее время: {:.1} с", start.elapsed().as_secs_f64());
    Ok(())
}