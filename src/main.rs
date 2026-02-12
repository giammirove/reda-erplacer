use crate::{
    drawing::draw_placement,
    placement::Placement,
    utils::{force_time_it, time_it},
};
use clap::Parser;
use eyre::Result;
use reda_db::read_db;
use std::ffi::OsString;

mod dct2d;
mod density;
mod drawing;
mod eforce;
mod epotential;
mod frequencies;
mod hpwl;
mod idct2;
mod idct_idxst;
mod idxst_idct;
mod irfft2d;
mod nesterov;
mod placement;
mod preconditioner;
mod scheduler;
mod utils;

#[derive(Parser, Debug)]
#[command(name = "dreamplace-lefdef")]
#[command(version = "0.1.0")]
#[command(about = "VLSI Global Placement using electron-placer with LEF/DEF input", long_about = None)]
struct Args {
    /// Input DEF file
    #[arg(short = 'd', long = "def", value_name = "FILE")]
    def: OsString,

    /// Input LEF file(s) - can specify multiple times
    #[arg(short = 'l', long = "lef", value_name = "FILE")]
    lef: OsString,

    // /// Target density (0.0 to 1.0)
    // #[arg(long, default_value = "0.9", value_name = "FLOAT")]
    // target_density: f32,
    /// Maximum placement iterations
    #[arg(long, default_value = "1", value_name = "NUM")]
    iterations: usize,

    /// Enable verbose logging
    #[arg(short = 'v', long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let log_level = if args.verbose { "debug" } else { "info" };
    let mut builder =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level));
    builder
        .write_style(env_logger::WriteStyle::Always)
        .format_timestamp(None)
        .format_module_path(false)
        .format_source_path(false)
        .format_file(false)
        .format_target(false)
        .init();

    let db = time_it("Reading DB", || {
        read_db::<f32>(&args.lef, &args.def, args.verbose)
    })?;

    let mut placement = time_it("Creating Placement", || Placement::new(&db));
    log::info!("{}", placement);

    draw_placement(placement.db, &placement.ps, 0);

    force_time_it("Placement", || {
        for i in 0..args.iterations {
            force_time_it("Step", || {
                let should_stop = placement.step();
                if i % 50 == 0 && args.verbose {
                    draw_placement(placement.db, &placement.ps, i);
                }

                if should_stop {
                    log::info!("Something is wrong!");
                    if args.verbose {
                        draw_placement(placement.db, &placement.ps, i);
                    }
                }
            });
        }
    });

    draw_placement(placement.db, &placement.ps, args.iterations);

    Ok(())
}
