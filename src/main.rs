use crate::{
    drawing::draw_placement,
    placement::Placement,
    utils::{
        force_time_it, load_macro_colors, load_macros, save_macros, spread_macros, time_it,
        write_netlist,
    },
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
#[command(name = "reda-erplacer")]
#[command(version = "0.1.0")]
struct Args {
    /// Input DEF file
    #[arg(short = 'd', long = "def", value_name = "FILE")]
    def: OsString,

    /// Input LEF file(s) - can specify multiple times
    #[arg(short = 'l', long = "lef", value_name = "FILE")]
    lef: OsString,

    /// Maximum placement iterations
    #[arg(long, default_value = "1", value_name = "NUM")]
    iterations: usize,

    /// Enable verbose logging
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Enable plotting
    #[arg(long, default_value = "true")]
    plotting: bool,

    /// Scale factor for fixed macro density (0.0-1.0, lower fills gaps between macros)
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    macro_density_scale: f32,

    /// Radius (in bins) of boosted target density near macros to fill gaps
    #[arg(long, default_value = "5", value_name = "NUM")]
    gap_boost_radius: i32,

    /// Max boost factor for target density near macros (1.0 = no boost)
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    gap_boost_max: f32,

    /// Number of bins per dimension for density grid
    #[arg(long, default_value = "512", value_name = "NUM")]
    num_bins: usize,

    /// Target overflow threshold for convergence (DREAMPlace uses 0.07)
    #[arg(long, default_value = "0.07", value_name = "FLOAT")]
    target_overflow: f32,

    /// Initial position noise as fraction of die size (0.0 = all at center, may need more iterations)
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    init_noise: f32,

    /// Treat all instances as movable (ignore fixed placement from DEF)
    #[arg(long, default_value = "false")]
    unfix_all: bool,

    /// Reposition macros uniformly at init using cluster colors (does not unfix them)
    #[arg(long, default_value = "false")]
    spread_macros: bool,

    /// Use DREAMPlace-style density computation (no target=0 for macros, no movable-target subtraction)
    #[arg(long, default_value = "false")]
    dreamplace: bool,

    /// Use full DIEAREA for placement instead of row-based placement region
    #[arg(long, default_value = "false")]
    full_diearea: bool,

    /// Initial density weight (default 8e-5, increase for mixed-size placement)
    #[arg(long, default_value = "8e-5", value_name = "FLOAT")]
    initial_density_weight: f32,

    /// Export netlist as CSV (net_id,pin_id,instance_id) and exit
    #[arg(long)]
    export_netlist: Option<String>,

    /// CSV file with macro cluster colors (macro_idx,r,g,b) from macro_clusters.py
    #[arg(long)]
    macro_colors: Option<String>,

    /// Save macro positions to CSV after placement (instance_id,x,y)
    #[arg(long, value_name = "PATH")]
    save_macros: Option<String>,

    /// Load macro positions from CSV before placement (instance_id,x,y)
    #[arg(long, value_name = "PATH")]
    load_macros: Option<String>,
}

fn unfix_all_instances(db: &mut reda_db::DB<f32>) {
    let n = db.instances.len();
    db.movable_area = db.movable_area + db.fixed_area;
    db.cell_utilization = db.movable_area / db.diearea.area();
    db.fixed_area = 0.0;
    db.num_movable = n;
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

    let mut db = time_it("Reading DB", || {
        read_db::<f32>(&args.lef, &args.def, args.full_diearea)
    })?;

    if let Some(path) = &args.export_netlist {
        return write_netlist(&db, path);
    }

    // Log fixed instances
    for i in db.num_movable..(db.instances.len() - db.num_terminal) {
        log::debug!(
            "Fixed instance [{}]: pos=({:.1}, {:.1}) size=({:.1} x {:.1})",
            i,
            db.instances.coords.x[i],
            db.instances.coords.y[i],
            db.instances.sizes.w[i],
            db.instances.sizes.h[i]
        );
    }

    if args.unfix_all {
        unfix_all_instances(&mut db);
    }

    let macro_density_scale =
        if (args.unfix_all || args.spread_macros) && args.macro_density_scale == 1.0 {
            0.1
        } else {
            args.macro_density_scale
        };

    let (macro_colors, instance_color_map) = load_macro_colors(&args.macro_colors, &db)?;

    // Spread fixed macros by cluster before creating placement (--spread-macros without --unfix-all)
    if args.spread_macros && !args.unfix_all && !instance_color_map.is_empty() {
        spread_macros(instance_color_map, &mut db)?;
    }

    if let Some(path) = &args.load_macros {
        load_macros(path, &mut db)?
    }

    let mut placement = time_it("Creating Placement", || {
        Placement::new(
            &db,
            macro_density_scale,
            args.gap_boost_radius,
            args.gap_boost_max,
            args.num_bins,
            args.target_overflow,
            args.init_noise,
            args.dreamplace,
            args.initial_density_weight,
            args.spread_macros,
            &macro_colors,
        )
    });
    log::info!("{}", placement);

    draw_placement(placement.db, &placement.ps, &macro_colors, 0);

    force_time_it("Placement", || {
        for i in 0..args.iterations {
            let should_stop = placement.step();
            if i % 50 == 0 && args.plotting {
                draw_placement(placement.db, &placement.ps, &macro_colors, i);
            }
            if should_stop {
                log::info!("Converged at iteration {}", i);
                draw_placement(placement.db, &placement.ps, &macro_colors, i);
                break;
            }
        }
    });

    placement.restore_best();

    draw_placement(placement.db, &placement.ps, &macro_colors, args.iterations);

    if let Some(path) = &args.save_macros {
        save_macros(path, &mut placement)?
    }

    Ok(())
}
