use eyre::Result;
use num_traits::Float;
use reda_db::{Numeric, DB};
use rustfft::num_complex::Complex;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use crate::placement::Placement;

#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        assert_approx_eq!($a, $b, 1e-3)
    };
    ($a:expr, $b:expr, $eps:expr) => {{
        let diff = ($a - $b).abs();
        assert!(
            diff < $eps,
            "Assertion failed: |{} - {}| = {} >= {}",
            $a,
            $b,
            diff,
            $eps
        );
    }};
}

#[macro_export]
macro_rules! assert_vec_approx_eq {
    ($a:expr, $b:expr) => {
        assert_vec_approx_eq!($a, $b, 1e-3)
    };
    ($a:expr, $b:expr, $eps:expr) => {{
        let a = &$a;
        let b = &$b;

        assert!(
            a.len() == b.len(),
            "Length mismatch: {} vs {}",
            a.len(),
            b.len()
        );

        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (*va - *vb).abs();
            assert!(
                diff < $eps,
                "Assertion failed at index {}: |{} - {}| = {} >= {}",
                i,
                va,
                vb,
                diff,
                $eps
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_complex_vec_approx_eq {
    ($a:expr, $b:expr) => {
        assert_vec_approx_eq!($a, $b, 1e-3)
    };
    ($a:expr, $b:expr, $eps:expr) => {{
        let a = &$a;
        let b = &$b;

        assert!(
            a.len() == b.len(),
            "Length mismatch: {} vs {}",
            a.len(),
            b.len()
        );

        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (*va - *vb).norm();
            assert!(
                diff < $eps,
                "Assertion failed at index {}: |{} - {}| = {} >= {}",
                i,
                va,
                vb,
                diff,
                $eps
            );
        }
    }};
}

#[macro_export]
macro_rules! assert_approx_eq_complex {
    ($a:expr, $b:expr) => {
        assert_approx_eq_complex!($a, $b, 1e-3)
    };
    ($a:expr, $b:expr, $eps:expr) => {{
        let diff = ($a - $b).norm(); // magnitude of difference
        assert!(
            diff < $eps,
            "Assertion failed: |{} - {}| = {} >= {}",
            $a,
            $b,
            diff,
            $eps
        );
    }};
}

pub(crate) fn time_it<F, R>(label: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    log::debug!("{:<10} took {:?}", label, start.elapsed());
    result
}

pub(crate) fn force_time_it<F, R>(label: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    log::info!("{:<10} took {:?}", label, start.elapsed());
    result
}

#[allow(dead_code)]
pub(crate) fn print_matrix_to_file<T: std::fmt::Debug>(
    matrix: &[T],
    m: usize,
    n: usize,
    filename: &str,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;

    for i in 0..m {
        for j in 0..n {
            write!(file, "{:12.6?} ", matrix[i * n + j])?;
        }
        writeln!(file)?;
    }

    Ok(())
}

#[allow(dead_code)]
pub(crate) fn read_complex_matrix_from_file<T>(
    filename: &str,
) -> Result<(Vec<Complex<T>>, usize, usize), String>
where
    T: Float + rustfft::FftNum,
{
    let file = File::open(filename).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut data: Vec<Complex<T>> = Vec::new();
    let mut n_cols = 0;
    let mut n_rows = 0;

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let values: Vec<T> = line
            .split_whitespace()
            .map(|s| {
                T::from(s.parse::<f64>().map_err(|e| e.to_string())?)
                    .ok_or_else(|| "Conversion failed".to_string())
            })
            .collect::<Result<_, _>>()?;

        if !values.is_empty() {
            if values.len() % 2 != 0 {
                return Err(format!(
                    "Line {} has an odd number of values, expected real/imag pairs",
                    n_rows + 1
                ));
            }

            // Convert pairs (re, im) into Complex<T>
            let mut row_complex = Vec::with_capacity(values.len() / 2);
            for chunk in values.chunks_exact(2) {
                row_complex.push(Complex::new(chunk[0], chunk[1]));
            }

            // Set n_cols based on number of complex numbers per row
            if n_cols == 0 {
                n_cols = row_complex.len();
            } else if n_cols != row_complex.len() {
                return Err(format!(
                    "Inconsistent number of complex columns in line {}",
                    n_rows + 1
                ));
            }

            data.extend(row_complex);
            n_rows += 1;
        }
    }

    Ok((data, n_rows, n_cols))
}

#[allow(dead_code)]
pub(crate) fn read_matrix_from_file<T>(filename: &str) -> Result<(Vec<T>, usize, usize), String>
where
    T: Float + rustfft::FftNum,
{
    let file = File::open(filename).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    let mut n_cols = 0;
    let mut n_rows = 0;

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let values: Vec<T> = line
            .split_whitespace()
            .map(|s| T::from(s.parse::<f64>().unwrap()).unwrap())
            .collect();

        if !values.is_empty() {
            if n_cols == 0 {
                n_cols = values.len();
            }
            data.extend(values);
            n_rows += 1;
        }
    }

    Ok((data, n_rows, n_cols))
}

pub(crate) fn make_expk<T>(n: usize) -> Vec<Complex<T>>
where
    T: Float,
{
    let two_n = T::from(2 * n).unwrap();
    let pi = T::from(PI).unwrap();

    let mut expk = Vec::with_capacity(n);

    for k in 0..n {
        let pik_by_2n = T::from(k).unwrap() * pi / two_n;
        let cos_val = pik_by_2n.cos();
        let neg_sin_val = -pik_by_2n.sin();
        expk.push(Complex::new(cos_val, neg_sin_val));
    }

    expk
}

#[inline(always)]
pub(crate) fn index(hid: usize, wid: usize, n: usize) -> usize {
    hid * n + wid
}

#[inline(always)]
pub(crate) fn complex_mul<T: Float>(a: Complex<T>, b: Complex<T>) -> Complex<T> {
    Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline(always)]
pub(crate) fn complex_conj<T: Float>(a: Complex<T>) -> Complex<T> {
    Complex::new(a.re, -a.im)
}

pub(crate) fn write_netlist<T: Numeric>(db: &DB<T>, path: &str) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "net_id,pin_id,instance_id")?;
    for (net_id, net) in db.netlist.nets.iter().enumerate() {
        for &pin_id in &net.pin_ids {
            let instance_id = db.netlist.pin_2_macro[pin_id];
            writeln!(f, "{},{},{}", net_id, pin_id, instance_id)?;
        }
    }
    log::info!("Exported netlist to {}", path);
    Ok(())
}

fn get_macro_threshold<T: Numeric>(db: &DB<T>) -> Result<T> {
    let mut sorted: Vec<T> = db.instances.areas[..db.num_movable].to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(sorted[sorted.len() / 2] * T::from(10.0).unwrap())
}

pub(crate) fn load_macro_colors<T: Numeric>(
    macro_colors: &Option<String>,
    db: &DB<T>,
) -> Result<(Vec<[u8; 3]>, std::collections::HashMap<usize, [u8; 3]>)> {
    if let Some(path) = &macro_colors {
        // Build instance_id -> macro_idx mapping using same threshold as drawing.rs
        let num_movable = db.num_movable;
        let macro_threshold = get_macro_threshold(db)?;
        let mut macro_idx_map: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut midx = 0usize;
        for i in 0..db.instances.len() {
            let is_macro = i >= num_movable || db.instances.areas[i] > macro_threshold;
            if is_macro {
                macro_idx_map.insert(i, midx);
                midx += 1;
            }
        }

        let content = std::fs::read_to_string(path)?;
        let mut colors = vec![[255u8, 0, 0]; midx];
        let mut instance_color_map: std::collections::HashMap<usize, [u8; 3]> =
            std::collections::HashMap::new();
        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() == 4 {
                if let (Ok(instance_id), Ok(r), Ok(g), Ok(b)) = (
                    parts[0].parse::<usize>(),
                    parts[1].parse::<u8>(),
                    parts[2].parse::<u8>(),
                    parts[3].parse::<u8>(),
                ) {
                    if let Some(&midx) = macro_idx_map.get(&instance_id) {
                        colors[midx] = [r, g, b];
                    }
                    instance_color_map.insert(instance_id, [r, g, b]);
                }
            }
        }
        log::info!("Loaded macro colors from {}", path);
        Ok((colors, instance_color_map))
    } else {
        Ok((vec![], std::collections::HashMap::<usize, [u8; 3]>::new()))
    }
}

pub(crate) fn spread_macros<T: Numeric>(
    instance_color_map: std::collections::HashMap<usize, [u8; 3]>,
    db: &mut DB<T>,
) -> Result<()> {
    // Group instance_ids by color
    let mut cluster_map: std::collections::HashMap<[u8; 3], Vec<usize>> =
        std::collections::HashMap::new();
    for (&iid, &color) in &instance_color_map {
        cluster_map.entry(color).or_default().push(iid);
    }
    let mut clusters: Vec<Vec<usize>> = cluster_map.into_values().collect();
    clusters.sort_by_key(|c| c[0]);
    let num_clusters = clusters.len();
    let grid_cols = (num_clusters as f64).sqrt().ceil() as usize;
    let grid_rows = (num_clusters + grid_cols - 1) / grid_cols;
    let dw = db.diearea.width();
    let dh = db.diearea.height();
    let region_w = dw / T::from(grid_cols as f32).unwrap();
    let region_h = dh / T::from(grid_rows as f32).unwrap();
    let zero = T::zero();
    let two = T::one() + T::one();
    let onedotone = T::from(1.1).unwrap();
    let offset_x = zero;
    let offset_y = zero;
    for (ci, cluster) in clusters.iter().enumerate() {
        let col = T::from(ci % grid_cols).unwrap();
        let row = T::from(ci / grid_cols).unwrap();
        let rx = offset_x + region_w * col + region_w / two;
        let ry = offset_y + region_h * row + region_h / two;
        let n = cluster.len();
        let sub_cols = (n as f64).sqrt().ceil() as usize;
        let sub_rows = (n + sub_cols - 1) / sub_cols;
        // Use actual macro size for tight packing
        let max_w = cluster
            .iter()
            .map(|&i| db.instances.sizes.w[i])
            .fold(zero, T::max);
        let max_h = cluster
            .iter()
            .map(|&i| db.instances.sizes.h[i])
            .fold(zero, T::max);
        let cell_w = max_w * onedotone; // 10% gap
        let cell_h = max_h * onedotone;
        // Center the sub-grid in the region
        let grid_w = cell_w * T::from(sub_cols as f32).unwrap();
        let grid_h = cell_h * T::from(sub_rows as f32).unwrap();
        let ox = rx - grid_w / two;
        let oy = ry - grid_h / two;
        for (k, &i) in cluster.iter().enumerate() {
            let sc = T::from(k % sub_cols).unwrap();
            let sr = T::from(k / sub_cols).unwrap();
            let w = db.instances.sizes.w[i];
            let h = db.instances.sizes.h[i];
            db.instances.coords.x[i] =
                (ox + cell_w * sc + cell_w / two - w / two).clamp(zero, dw - w);
            db.instances.coords.y[i] =
                (oy + cell_h * sr + cell_h / two - h / two).clamp(zero, dh - h);
        }
    }
    log::info!(
        "Spread {} fixed macros into {} clusters",
        instance_color_map.len(),
        num_clusters
    );

    Ok(())
}

pub(crate) fn load_macros<T: Numeric>(path: &str, db: &mut DB<T>) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    for line in content.lines().skip(1) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(i), Ok(x), Ok(y)) = (
                parts[0].parse::<usize>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
            ) {
                if i < db.instances.coords.x.len() {
                    db.instances.coords.x[i] = T::from(x).unwrap();
                    db.instances.coords.y[i] = T::from(y).unwrap();
                }
            }
        }
    }
    log::info!("Loaded macro positions from {}", path);
    Ok(())
}

pub(crate) fn save_macros<T: Numeric>(path: &str, placement: &mut Placement<T>) -> Result<()> {
    let db = placement.db;
    let num_movable = db.num_movable;
    let macro_threshold = {
        let mut sorted: Vec<T> = db.instances.areas[..num_movable].to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2] * T::from(10.0).unwrap()
    };
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "instance_id,x,y")?;
    for i in 0..db.instances.len() {
        let is_macro = i >= num_movable || db.instances.areas[i] > macro_threshold;
        if is_macro {
            let x = if i < num_movable {
                placement.ps.instances.x[i]
            } else {
                db.instances.coords.x[i]
            };
            let y = if i < num_movable {
                placement.ps.instances.y[i]
            } else {
                db.instances.coords.y[i]
            };
            writeln!(f, "{},{:?},{:?}", i, x, y)?;
        }
    }
    log::info!("Saved macro positions to {}", path);
    Ok(())
}
