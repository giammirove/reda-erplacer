use num_traits::Float;
use rustfft::num_complex::Complex;
use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

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
