use crate::{
    irfft2d::irfft_2d,
    utils::{complex_conj, complex_mul, index},
};
use num_traits::{Float, Zero};
use parking_lot::Mutex;
use rustfft::num_complex::Complex;

fn idxst_idct_preprocess<T>(
    input: &[T],
    output: &mut [Complex<T>],
    m: usize,
    n: usize,
    expk_m: &[Complex<T>],
    expk_n: &[Complex<T>],
) where
    T: Float,
{
    let half_m = m / 2;
    let half_n = n / 2;
    let two = T::one() + T::one();

    for hid in 0..half_m {
        for wid in 0..half_n {
            let cond = ((hid != 0) as usize) << 1 | (wid != 0) as usize;

            match cond {
                0 => {
                    output[0] = Complex::zero();
                    output[half_n] = Complex::zero();

                    let a = input[index(half_m, 0, n)];
                    let tmp = Complex::new(a, a);

                    let v = complex_mul(expk_m[half_m], tmp);
                    output[index(half_m, 0, half_n + 1)] = complex_conj(v);

                    let b = input[index(half_m, half_n, n)];
                    let tmp = Complex::new(T::zero(), two * b);

                    let ab = complex_mul(expk_m[half_m], expk_n[half_n]);
                    output[index(half_m, half_n, half_n + 1)] = complex_conj(complex_mul(ab, tmp));
                }

                1 => {
                    output[wid] = Complex::zero();

                    let a = input[index(half_m, wid, n)];
                    let b = input[index(half_m, n - wid, n)];

                    let sum = a + b;
                    let diff = a - b;

                    let tmp = Complex::new(diff, sum);

                    let ab = complex_mul(expk_m[half_m], expk_n[wid]);
                    output[index(half_m, wid, half_n + 1)] = complex_conj(complex_mul(ab, tmp));
                }

                2 => {
                    let a = input[index(m - hid, 0, n)];
                    let b = input[index(hid, 0, n)];

                    let up = Complex::new(a, b);
                    let down = Complex::new(b, a);

                    output[index(hid, 0, half_n + 1)] = complex_conj(complex_mul(expk_m[hid], up));
                    output[index(m - hid, 0, half_n + 1)] =
                        complex_conj(complex_mul(expk_m[m - hid], down));

                    let a = input[index(m - hid, half_n, n)];
                    let b = input[index(hid, half_n, n)];

                    let sum = a + b;
                    let diff = a - b;

                    let up = Complex::new(diff, sum);
                    let down = Complex::new(-diff, sum);

                    let em = complex_mul(expk_m[hid], expk_n[half_n]);
                    let em2 = complex_mul(expk_m[m - hid], expk_n[half_n]);

                    output[index(hid, half_n, half_n + 1)] = complex_conj(complex_mul(em, up));
                    output[index(m - hid, half_n, half_n + 1)] =
                        complex_conj(complex_mul(em2, down));
                }

                3 => {
                    let a = input[index(m - hid, wid, n)];
                    let b = input[index(m - hid, n - wid, n)];
                    let c = input[index(hid, wid, n)];
                    let d = input[index(hid, n - wid, n)];

                    let s1 = c + b;
                    let d1 = a - d;

                    let s2 = a + d;
                    let d2 = c - b;

                    let up = Complex::new(d1, s1);
                    let down = Complex::new(d2, s2);

                    let em = complex_mul(expk_m[hid], expk_n[wid]);
                    let em2 = complex_mul(expk_m[m - hid], expk_n[wid]);

                    output[index(hid, wid, half_n + 1)] = complex_conj(complex_mul(em, up));
                    output[index(m - hid, wid, half_n + 1)] = complex_conj(complex_mul(em2, down));
                }

                _ => unreachable!(),
            }
        }
    }
}

fn idxst_idct_postprocess<T>(x: &[T], y: &mut [T], m: usize, n: usize)
where
    T: Float,
{
    let mn = T::from(m * n).unwrap();

    for hid in 0..m {
        for wid in 0..n {
            let cond = ((hid < m / 2) as usize) << 1 | (wid < n / 2) as usize;
            match cond {
                0 => {
                    // hid >= M/2 && wid >= N/2
                    let idx = index(((m - hid) << 1) - 1, ((n - wid) << 1) - 1, n);
                    y[idx] = -x[index(hid, wid, n)] * mn;
                    idx
                }
                1 => {
                    // hid >= M/2 && wid < N/2
                    let idx = index(((m - hid) << 1) - 1, wid << 1, n);
                    y[idx] = -x[index(hid, wid, n)] * mn;
                    idx
                }
                2 => {
                    // hid < M/2 && wid >= N/2
                    let idx = index(hid << 1, ((n - wid) << 1) - 1, n);
                    y[idx] = x[index(hid, wid, n)] * mn;
                    idx
                }
                3 => {
                    // hid < M/2 && wid < N/2
                    let idx = index(hid << 1, wid << 1, n);
                    y[idx] = x[index(hid, wid, n)] * mn;
                    idx
                }
                _ => unreachable!(),
            };
        }
    }
}

pub(crate) fn idxst_idct_forward<T>(
    x: &[T],
    expk_m: &[Complex<T>],
    expk_n: &[Complex<T>],
    out: &mut [T],
    m: usize,
    n: usize,
) where
    T: Float + rustfft::FftNum,
{
    let half_n = n / 2;

    // Step 1: Preprocess
    let mut buf = vec![Complex::zero(); m * (half_n + 1)];
    let mut fft_out = vec![T::zero(); m * n];
    idxst_idct_preprocess(x, &mut buf, m, n, expk_m, expk_n);

    // Step 2: Perform 2D inverse real FFT
    let mutex_buf: Mutex<&mut [Complex<T>]> = Mutex::new(&mut buf);
    let mutex_fft_out: Mutex<&mut [T]> = Mutex::new(&mut fft_out);
    irfft_2d(&mutex_buf, &mutex_fft_out, m, n);

    // Step 3: Postprocess
    idxst_idct_postprocess(&fft_out, out, m, n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_it;
    use crate::utils::make_expk;
    use crate::utils::read_complex_matrix_from_file;
    use crate::utils::read_matrix_from_file;
    use crate::{assert_approx_eq, assert_approx_eq_complex};

    // ============================================================================================================

    #[test]
    fn test_idxst_idct_preprocess_512x512() {
        let (dreamplace_input, m, n) =
            read_matrix_from_file::<f64>("./tests/fft2/idxst_idct/idxst_idct_input_512x512.txt")
                .unwrap();
        let (dreamplace_pre, m2, n2) = read_complex_matrix_from_file::<f64>(
            "./tests/fft2/idxst_idct/idxst_idct_pre_512x512.txt",
        )
        .unwrap();

        assert_eq!(m2, m);
        assert_eq!(n2, n / 2 + 1);

        let expk_m = make_expk(m);
        let expk_n = make_expk(n);
        let mut out = vec![Complex::zero(); m * n];
        time_it("idxst_idct_preprocess", || {
            idxst_idct_preprocess(&dreamplace_input, &mut out, m, n, &expk_m, &expk_n);
        });

        for i in 0..dreamplace_pre.len() {
            let o = out[i];
            let d = dreamplace_pre[i];
            assert_approx_eq_complex!(o, d, 1e-5);
        }
    }

    #[test]
    fn test_idxst_idct_irfft2_512x512() {
        let (mut dreamplace_pre, mc, nc) = read_complex_matrix_from_file::<f64>(
            "./tests/fft2/idxst_idct/idxst_idct_pre_512x512.txt",
        )
        .unwrap();
        let (dreamplace_fft, m, n) =
            read_matrix_from_file::<f64>("./tests/fft2/idxst_idct/idxst_idct_fft_512x512.txt")
                .unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);
        assert_eq!(mc, m);
        assert_eq!(nc, n / 2 + 1);

        let mutex_dreamplace_pre: Mutex<&mut [Complex<f64>]> = Mutex::new(&mut dreamplace_pre);

        let mut out = vec![0.0; m * n];
        let mutex_fft_out: Mutex<&mut [f64]> = Mutex::new(&mut out);
        time_it("irfft_2d", || {
            irfft_2d(&mutex_dreamplace_pre, &mutex_fft_out, m, n)
        });

        for i in 0..out.len() {
            let o = out[i];
            let d = dreamplace_fft[i];
            assert_approx_eq!(o, d, 1e-5);
        }
    }

    #[test]
    fn test_idxst_idct_postprocess_512x512() {
        let (dreamplace_fft, m, n) =
            read_matrix_from_file::<f64>("./tests/fft2/idxst_idct/idxst_idct_fft_512x512.txt")
                .unwrap();
        let (dreamplace_post, m2, n2) =
            read_matrix_from_file::<f64>("./tests/fft2/idxst_idct/idxst_idct_post_512x512.txt")
                .unwrap();

        assert_eq!(m, m2);
        assert_eq!(n, n2);

        let mut out = vec![0.0; m * n];
        time_it("idxst_idct_postprocess", || {
            idxst_idct_postprocess(&dreamplace_fft, &mut out, m, n);
        });

        for i in 0..dreamplace_post.len() {
            let o = out[i];
            let d = dreamplace_post[i];
            assert_approx_eq!(o, d, 1e-5);
        }
    }

    #[test]
    fn test_idxst_idct_512x512() {
        let (dreamplace_input, m, n) =
            read_matrix_from_file::<f64>("./tests/fft2/idxst_idct/idxst_idct_input_512x512.txt")
                .unwrap();
        let (dreamplace_post, _, _) =
            read_matrix_from_file::<f64>("./tests/fft2/idxst_idct/idxst_idct_post_512x512.txt")
                .unwrap();

        let expk_m = make_expk(m);
        let expk_n = make_expk(n);
        let mut out = vec![0.0; m * n];
        time_it("idxst_idct_forward", || {
            idxst_idct_forward(&dreamplace_input, &expk_m, &expk_n, &mut out, m, n);
        });

        for i in 0..m * n {
            let o = out[i];
            let d = dreamplace_post[i];
            assert_approx_eq!(o, d, 1e-3);
        }
    }

    // ============================================================================================================
}
