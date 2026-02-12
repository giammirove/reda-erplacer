use crate::{
    irfft2d::irfft_2d,
    utils::{complex_conj, complex_mul, index},
};
use num_traits::Float;
use parking_lot::Mutex;
use rustfft::num_complex::Complex;

pub(crate) fn idct2_fft2_preprocess<T>(
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
            let cond = ((hid != 0) as usize) << 1 | ((wid != 0) as usize);

            match cond {
                0 => {
                    output[0] = Complex::new(input[0], T::zero());

                    let v = input[half_n];
                    let tmp = Complex::new(v, v);
                    output[half_n] = (expk_n[half_n] * tmp).conj();

                    let v = input[index(half_m, 0, n)];
                    let tmp = Complex::new(v, v);
                    output[index(half_m, 0, half_n + 1)] =
                        complex_conj(complex_mul(expk_m[half_m], tmp));

                    let v = input[index(half_m, half_n, n)];
                    let tmp = Complex::new(T::zero(), two * v);
                    output[index(half_m, half_n, half_n + 1)] = complex_conj(complex_mul(
                        complex_mul(expk_m[half_m], expk_n[half_n]),
                        tmp,
                    ));
                }

                1 => {
                    let a = input[wid];
                    let b = input[n - wid];
                    let tmp = Complex::new(a, b);
                    output[wid] = complex_conj(complex_mul(expk_n[wid], tmp));

                    let a = input[index(half_m, wid, n)];
                    let b = input[index(half_m, n - wid, n)];

                    let sum = a + b;
                    let diff = a - b;

                    let tmp = Complex::new(diff, sum);
                    output[index(half_m, wid, half_n + 1)] =
                        complex_conj(complex_mul(complex_mul(expk_m[half_m], expk_n[wid]), tmp));
                }

                2 => {
                    let a = input[index(hid, 0, n)];
                    let b = input[index(m - hid, 0, n)];

                    let up = Complex::new(a, b);
                    let down = Complex::new(b, a);

                    output[index(hid, 0, half_n + 1)] = complex_conj(complex_mul(expk_m[hid], up));
                    output[index(m - hid, 0, half_n + 1)] =
                        complex_conj(complex_mul(expk_m[m - hid], down));

                    let a = input[index(hid, half_n, n)];
                    let b = input[index(m - hid, half_n, n)];

                    let sum = a + b;
                    let diff = a - b;

                    let up = Complex::new(diff, sum);
                    let down = Complex::new(-diff, sum);

                    let w = expk_n[half_n];

                    output[index(hid, half_n, half_n + 1)] =
                        complex_conj(complex_mul(complex_mul(expk_m[hid], w), up));
                    output[index(m - hid, half_n, half_n + 1)] =
                        complex_conj(complex_mul(complex_mul(expk_m[m - hid], w), down));
                }

                3 => {
                    let a = input[index(hid, wid, n)];
                    let b = input[index(hid, n - wid, n)];
                    let c = input[index(m - hid, wid, n)];
                    let d = input[index(m - hid, n - wid, n)];

                    let s1 = a + d;
                    let s2 = b + c;
                    let d1 = a - d;
                    let d2 = c - b;

                    let up = Complex::new(d1, s2);
                    let down = Complex::new(d2, s1);

                    let w = complex_mul(expk_m[hid], expk_n[wid]);
                    let w_sym = complex_mul(expk_m[m - hid], expk_n[wid]);

                    output[index(hid, wid, half_n + 1)] = complex_conj(complex_mul(w, up));
                    output[index(m - hid, wid, half_n + 1)] =
                        complex_conj(complex_mul(w_sym, down));
                }

                _ => unreachable!(),
            }
        }
    }
}

fn idct2_fft2_postprocess<T>(x: &[T], y: &mut [T], m: usize, n: usize)
where
    T: Float,
{
    let mn = T::from(m * n).unwrap();

    for hid in 0..m {
        for wid in 0..n {
            let cond = ((hid < m / 2) as usize) << 1 | ((wid < n / 2) as usize);

            let idx = match cond {
                0 => index(((m - hid) << 1) - 1, ((n - wid) << 1) - 1, n),
                1 => index(((m - hid) << 1) - 1, wid << 1, n),
                2 => index(hid << 1, ((n - wid) << 1) - 1, n),
                3 => index(hid << 1, wid << 1, n),
                _ => unreachable!(),
            };

            y[idx] = x[index(hid, wid, n)] * mn;
        }
    }
}

pub(crate) fn idct2_fft2_forward<T>(
    x: &[T],
    expk_m: &[Complex<T>],
    expk_n: &[Complex<T>],
    out: &mut [T],
    buf: &mut [Complex<T>],
    m: usize,
    n: usize,
) where
    T: Float + rustfft::FftNum,
{
    // Preprocess step
    let mut fft_out = vec![T::zero(); m * n];
    idct2_fft2_preprocess(x, buf, m, n, expk_m, expk_n);

    // Apply 2D inverse real FFT
    let mutex_buf: Mutex<&mut [Complex<T>]> = Mutex::new(buf);
    let mutex_fft_out: Mutex<&mut [T]> = Mutex::new(&mut fft_out);
    irfft_2d(&mutex_buf, &mutex_fft_out, m, n);

    // Postprocess step
    idct2_fft2_postprocess(&fft_out, out, m, n);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_it;
    use crate::utils::make_expk;
    use crate::utils::read_complex_matrix_from_file;
    use crate::utils::read_matrix_from_file;
    use crate::{assert_approx_eq, assert_approx_eq_complex};
    use num_traits::Zero;

    #[test]
    fn test_idct2_preprocess_512x512() {
        let (dreamplace_input, m, n) =
            read_matrix_from_file::<f64>("./tests/fft2/idct2/idct2_input_512x512.txt").unwrap();
        let (dreamplace_pre, m2, n2) =
            read_complex_matrix_from_file::<f64>("./tests/fft2/idct2/idct2_pre_512x512.txt")
                .unwrap();

        assert_eq!(m2, m);
        assert_eq!(n2, n / 2 + 1);

        let expk_m = make_expk(m);
        let expk_n = make_expk(n);
        let mut out = vec![Complex::zero(); m * n];
        time_it("idct2_fft2_preprocess", || {
            idct2_fft2_preprocess(&dreamplace_input, &mut out, m, n, &expk_m, &expk_n);
        });

        for i in 0..dreamplace_pre.len() {
            let o = out[i];
            let d = dreamplace_pre[i];
            assert_approx_eq_complex!(o, d, 1e-3);
        }
    }

    #[test]
    fn test_idct2_irfft2_512x512() {
        let (mut dreamplace_pre, mc, nc) =
            read_complex_matrix_from_file::<f32>("./tests/fft2/idct2/idct2_pre_512x512.txt")
                .unwrap();
        let (dreamplace_fft, m, n) =
            read_matrix_from_file::<f32>("./tests/fft2/idct2/idct2_fft_512x512.txt").unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);
        assert_eq!(mc, m);
        assert_eq!(nc, n / 2 + 1);

        let mutex_dreamplace_pre: Mutex<&mut [Complex<f32>]> = Mutex::new(&mut dreamplace_pre);

        let mut out = vec![0.0; m * n];
        let mutex_fft_out: Mutex<&mut [f32]> = Mutex::new(&mut out);
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
    fn test_idct2_postprocess_512x512() {
        let (dreamplace_fft, m, n) =
            read_matrix_from_file::<f64>("./tests/fft2/idct2/idct2_fft_512x512.txt").unwrap();
        let (dreamplace_post, m2, n2) =
            read_matrix_from_file::<f64>("./tests/fft2/idct2/idct2_post_512x512.txt").unwrap();

        assert_eq!(m2, m);
        assert_eq!(n2, n);

        let mut out = vec![0.0; m * n];
        time_it("idct2_fft2_preprocess", || {
            idct2_fft2_postprocess(&dreamplace_fft, &mut out, m, n);
        });

        for i in 0..dreamplace_post.len() {
            let o = out[i];
            let d = dreamplace_post[i];
            assert_approx_eq!(o, d, 1e-5);
        }
    }

    #[test]
    fn test_idct2_512x512() {
        let (dreamplace_input, m, n) =
            read_matrix_from_file::<f32>("./tests/fft2/idct2/idct2_input_512x512.txt").unwrap();
        let (dreamplace_post, _, _) =
            read_matrix_from_file::<f32>("./tests/fft2/idct2/idct2_post_512x512.txt").unwrap();

        let expk_m = make_expk(m);
        let expk_n = make_expk(n);
        let mut buf = make_expk(m * n);
        let mut out = vec![0.0; m * n];
        time_it("idct2_fft2_forward", || {
            idct2_fft2_forward(
                &dreamplace_input,
                &expk_m,
                &expk_n,
                &mut out,
                &mut buf,
                m,
                n,
            );
        });

        for i in 0..m * n {
            let o = out[i];
            let d = dreamplace_post[i];
            assert_approx_eq!(o, d, 1e-2);
        }
    }
}
