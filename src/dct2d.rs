use core::fmt;
use num_traits::{Float, Zero};
use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::Numeric;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub(crate) struct DCT2D<T>
where
    T: Numeric + rustfft::FftNum,
{
    pub(crate) preprocessed: Vec<Complex<T>>,
    pub(crate) rfft_2d_output: Vec<Complex<T>>,
}
impl<T> fmt::Debug for DCT2D<T>
where
    T: Numeric + rustfft::FftNum,
{
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}
impl<T> DCT2D<T>
where
    T: Numeric + rustfft::FftNum,
{
    pub(crate) fn new(m: usize, n: usize) -> Self {
        let half_n = n / 2 + 1;
        let preprocessed = vec![Complex::zero(); 2 * m * half_n];
        let rfft_2d_output = vec![Complex::zero(); m * half_n];

        Self {
            preprocessed,
            rfft_2d_output,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.preprocessed
            .par_iter_mut()
            .for_each(|x| *x = Complex::zero());
    }
}

#[inline(always)]
fn dct2d_preprocess<T>(x: &[T], y: &mut [Complex<T>], m: usize, n: usize)
where
    T: Numeric,
{
    let zero = T::zero();
    let half_n = n / 2;

    for hid in 0..m {
        let dst_row = if (hid & 1) == 0 {
            hid
        } else {
            2 * m - (hid + 1)
        };
        let dst_row_offset = dst_row * half_n;
        let src_row_offset = hid * n;

        // Process even columns
        for i in 0..half_n {
            let src_idx = src_row_offset + (i << 1);
            let dst_idx = dst_row_offset + i;
            unsafe {
                *y.get_unchecked_mut(dst_idx) = Complex::new(*x.get_unchecked(src_idx), zero);
            }
        }

        // Process odd columns
        for i in 0..half_n {
            let src_idx = src_row_offset + (i << 1) + 1;
            let dst_idx = dst_row_offset + (n - i - 1);
            unsafe {
                *y.get_unchecked_mut(dst_idx) = Complex::new(*x.get_unchecked(src_idx), zero);
            }
        }
    }
}

#[inline(always)]
fn real_mul<T: Float>(a: Complex<T>, b: Complex<T>) -> T {
    a.re * b.re - a.im * b.im
}

#[inline(always)]
fn imag_mul<T: Float>(a: Complex<T>, b: Complex<T>) -> T {
    a.re * b.im + a.im * b.re
}

// use of unchecked is justified by the fact that this
// is a fixed state machine where the input does
// not change the behaviour
// and functionality is verified using unit tests
// TODO: for some reason using Mutex is faster than not Mutex (single-thread execution)
unsafe fn dct2d_postprocess<T>(
    v: &[Complex<T>],
    y_mutex: &Mutex<&mut [T]>,
    m: usize,
    n: usize,
    expk_m: &[Complex<T>],
    expk_n: &[Complex<T>],
) where
    T: Numeric + rustfft::FftNum,
{
    let mn = T::from(m * n).unwrap();
    let half_m = m / 2;
    let half_n = n / 2;

    let four_over_mn = T::from(4.0).unwrap() / mn;
    let two_over_mn = T::from(2.0).unwrap() / mn;

    let v_stride = half_n + 1;

    let expkm_halfm = *expk_m.get_unchecked(half_m);
    let expkn_halfn = *expk_n.get_unchecked(half_n);

    // DC corner

    let y: &mut [T] = &mut *y_mutex.data_ptr();

    *y.get_unchecked_mut(0) = v.get_unchecked(0).re * four_over_mn;

    *y.get_unchecked_mut(half_n) = real_mul(expkn_halfn, *v.get_unchecked(half_n)) * four_over_mn;

    let row_halfm = half_m * n;

    *y.get_unchecked_mut(row_halfm) =
        expkm_halfm.re * v.get_unchecked(half_m * v_stride).re * four_over_mn;

    *y.get_unchecked_mut(row_halfm + half_n) = expkm_halfm.re
        * real_mul(expkn_halfn, *v.get_unchecked(half_m * v_stride + half_n))
        * four_over_mn;

    // top edge: hid = 0

    for wid in 1..half_n {
        let y: &mut [T] = &mut *y_mutex.data_ptr();

        let tmp = *v.get_unchecked(wid);
        let expkn = *expk_n.get_unchecked(wid);

        *y.get_unchecked_mut(wid) = real_mul(expkn, tmp) * four_over_mn;
        *y.get_unchecked_mut(n - wid) = -imag_mul(expkn, tmp) * four_over_mn;

        let tmp = *v.get_unchecked(half_m * v_stride + wid);

        *y.get_unchecked_mut(row_halfm + wid) =
            expkm_halfm.re * real_mul(expkn, tmp) * four_over_mn;

        *y.get_unchecked_mut(row_halfm + (n - wid)) =
            -(expkm_halfm.re * imag_mul(expkn, tmp) * four_over_mn);
    }

    // left edge: wid = 0

    for hid in 1..half_m {
        let y: &mut [T] = &mut *y_mutex.data_ptr();

        let row_up = hid * n;
        let row_down = (m - hid) * n;

        let k = hid * v_stride;
        let km = (m - hid) * v_stride;

        let tmp1 = v.get_unchecked(k);
        let tmp2 = v.get_unchecked(km);

        let expm = *expk_m.get_unchecked(hid);

        let sum_re = tmp1.re + tmp2.re;
        let diff_im = tmp2.im - tmp1.im;

        let up = expm.re * sum_re + expm.im * diff_im;

        let down = -(expm.im * sum_re) + expm.re * diff_im;

        *y.get_unchecked_mut(row_up) = up * two_over_mn;
        *y.get_unchecked_mut(row_down) = down * two_over_mn;

        let v1 = v.get_unchecked(k + half_n);
        let v2 = v.get_unchecked(km + half_n);

        let tmp1 = v1 + v2;
        let tmp2 = v1 - v2;

        let up = Complex::new(
            expm.re * tmp1.re - expm.im * tmp2.im,
            expm.re * tmp1.im + expm.im * tmp2.re,
        );

        let down = Complex::new(
            -(expm.im * tmp1.re) - expm.re * tmp2.im,
            -(expm.im * tmp1.im) + expm.re * tmp2.re,
        );

        *y.get_unchecked_mut(row_up + half_n) = real_mul(expkn_halfn, up) * two_over_mn;
        *y.get_unchecked_mut(row_down + half_n) = real_mul(expkn_halfn, down) * two_over_mn;
    }

    // interior

    let v_stride = half_n + 1;

    for hid in 1..half_m {
        let row_up = hid * n;
        let row_down = (m - hid) * n;
        let y: &mut [T] = &mut *y_mutex.data_ptr();
        let y_base = y.as_mut_ptr();
        let row_up_ptr = y_base.add(row_up);
        let row_down_ptr = y_base.add(row_down);

        let expm = expk_m.get_unchecked(hid);
        let er = expm.re;
        let ei = expm.im;

        let v_up_base = hid * v_stride;
        let v_down_base = (m - hid) * v_stride;
        let v_ptr = v.as_ptr();
        let expn_ptr = expk_n.as_ptr();

        for wid in 1..half_n {
            let a = *v_ptr.add(v_up_base + wid);
            let b = *v_ptr.add(v_down_base + wid);

            let s_re = a.re + b.re;
            let s_im = a.im + b.im;
            let d_re = a.re - b.re;
            let d_im = a.im - b.im;

            let up_re = er * s_re - ei * d_im;
            let up_im = er * s_im + ei * d_re;
            let dn_re = -ei * s_re - er * d_im;
            let dn_im = -ei * s_im + er * d_re;

            let expn = *expn_ptr.add(wid);
            let sr = expn.re * two_over_mn;
            let si = expn.im * two_over_mn;

            let r_up = sr * up_re - si * up_im;
            let i_up = sr * up_im + si * up_re;
            let r_dn = sr * dn_re - si * dn_im;
            let i_dn = sr * dn_im + si * dn_re;

            let dst = n - wid;
            *row_up_ptr.add(wid) = r_up;
            *row_down_ptr.add(wid) = r_dn;
            *row_up_ptr.add(dst) = -i_up;
            *row_down_ptr.add(dst) = -i_dn;
        }
    }
}

pub(crate) fn dct2_fft2_forward<T>(
    x: &[T],
    expk_m: &[Complex<T>],
    expk_n: &[Complex<T>],
    dct2d: &mut DCT2D<T>,
    mut out: &mut [T],
    m: usize,
    n: usize,
) where
    T: Numeric + rustfft::FftNum,
{
    dct2d.reset();

    // Step 1: Preprocess
    dct2d_preprocess(x, &mut dct2d.preprocessed, m, n);

    // Step 2: Perform 2D real FFT
    let out_mutex: Mutex<&mut [Complex<T>]> = Mutex::new(&mut dct2d.rfft_2d_output);
    unsafe { rfft_2d(&dct2d.preprocessed, &out_mutex, m, n) };

    // Step 3: Postprocess
    let out_mutex: Mutex<&mut [T]> = Mutex::new(&mut out);
    unsafe { dct2d_postprocess(&dct2d.rfft_2d_output, &out_mutex, m, n, expk_m, expk_n) };
}

unsafe fn rfft_2d<T>(
    input: &[Complex<T>],
    output_mutex: &Mutex<&mut [Complex<T>]>,
    m: usize,
    n: usize,
) where
    T: Numeric + rustfft::FftNum,
{
    let n_freq = n / 2 + 1;

    let mut planner = FftPlanner::<T>::new();
    let fft_row = planner.plan_fft_forward(n);

    (0..m).into_par_iter().for_each_init(
        || {
            (
                vec![Complex::zero(); n],
                vec![Complex::zero(); fft_row.get_inplace_scratch_len()],
            )
        },
        |(row_buf, scratch), i| unsafe {
            let output: &mut [Complex<T>] = &mut *output_mutex.data_ptr();
            let output_ptr = output.as_mut_ptr();
            let input_ptr = input.as_ptr();

            let src = input_ptr.add(i * n);
            let dst = output_ptr.add(i * n_freq);

            std::ptr::copy_nonoverlapping(src, row_buf.as_mut_ptr(), n);

            fft_row.process_with_scratch(row_buf, scratch);

            std::ptr::copy_nonoverlapping(row_buf.as_ptr(), dst, n_freq);
        },
    );

    let fft_col = planner.plan_fft_forward(m);
    (0..n_freq).into_par_iter().for_each_init(
        || {
            (
                vec![Complex::zero(); m],
                vec![Complex::zero(); fft_col.get_inplace_scratch_len()],
            )
        },
        |(col_buf, scratch), j| unsafe {
            let output: &mut [Complex<T>] = &mut *output_mutex.data_ptr();

            for i in 0..m {
                *col_buf.get_unchecked_mut(i) = *output.get_unchecked(i * n_freq + j);
            }

            fft_col.process_with_scratch(col_buf, scratch);

            for i in 0..m {
                *output.get_unchecked_mut(i * n_freq + j) = *col_buf.get_unchecked(i);
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assert_approx_eq, assert_approx_eq_complex, assert_complex_vec_approx_eq,
        assert_vec_approx_eq, time_it,
        utils::{make_expk, read_complex_matrix_from_file, read_matrix_from_file},
    };

    #[test]
    fn test_make_expk_512x512() {
        let (dreamplace_expk_m, m, _) =
            read_complex_matrix_from_file::<f32>("./tests/fft2/expk_m.txt").unwrap();
        let (dreamplace_expk_n, n, _) =
            read_complex_matrix_from_file::<f32>("./tests/fft2/expk_n.txt").unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);

        let expk_m: Vec<Complex<f32>> = make_expk(m);
        let expk_n: Vec<Complex<f32>> = make_expk(n);

        for i in 0..m {
            let o = expk_m[i];
            let d = dreamplace_expk_m[i];
            assert_approx_eq_complex!(o, d, 1e-6);
        }

        for i in 0..n {
            let o = expk_n[i];
            let d = dreamplace_expk_n[i];
            assert_approx_eq_complex!(o, d, 1e-6);
        }
    }

    #[test]
    fn test_dct2_preprocess_512x512() {
        let (dreamplace_input, m, n) =
            read_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_input_512x512.txt").unwrap();
        let (dreamplace_pre, m2, n2) =
            read_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_pre_512x512.txt").unwrap();

        assert_eq!(m2, m);
        assert_eq!(n2, n);

        let mut out = vec![Complex::zero(); m * n];
        // twice to warm the cache
        for _ in 0..2 {
            out.fill(Complex::zero());
            time_it("dct2_fft2_preprocess", || {
                dct2d_preprocess(&dreamplace_input, &mut out, m, n);
            });
        }

        for i in 0..dreamplace_pre.len() {
            let o = out[i].re;
            let d = dreamplace_pre[i];
            assert_approx_eq!(o, d, 1e-1);
        }
    }

    #[test]
    fn test_dct2_rfft2_512x512() {
        let (dreamplace_pre, m, n) =
            read_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_pre_512x512.txt").unwrap();
        let (dreamplace_fft, m2, n2) =
            read_complex_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_fft_512x512.txt").unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);
        assert_eq!(m2, 512);
        assert_eq!(n2, 257);

        let mut dreamplace_pre_complex = vec![Complex::zero(); m * n];
        for i in 0..m * n {
            dreamplace_pre_complex[i] = Complex::new(dreamplace_pre[i], 0.0);
        }

        let mut out = vec![Complex::zero(); m2 * n2];
        let out_mutex: Mutex<&mut [Complex<f32>]> = Mutex::new(&mut out);
        time_it("dct2_fft2_forward", || unsafe {
            rfft_2d(&dreamplace_pre_complex, &out_mutex, m, n);
        });

        assert_complex_vec_approx_eq!(out, dreamplace_fft, 1e-1);
    }

    #[test]
    fn test_dct2_postprocess_512x512() {
        let (dreamplace_fft, m, n) =
            read_complex_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_fft_512x512.txt").unwrap();
        let (dreamplace_post, m2, n2) =
            read_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_post_512x512.txt").unwrap();

        assert_eq!(m2, 512);
        assert_eq!(n2, 512);

        assert_eq!(m2, m);
        assert_eq!(n2 / 2 + 1, n);

        let expk_m = make_expk(m2);
        let expk_n = make_expk(n2);
        let mut out = vec![0.; m2 * n2];
        let out_mutex: Mutex<&mut [f32]> = Mutex::new(&mut out);
        time_it("dct2_fft2_postprocess", || unsafe {
            dct2d_postprocess(&dreamplace_fft, &out_mutex, m2, n2, &expk_m, &expk_n);
        });

        assert_vec_approx_eq!(out, dreamplace_post, 1e-1);
    }

    #[test]
    fn test_dct2_fft2_512x512() {
        let (dreamplace_density_map, m, n) =
            read_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_input_512x512.txt").unwrap();
        let (dreamplace_post, _, _) =
            read_matrix_from_file::<f32>("./tests/fft2/dct2/dct2_post_512x512.txt").unwrap();

        let mut dct2d = DCT2D::new(m, n);

        let expk_m = make_expk(m);
        let expk_n = make_expk(n);
        let mut out = vec![0.0; m * n];
        time_it("dct2_fft2_forward", || {
            dct2_fft2_forward(
                &dreamplace_density_map,
                &expk_m,
                &expk_n,
                &mut dct2d,
                &mut out,
                m,
                n,
            );
        });

        assert_vec_approx_eq!(out, dreamplace_post, 1e-5);
    }
}
