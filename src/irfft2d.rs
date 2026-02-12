use num_traits::{Float, Zero};
use parking_lot::Mutex;
use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub(crate) fn irfft_2d<T>(
    input_mutex: &Mutex<&mut [Complex<T>]>,
    output_mutex: &Mutex<&mut [T]>,
    m: usize,
    n: usize,
) where
    T: Float + rustfft::FftNum,
{
    let mut planner = FftPlanner::<T>::new();

    let n_freq = n / 2 + 1;

    let ifft_col = planner.plan_fft_inverse(m);
    let norm_m = T::from(m).unwrap();

    (0..n_freq).into_par_iter().for_each_init(
        || {
            (
                vec![Complex::zero(); m],
                vec![Complex::zero(); ifft_col.get_inplace_scratch_len()],
            )
        },
        |(col_buf, scratch), j| unsafe {
            let input_ptr: &mut [Complex<T>] = &mut *input_mutex.data_ptr();
            // gather column
            for i in 0..m {
                *col_buf.get_unchecked_mut(i) = *input_ptr.get_unchecked(i * n_freq + j);
            }

            ifft_col.process_with_scratch(col_buf, scratch);

            // write back (normalized)
            for i in 0..m {
                *input_ptr.get_unchecked_mut(i * n_freq + j) = *col_buf.get_unchecked(i) / norm_m;
            }
        },
    );

    let ifft_row = planner.plan_fft_inverse(n);
    let norm_n = T::from(n).unwrap();

    (0..m).into_par_iter().for_each_init(
        || {
            (
                vec![Complex::<T>::zero(); n],
                vec![Complex::<T>::zero(); ifft_row.get_inplace_scratch_len()],
            )
        },
        |(row_buf, scratch), i| unsafe {
            let row_freq = &mut *input_mutex.data_ptr();

            // copy onesided spectrum
            for j in 0..n_freq {
                *row_buf.get_unchecked_mut(j) = *row_freq.get_unchecked(i * n_freq + j);
            }

            // hermitian reconstruction
            for j in 1..(n_freq - 1) {
                *row_buf.get_unchecked_mut(n - j) = row_buf.get_unchecked(j).conj();
            }

            // Nyquist term real-only
            if n % 2 == 0 && n_freq > 1 {
                let ny = *row_buf.get_unchecked(n / 2);
                *row_buf.get_unchecked_mut(n / 2) = Complex::new(ny.re, T::zero());
            }

            ifft_row.process_with_scratch(row_buf, scratch);

            // store real output
            let out_row: &mut [T] = &mut *output_mutex.data_ptr();
            for j in 0..n {
                *out_row.get_unchecked_mut(i * n + j) = row_buf.get_unchecked(j).re / norm_n;
            }
        },
    );
}
