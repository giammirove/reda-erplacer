use reda_db::Numeric;
use std::f64::consts::PI;

pub(crate) fn compute_frequency_matrices<T>(
    m: usize,
    n: usize,
    bin_size_x: T,
    bin_size_y: T,
) -> (Vec<T>, Vec<T>, Vec<T>)
where
    T: Numeric + std::fmt::Debug,
{
    // wu = torch.arange(M).mul(2 * pi / M).view([M, 1])
    let mut wu = vec![T::zero(); m];
    let two_pi_over_m = T::from(2.0 * PI).unwrap() / T::from(m).unwrap();
    for (i, ewu) in wu.iter_mut().enumerate().take(m) {
        *ewu = T::from(i).unwrap() * two_pi_over_m;
    }

    // wv = torch.arange(N).mul(2 * np.pi / N).view( [1, N]).mul_(self.bin_size_x / self.bin_size_y)
    let mut wv = vec![T::zero(); n];
    let wv_div = bin_size_x / bin_size_y;
    let two_pi_over_n = T::from(2.0 * PI).unwrap() / T::from(n).unwrap();
    for (j, ewv) in wv.iter_mut().enumerate().take(n) {
        *ewv = T::from(j).unwrap() * two_pi_over_n * wv_div;
    }

    // wu2_plus_wv2 = wu.pow(2) + wv.pow(2)
    let mut wu2_plus_wv2 = vec![T::zero(); m * n];
    for i in 0..m {
        let wu_val = wu[i];
        for j in 0..n {
            let wv_val = wv[j];
            wu2_plus_wv2[i * n + j] = wu_val * wu_val + wv_val * wv_val;
        }
    }

    // wu2_plus_wv2[0, 0] = 1.0 (avoid zero-division)
    wu2_plus_wv2[0] = T::one();

    // inv_wu2_plus_wv2 = 1.0 / wu2_plus_wv2
    let mut inv_wu2_plus_wv2 = vec![T::zero(); m * n];
    for i in 0..(m * n) {
        inv_wu2_plus_wv2[i] = T::one() / wu2_plus_wv2[i];
    }

    // inv_wu2_plus_wv2[0, 0] = 0.0
    inv_wu2_plus_wv2[0] = T::zero();

    // wu_by_wu2_plus_wv2_half = wu.mul(inv_wu2_plus_wv2).mul(1/2)
    let mut wu_by_wu2_plus_wv2_half = vec![T::zero(); m * n];
    let half = T::from(0.5).unwrap();
    for i in 0..m {
        for j in 0..n {
            wu_by_wu2_plus_wv2_half[i * n + j] = wu[i] * inv_wu2_plus_wv2[i * n + j] * half;
        }
    }

    // wv_by_wu2_plus_wv2_half = wv.mul(inv_wu2_plus_wv2).mul(1/2)
    let mut wv_by_wu2_plus_wv2_half = vec![T::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            wv_by_wu2_plus_wv2_half[i * n + j] = wv[j] * inv_wu2_plus_wv2[i * n + j] * half;
        }
    }

    (
        wu_by_wu2_plus_wv2_half,
        wv_by_wu2_plus_wv2_half,
        inv_wu2_plus_wv2,
    )
}
