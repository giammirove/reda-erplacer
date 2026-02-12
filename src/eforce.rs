use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::Numeric;

#[inline(always)]
fn triangle_density_function<T>(x: T, node_size: T, xl: T, k: usize, bin_size: T) -> T
where
    T: Numeric,
{
    let bin_xl = xl + unsafe { T::from(k).unwrap_unchecked() } * bin_size;
    (x + node_size).min(bin_xl + bin_size) - x.max(bin_xl)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_electric_force<T>(
    num_movable: usize,
    num_bins_x: usize,
    num_bins_y: usize,
    field_map_x: &[T],
    field_map_y: &[T],
    x: &[T],
    y: &[T],
    node_size_x_clamped: &[T],
    node_size_y_clamped: &[T],
    offset_x: &[T],
    offset_y: &[T],
    ratio: &[T],
    xl: T,
    yl: T,
    bin_size_x: T,
    bin_size_y: T,
    mutex_grad_x: &Mutex<&mut [T]>,
    mutex_grad_y: &Mutex<&mut [T]>,
) where
    T: Numeric,
{
    let inv_bin_size_x = T::one() / bin_size_x;
    let inv_bin_size_y = T::one() / bin_size_y;

    (0..num_movable).into_par_iter().for_each(|i| {
        let grad_x: &mut [T] = unsafe { &mut *mutex_grad_x.data_ptr() };
        let grad_y: &mut [T] = unsafe { &mut *mutex_grad_y.data_ptr() };
        let node_size_x = node_size_x_clamped[i];
        let node_size_y = node_size_y_clamped[i];
        let node_x = x[i] + offset_x[i];
        let node_y = y[i] + offset_y[i];
        let node_ratio = ratio[i];

        let bin_index_xl = ((node_x - xl) * inv_bin_size_x)
            .to_isize()
            .unwrap_or(0)
            .max(0) as usize;

        let bin_index_xh = ((((node_x + node_size_x - xl) * inv_bin_size_x)
            .to_isize()
            .unwrap_or(0)
            + 1) as usize)
            .min(num_bins_x);

        let bin_index_yl = ((node_y - yl) * inv_bin_size_y)
            .to_isize()
            .unwrap_or(0)
            .max(0) as usize;

        let bin_index_yh = ((((node_y + node_size_y - yl) * inv_bin_size_y)
            .to_isize()
            .unwrap_or(0)
            + 1) as usize)
            .min(num_bins_y);

        let mut gx = T::zero();
        let mut gy = T::zero();

        for k in bin_index_xl..bin_index_xh {
            let px = triangle_density_function(node_x, node_size_x, xl, k, bin_size_x);

            for h in bin_index_yl..bin_index_yh {
                let py = triangle_density_function(node_y, node_size_y, yl, h, bin_size_y);

                let area = px * py;
                let idx = k * num_bins_y + h;

                gx += area * field_map_x[idx];
                gy += area * field_map_y[idx];
            }
        }

        grad_x[i] = -gx * node_ratio;
        grad_y[i] = -gy * node_ratio;
    });
}
