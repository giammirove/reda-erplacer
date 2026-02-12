use crate::placement::Bins;
use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::{Numeric, VecCoords, VecSizes};

#[inline(always)]
fn triangle_density_function<T>(x: T, node_size: T, xl: T, k: usize, bin_size: T) -> T
where
    T: Numeric,
{
    let bin_xl = xl + unsafe { T::from(k).unwrap_unchecked() } * bin_size;
    ((x + node_size).min(bin_xl + bin_size) - x.max(bin_xl)).max(T::zero())
}

#[inline(always)]
fn compute_density_overflow<T>(x: T, node_size: T, bin_center: T, bin_size: T) -> T
where
    T: Numeric,
{
    let two = T::one() + T::one();
    ((x + node_size).min(bin_center + bin_size / two) - (x.max(bin_center - bin_size / two)))
        .max(T::zero())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_exact_density_overflow_map_parallel<T>(
    x: &[T],
    y: &[T],
    size_x: &[T],
    size_y: &[T],
    bin_center_x: &[T],
    bin_center_y: &[T],
    xl: T,
    yl: T,
    bin_size_x: T,
    bin_size_y: T,
    buf_map: &mut [T],
    thread_maps: &mut [Vec<T>],
) where
    T: Numeric + std::fmt::Debug,
{
    let one = T::one();
    let inv_bin_size_x = one / bin_size_x;
    let inv_bin_size_y = one / bin_size_y;
    let num_bins_x = bin_center_x.len();
    let num_bins_y = bin_center_y.len();

    let num_threads = thread_maps.len();
    let chunk_size = x.len().div_ceil(num_threads);

    thread_maps
        .par_iter_mut()
        .enumerate()
        .for_each(|(thread_index, thread_buf)| {
            let start = thread_index * chunk_size;
            let end = ((thread_index + 1) * chunk_size).min(x.len());

            for i in start..end {
                let xi = x[i];
                let yi = y[i];
                let sxi = size_x[i];
                let syi = size_y[i];

                let mut bx_l = ((xi - xl) * inv_bin_size_x).floor().to_usize().unwrap();
                let mut bx_h = ((xi - xl + sxi) * inv_bin_size_x)
                    .ceil()
                    .to_usize()
                    .unwrap()
                    + 1;
                bx_l = bx_l.max(0);
                bx_h = bx_h.min(num_bins_x - 1);

                let mut by_l = ((yi - yl) * inv_bin_size_y).floor().to_usize().unwrap();
                let mut by_h = ((yi - yl + syi) * inv_bin_size_y)
                    .ceil()
                    .to_usize()
                    .unwrap()
                    + 1;
                by_l = by_l.max(0);
                by_h = by_h.min(num_bins_y - 1);

                for (bx, _) in bin_center_x.iter().enumerate().take(bx_h).skip(bx_l) {
                    let px = compute_density_overflow(xi, sxi, bin_center_x[bx], bin_size_x);
                    let row = bx * num_bins_y;

                    for by in by_l..by_h {
                        let py = compute_density_overflow(yi, syi, bin_center_y[by], bin_size_y);
                        thread_buf[row + by] += px * py;
                    }
                }
            }
        });

    for thread_buf in thread_maps {
        buf_map.par_iter_mut().enumerate().for_each(|(i, v)| {
            *v += thread_buf[i];
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_density_overflow_map_parallel<T>(
    x: &[T],
    y: &[T],
    size_x: &[T],
    size_y: &[T],
    offset_x: &[T],
    offset_y: &[T],
    ratios: &[T],
    xl: T,
    yl: T,
    num_bins_x: usize,
    num_bins_y: usize,
    bin_size_x: T,
    bin_size_y: T,
    initial_density_map: &[T],
    density_map: &mut [T],
    density_map_mutex: &[Mutex<T>],
) where
    T: Numeric,
{
    let one = T::one();
    let inv_bin_size_x = one / bin_size_x;
    let inv_bin_size_y = one / bin_size_y;

    (0..x.len()).into_par_iter().for_each(|i| {
        let xi = unsafe { *x.get_unchecked(i) + *offset_x.get_unchecked(i) };
        let yi = unsafe { *y.get_unchecked(i) + *offset_y.get_unchecked(i) };
        let sxi = unsafe { *size_x.get_unchecked(i) };
        let syi = unsafe { *size_y.get_unchecked(i) };
        let ratio = unsafe { *ratios.get_unchecked(i) };

        let mut bx_l = unsafe { ((xi - xl) * inv_bin_size_x).to_isize().unwrap_unchecked() };
        let mut bx_h = unsafe {
            ((xi - xl + sxi) * inv_bin_size_x)
                .to_isize()
                .unwrap_unchecked()
        } + 1;
        bx_l = bx_l.max(0);
        bx_h = bx_h.min(num_bins_x as isize - 1);

        let mut by_l = unsafe { ((yi - yl) * inv_bin_size_y).to_isize().unwrap_unchecked() };
        let mut by_h = unsafe {
            ((yi - yl + syi) * inv_bin_size_y)
                .to_isize()
                .unwrap_unchecked()
        } + 1;
        by_l = by_l.max(0);
        by_h = by_h.min(num_bins_y as isize - 1);

        for bx in bx_l as usize..bx_h as usize {
            let px = ratio * triangle_density_function(xi, sxi, xl, bx, bin_size_x);
            let row = bx * num_bins_y;
            for by in by_l as usize..by_h as usize {
                let py = triangle_density_function(yi, syi, yl, by, bin_size_y);
                let idx = row + by;
                unsafe {
                    *density_map_mutex.get_unchecked(idx).lock() += px * py;
                }
            }
        }
    });

    density_map
        .par_iter_mut()
        .zip(density_map_mutex.par_iter())
        .zip(initial_density_map.par_iter())
        .for_each(|((dest, mutex), &initial)| {
            *dest = unsafe { *mutex.data_ptr() } + initial;
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_density_overflow_map_sequential<T: Numeric>(
    x: &[T],
    y: &[T],
    size_x: &[T],
    size_y: &[T],
    offset_x: &[T],
    offset_y: &[T],
    ratios: &[T],
    xl: T,
    yl: T,
    num_bins_x: usize,
    num_bins_y: usize,
    bin_size_x: T,
    bin_size_y: T,
    initial_density_map: &[T],
    density_map: &mut [T],
) {
    let one = T::one();
    let inv_bin_size_x = one / bin_size_x;
    let inv_bin_size_y = one / bin_size_y;

    for i in 0..x.len() {
        let xi = x[i] + offset_x[i];
        let yi = y[i] + offset_y[i];
        let sxi = size_x[i];
        let syi = size_y[i];
        let ratio = ratios[i];

        let mut bx_l = ((xi - xl) * inv_bin_size_x).floor().to_isize().unwrap();
        let mut bx_h = ((xi - xl + sxi) * inv_bin_size_x)
            .ceil()
            .to_isize()
            .unwrap()
            + 1;
        bx_l = bx_l.max(0);
        bx_h = bx_h.min(num_bins_x as isize - 1);

        let mut by_l = ((yi - yl) * inv_bin_size_y).floor().to_isize().unwrap();
        let mut by_h = ((yi - yl + syi) * inv_bin_size_y)
            .ceil()
            .to_isize()
            .unwrap()
            + 1;
        by_l = by_l.max(0);
        by_h = by_h.min(num_bins_y as isize - 1);

        for bx in bx_l as usize..bx_h as usize {
            let px = ratio * triangle_density_function(xi, sxi, xl, bx, bin_size_x);
            let row = bx * num_bins_y;

            for by in by_l as usize..by_h as usize {
                let py = triangle_density_function(yi, syi, yl, by, bin_size_y);
                density_map[row + by] += px * py;
            }
        }
    }

    density_map
        .par_iter_mut()
        .zip(initial_density_map.par_iter())
        .for_each(|(dest, &initial)| {
            *dest += initial;
        });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_density_map<T: Numeric>(
    num_movable: usize,
    bins: &Bins<T>,
    instances_coords: &VecCoords<T>,
    instances_clamped_sizes: &VecSizes<T>,
    instances_clamped_offsets: &VecCoords<T>,
    ratios: &[T],
    initial_density_map: &[T],
    density_map: &mut [T],
    density_map_mutex: &[Mutex<T>],
) {
    compute_density_overflow_map_parallel(
        &instances_coords.x[..num_movable],
        &instances_coords.y[..num_movable],
        &instances_clamped_sizes.w[..num_movable],
        &instances_clamped_sizes.h[..num_movable],
        &instances_clamped_offsets.x[..num_movable],
        &instances_clamped_offsets.y[..num_movable],
        &ratios[..num_movable],
        T::zero(),
        T::zero(),
        bins.num_bins.x,
        bins.num_bins.y,
        bins.bin_size.width,
        bins.bin_size.height,
        initial_density_map,
        density_map,
        density_map_mutex,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_density_map_fallback<T: Numeric>(
    num_movable: usize,
    bins: &Bins<T>,
    instances_coords: &VecCoords<T>,
    instances_clamped_sizes: &VecSizes<T>,
    instances_clamped_offsets: &VecCoords<T>,
    ratios: &[T],
    initial_density_map: &[T],
    density_map: &mut [T],
) {
    compute_density_overflow_map_sequential(
        &instances_coords.x[..num_movable],
        &instances_coords.y[..num_movable],
        &instances_clamped_sizes.w[..num_movable],
        &instances_clamped_sizes.h[..num_movable],
        &instances_clamped_offsets.x[..num_movable],
        &instances_clamped_offsets.y[..num_movable],
        &ratios[..num_movable],
        T::zero(),
        T::zero(),
        bins.num_bins.x,
        bins.num_bins.y,
        bins.bin_size.width,
        bins.bin_size.height,
        initial_density_map,
        density_map,
    );
}

pub(crate) fn compute_initial_density_map<T>(
    num_movable: usize,
    bins: &Bins<T>,
    instances_coords: &VecCoords<T>,
    instances_sizes: &VecSizes<T>,
    buf_map: &mut [T],
    thread_maps: &mut [Vec<T>],
) where
    T: Numeric + std::fmt::Debug,
{
    compute_exact_density_overflow_map_parallel(
        &instances_coords.x[num_movable..],
        &instances_coords.y[num_movable..],
        &instances_sizes.w[num_movable..],
        &instances_sizes.h[num_movable..],
        &bins.bin_centers.x,
        &bins.bin_centers.y,
        T::zero(),
        T::zero(),
        bins.bin_size.width,
        bins.bin_size.height,
        buf_map,
        thread_maps,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_vec_approx_eq;
    use crate::utils::{read_matrix_from_file, time_it};

    #[test]
    fn test_initial_density_map() {
        let num_nodes = 4;

        let (x, nx, _) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_x.txt").unwrap();
        let (y, ny, _) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_y.txt").unwrap();

        assert_eq!(nx, num_nodes);
        assert_eq!(ny, num_nodes);

        let (size_x, _, _) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_node_size_x.txt")
                .unwrap();
        let (size_y, _, _) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_node_size_y.txt")
                .unwrap();

        let (bin_center_x, bm, _) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_bin_center_x.txt")
                .unwrap();
        let (bin_center_y, bn, _) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_bin_center_y.txt")
                .unwrap();

        assert_eq!(bm, 512);
        assert_eq!(bn, 512);

        let (density_out, m, n) =
            read_matrix_from_file::<f32>("./tests/density/initial_density_out.txt").unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);

        let num_threads = 1;
        let bin_w = (1.7396e+06 - 6000.) / 200. / 512.;
        let bin_h = (1.1724e+06 - 6000.) / 200. / 512.;
        let mut thread_maps: Vec<Vec<f32>> = (0..num_threads).map(|_| vec![0.; m * n]).collect();
        let mut density_map = vec![0.; m * n];

        compute_exact_density_overflow_map_parallel(
            &x,
            &y,
            &size_x,
            &size_y,
            &bin_center_x,
            &bin_center_y,
            0.,
            0.,
            bin_w,
            bin_h,
            &mut density_map,
            &mut thread_maps,
        );

        assert_vec_approx_eq!(density_map, density_out, 1e-2);
    }

    #[test]
    fn test_density_map() {
        let num_nodes = 72090;

        let (x, nx, _) = read_matrix_from_file::<f64>("./tests/density/density_x.txt").unwrap();
        let (y, ny, _) = read_matrix_from_file::<f64>("./tests/density/density_y.txt").unwrap();

        assert_eq!(nx, num_nodes);
        assert_eq!(ny, num_nodes);

        let (size_x, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_node_size_x.txt").unwrap();
        let (size_y, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_node_size_y.txt").unwrap();

        let (offset_x, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_offset_x.txt").unwrap();
        let (offset_y, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_offset_y.txt").unwrap();

        let (ratios, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_ratios.txt").unwrap();

        let (density_out, m, n) =
            read_matrix_from_file::<f64>("./tests/density/density_out.txt").unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);

        let (initial_density_map, _, _) =
            read_matrix_from_file::<f64>("./tests/density/initial_density_map.txt").unwrap();

        let bin_w = (1.7396e+06 - 6000.) / 200. / 512.;
        let bin_h = (1.1724e+06 - 6000.) / 200. / 512.;
        let mut density_map = vec![0.; m * n];

        time_it("density map sequential", || {
            compute_density_overflow_map_sequential(
                &x,
                &y,
                &size_x,
                &size_y,
                &offset_x,
                &offset_y,
                &ratios,
                0.,
                0.,
                m,
                n,
                bin_w,
                bin_h,
                &initial_density_map,
                &mut density_map,
            );
        });

        assert_vec_approx_eq!(density_map, density_out, 1e-1);
    }

    #[test]
    fn test_density_map_parallel() {
        let num_nodes = 72090;

        let (x, nx, _) = read_matrix_from_file::<f64>("./tests/density/density_x.txt").unwrap();
        let (y, ny, _) = read_matrix_from_file::<f64>("./tests/density/density_y.txt").unwrap();

        assert_eq!(nx, num_nodes);
        assert_eq!(ny, num_nodes);

        let (size_x, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_node_size_x.txt").unwrap();
        let (size_y, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_node_size_y.txt").unwrap();

        let (offset_x, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_offset_x.txt").unwrap();
        let (offset_y, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_offset_y.txt").unwrap();

        let (ratios, _, _) =
            read_matrix_from_file::<f64>("./tests/density/density_ratios.txt").unwrap();

        let (density_out, m, n) =
            read_matrix_from_file::<f64>("./tests/density/density_out.txt").unwrap();

        assert_eq!(m, 512);
        assert_eq!(n, 512);

        let (initial_density_map, _, _) =
            read_matrix_from_file::<f64>("./tests/density/initial_density_map.txt").unwrap();

        let bin_w = (1.7396e+06 - 6000.) / 200. / 512.;
        let bin_h = (1.1724e+06 - 6000.) / 200. / 512.;
        let mut density_map: Vec<f64> = vec![0.; m * n];

        let mut density_map_mutex: Vec<Mutex<f64>> = (0..m * n).map(|_| Mutex::new(0.0)).collect();
        for _ in 0..2 {
            density_map.par_iter_mut().for_each(|m| *m = 0.0);
            density_map_mutex.par_iter().for_each(|m| *m.lock() = 0.0);
            time_it("density map parallel", || {
                compute_density_overflow_map_parallel(
                    &x,
                    &y,
                    &size_x,
                    &size_y,
                    &offset_x,
                    &offset_y,
                    &ratios,
                    0.,
                    0.,
                    m,
                    n,
                    bin_w,
                    bin_h,
                    &initial_density_map,
                    &mut density_map,
                    &mut density_map_mutex,
                );
            });
        }

        assert_vec_approx_eq!(density_map, density_out, 1e-2);
    }
}
