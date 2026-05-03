use crate::{
    dct2d::{dct2_fft2_forward, DCT2D},
    density::{compute_density_map, compute_initial_density_map},
    eforce::compute_electric_force,
    frequencies::compute_frequency_matrices,
    idct2::idct2_fft2_forward,
    idct_idxst::idct_idxst_forward,
    idxst_idct::idxst_idct_forward,
    placement::{Bins, PlacementSolution},
    utils::{make_expk, time_it},
};
use image::{ImageBuffer, Rgb};
use num_traits::Float;
use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::{Numeric, VecCoords, VecSizes, DB};
use rustfft::num_complex::Complex;
use std::f32::consts::SQRT_2;

#[derive(Debug)]
pub(crate) struct PotentialComputation<T>
where
    T: Numeric,
{
    initial_density_map: Vec<T>,
    target_density_map: Vec<T>,
    density_map: Vec<T>,
    density_map_mutex: Vec<Mutex<T>>,
    expk_m: Vec<Complex<T>>,
    expk_n: Vec<Complex<T>>,

    instance_clamped_sizes: VecSizes<T>,
    instance_clamped_offsets: VecCoords<T>,
    instance_ratios: Vec<T>,

    wu_by_wu2_plus_wv2_half: Vec<T>,
    wv_by_wu2_plus_wv2_half: Vec<T>,
    inv_wu2_plus_wv2: Vec<T>,

    auv: Vec<T>,
    auv_by_wu2_plus_wv2_wu: Vec<T>,
    auv_by_wu2_plus_wv2_wv: Vec<T>,
    field_map_x: Vec<T>,
    field_map_y: Vec<T>,
    auv_by_wu2_plus_wv2: Vec<T>,
    buf: Vec<Complex<T>>,
    potential_map: Vec<T>,
    energy_map: Vec<T>,
    dct2d: DCT2D<T>,

    pub(crate) grad_x: Vec<T>,
    pub(crate) grad_y: Vec<T>,

    pub(crate) movable_macro_mask: Vec<bool>,
    dreamplace: bool,
}

impl<T> PotentialComputation<T>
where
    T: Numeric + std::fmt::Debug,
{
    pub(crate) fn new(
        db: &DB<T>,
        bins: &Bins<T>,
        target_density: T,
        m: usize,
        n: usize,
        num_threads: usize,
        macro_density_scale: f32,
        gap_boost_radius: i32,
        gap_boost_max: f32,
        dreamplace: bool,
    ) -> Self {
        let zero = T::zero();
        let one = T::one();

        let num_nodes = db.instances.len();

        let density_map = vec![zero; m * n];
        let density_map_mutex: Vec<Mutex<T>> = (0..m * n).map(|_| Mutex::new(T::zero())).collect();
        let expk_m = make_expk(m);
        let expk_n = make_expk(n);

        let mut instance_sizes_clamped_w = vec![zero; num_nodes];
        let mut instance_sizes_clamped_h = vec![zero; num_nodes];
        let mut instance_offsets_x = vec![zero; num_nodes];
        let mut instance_offsets_y = vec![zero; num_nodes];
        let mut instance_ratios = vec![one; num_nodes];
        let sqrt2 = T::from(SQRT_2).unwrap();
        let half = T::from(0.5).unwrap();
        for i in 0..num_nodes {
            let sizes = &db.instances.sizes;
            let w = sizes.w[i];
            let h = sizes.h[i];

            let clamped_w = w.max(bins.bin_size.width * sqrt2);
            let offset_x = (w - clamped_w).mul(half);

            let clamped_h = h.max(bins.bin_size.height * sqrt2);
            let offset_y = (h - clamped_h).mul(half);

            let mut ratio = (w * h) / (clamped_w * clamped_h);
            if i >= db.num_movable {
                ratio = target_density;
            }
            instance_ratios[i] = ratio;
            instance_sizes_clamped_w[i] = clamped_w;
            instance_sizes_clamped_h[i] = clamped_h;
            instance_offsets_x[i] = offset_x;
            instance_offsets_y[i] = offset_y;
        }
        // Identify movable macros by area threshold (10x median movable area)
        // Only apply when target_density < 1 (DREAMPlace guard) and not all instances are movable
        let all_movable = db.num_movable == db.instances.len();
        let movable_macro_mask = if db.num_movable > 0 && target_density < T::one() && !all_movable
        {
            let mut sorted_areas: Vec<T> = db.instances.areas[..db.num_movable].to_vec();
            sorted_areas.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted_areas[sorted_areas.len() / 2];
            let threshold = median * T::from(10.0).unwrap();
            let mask: Vec<bool> = (0..db.num_movable)
                .map(|i| db.instances.areas[i] > threshold)
                .collect();
            // Set ratio = target_density for movable macros (like fixed cells)
            for i in 0..db.num_movable {
                if mask[i] {
                    instance_ratios[i] = target_density;
                }
            }
            mask
        } else {
            vec![false; db.num_movable]
        };
        let num_movable_macros = movable_macro_mask.iter().filter(|&&m| m).count();
        if num_movable_macros > 0 {
            log::info!(
                "Movable macro mask: {} macros detected (density grad zeroed)",
                num_movable_macros
            );
        }

        let instance_clamped_sizes =
            VecSizes::new(instance_sizes_clamped_w, instance_sizes_clamped_h);
        let instance_clamped_offsets = VecCoords::new(instance_offsets_x, instance_offsets_y);

        let auv = vec![zero; m * n];
        let auv_by_wu2_plus_wv2_wu = vec![zero; m * n];
        let auv_by_wu2_plus_wv2_wv = vec![zero; m * n];
        let field_map_x = vec![zero; m * n];
        let field_map_y = vec![zero; m * n];
        let auv_by_wu2_plus_wv2 = vec![zero; m * n];

        let (wu_by_wu2_plus_wv2_half, wv_by_wu2_plus_wv2_half, inv_wu2_plus_wv2) =
            compute_frequency_matrices::<T>(m, n, bins.bin_size.width, bins.bin_size.height);

        let buf = make_expk(m * n);
        let potential_map = vec![zero; m * n];
        let energy_map = vec![zero; m * n];

        let dct2d = DCT2D::new(m, n);

        let grad_x = vec![zero; num_nodes];
        let grad_y = vec![zero; num_nodes];

        let mut initial_density_map = vec![zero; m * n];
        let mut initial_thread_maps: Vec<Vec<T>> =
            (0..num_threads).map(|_| vec![zero; m * n]).collect();
        compute_initial_density_map(
            db.num_movable,
            bins,
            &db.instances.coords,
            &instance_clamped_sizes,
            &mut initial_density_map,
            &mut initial_thread_maps,
        );
        for item in &mut initial_density_map {
            *item = *item * target_density * T::from(macro_density_scale).unwrap();
        }

        // Per-bin target density.
        // dreamplace mode: uniform target everywhere (DREAMPlace feeds raw density to Poisson).
        // ERPlacer mode: macro bins get target=0, optional BFS gap boost.
        let bin_area = bins.bin_size.width * bins.bin_size.height;
        let full_target = target_density * bin_area;
        let target_density_map: Vec<T> = if dreamplace {
            vec![full_target; m * n]
        } else {
            let occupancy_threshold = bin_area * T::from(0.5).unwrap();
            let is_macro: Vec<bool> = initial_density_map
                .iter()
                .map(|&d| d > occupancy_threshold)
                .collect();

            if gap_boost_radius > 0 {
                // BFS distance from each free bin to nearest macro bin
                let mut dist: Vec<i32> = vec![i32::MAX; m * n];
                let mut queue = std::collections::VecDeque::new();
                for idx in 0..m * n {
                    if is_macro[idx] {
                        dist[idx] = 0;
                        queue.push_back(idx);
                    }
                }
                while let Some(idx) = queue.pop_front() {
                    let bx = idx / n;
                    let by = idx % n;
                    let d = dist[idx] + 1;
                    for (nx, ny) in [
                        (bx.wrapping_sub(1), by),
                        (bx + 1, by),
                        (bx, by.wrapping_sub(1)),
                        (bx, by + 1),
                    ] {
                        if nx < m && ny < n {
                            let nidx = nx * n + ny;
                            if d < dist[nidx] {
                                dist[nidx] = d;
                                queue.push_back(nidx);
                            }
                        }
                    }
                }

                let max_boost = T::from(gap_boost_max).unwrap();
                (0..m * n)
                    .map(|idx| {
                        if is_macro[idx] {
                            return T::zero();
                        }
                        let d = dist[idx];
                        if d <= gap_boost_radius {
                            let t = T::from((gap_boost_radius - d) as f32).unwrap()
                                / T::from((gap_boost_radius - 1) as f32).unwrap();
                            full_target * (T::one() + t * (max_boost - T::one()))
                        } else {
                            full_target
                        }
                    })
                    .collect()
            } else {
                (0..m * n)
                    .map(|idx| {
                        if is_macro[idx] {
                            T::zero()
                        } else {
                            full_target
                        }
                    })
                    .collect()
            }
        };

        Self {
            initial_density_map,
            target_density_map,
            density_map,
            density_map_mutex,
            expk_m,
            expk_n,

            instance_clamped_sizes,
            instance_clamped_offsets,
            instance_ratios,

            wu_by_wu2_plus_wv2_half,
            wv_by_wu2_plus_wv2_half,
            inv_wu2_plus_wv2,

            auv,
            auv_by_wu2_plus_wv2_wu,
            auv_by_wu2_plus_wv2_wv,
            field_map_x,
            field_map_y,
            auv_by_wu2_plus_wv2,
            buf,
            potential_map,
            energy_map,
            dct2d,

            grad_x,
            grad_y,

            movable_macro_mask,
            dreamplace,
        }
    }

    pub(crate) fn reset(&mut self) {
        let zero = T::zero();

        self.density_map.par_iter_mut().for_each(|m| *m = zero);
        self.density_map_mutex
            .par_iter()
            .for_each(|m| unsafe { *m.data_ptr() = zero });
    }
}

pub(crate) fn mat_mul_scalar_par<T>(a: &mut [T], k: T)
where
    T: Float + Send + Sync,
{
    a.par_iter_mut().for_each(|x| *x = *x * k);
}

pub(crate) fn mat_mul_par<T>(a: &[T], b: &[T], out: &mut [T])
where
    T: Float + Send + Sync,
{
    out.par_iter_mut().zip(a).zip(b).for_each(|((o, &x), &y)| {
        *o = x * y;
    });
}

#[allow(dead_code)]
pub(crate) fn draw_density_map(map: &[f32], m: usize, n: usize, index: usize) {
    let filename = format!("density_map/density_map_{:05}.png", index);

    let min_val = map.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = map.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut imgbuf = ImageBuffer::new(m as u32, n as u32);

    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let normalized = ((map[idx] - min_val) / (max_val - min_val + 1e-12)).clamp(0.0, 1.0);

            // Simple heatmap: blue -> red
            let r = (normalized * 255.0) as u8;
            let g = 0u8;
            let b = ((1.0 - normalized) * 255.0) as u8;

            imgbuf.put_pixel(i as u32, (n - j - 1) as u32, Rgb([r, g, b]));
        }
    }

    imgbuf.save(&filename).expect("Failed to save image");
}

pub(crate) fn compute_potential<T>(
    _iteration: usize,
    num_movable: usize,
    _target_density: T,
    movable_area: T,
    bins: &Bins<T>,
    ps: &PlacementSolution<T>,
    pc: &mut PotentialComputation<T>,
) -> (T, T)
where
    T: Numeric,
{
    let m = bins.num_bins.x;
    let n = bins.num_bins.y;

    time_it("potential reset", || {
        pc.reset();
    });

    time_it("compute_density_map", || {
        compute_density_map(
            num_movable,
            bins,
            &ps.instances,
            &pc.instance_clamped_sizes,
            &pc.instance_clamped_offsets,
            &pc.instance_ratios,
            &pc.initial_density_map,
            &mut pc.density_map,
            &pc.density_map_mutex,
        );
    });

    let mut overflow = T::zero();
    let bin_area = bins.bin_size.width * bins.bin_size.height;
    if pc.dreamplace {
        // DREAMPlace: overflow from full density (movable + fixed)
        let target_bin = ps.target_density * bin_area;
        for &d in pc.density_map.iter() {
            overflow += (d - target_bin).max(T::zero());
        }
    } else {
        for (d, &t) in pc.density_map.iter().zip(pc.target_density_map.iter()) {
            // Only count overflow in free-space bins (target > 0)
            if t > T::zero() {
                overflow += (*d - t).max(T::zero());
            }
        }
    }
    overflow = overflow / movable_area;

    // draw_density_map_f32_with_index(&pc.initial_density_map, m, n, iteration);

    let expk_m = &pc.expk_m;
    let expk_n = &pc.expk_n;

    let bin_area = bins.bin_size.width * bins.bin_size.height;
    let k = T::one() / bin_area;
    if pc.dreamplace {
        // DREAMPlace: feed density/bin_area directly to Poisson (no movable-target subtraction)
        mat_mul_scalar_par(&mut pc.density_map, k)
    } else {
        // ERPlacer: feed (movable_density - target) / bin_area
        pc.density_map
            .par_iter_mut()
            .zip(pc.initial_density_map.par_iter())
            .zip(pc.target_density_map.par_iter())
            .for_each(|((d, &initial), &target)| {
                let movable = *d - initial; // strip out fixed cell density
                *d = (movable - target) * k;
            });
    }

    time_it("dct2_fft2_forward", || {
        dct2_fft2_forward(
            &pc.density_map,
            expk_m,
            expk_n,
            &mut pc.dct2d,
            &mut pc.auv,
            m,
            n,
        );
    });

    mat_mul_par(
        &pc.auv,
        &pc.wu_by_wu2_plus_wv2_half,
        &mut pc.auv_by_wu2_plus_wv2_wu,
    );
    mat_mul_par(
        &pc.auv,
        &pc.wv_by_wu2_plus_wv2_half,
        &mut pc.auv_by_wu2_plus_wv2_wv,
    );

    time_it("idxst_idct_forward", || {
        idxst_idct_forward(
            &pc.auv_by_wu2_plus_wv2_wu,
            &pc.expk_m,
            &pc.expk_n,
            &mut pc.field_map_x,
            m,
            n,
        );
    });
    time_it("idct_idxst_forward", || {
        idct_idxst_forward(
            &pc.auv_by_wu2_plus_wv2_wv,
            &pc.expk_m,
            &pc.expk_n,
            &mut pc.field_map_y,
            m,
            n,
        );
    });

    mat_mul_par(&pc.auv, &pc.inv_wu2_plus_wv2, &mut pc.auv_by_wu2_plus_wv2);

    time_it("idct2_fft2_forward", || {
        idct2_fft2_forward(
            &pc.auv_by_wu2_plus_wv2,
            expk_m,
            expk_n,
            &mut pc.potential_map,
            &mut pc.buf,
            m,
            n,
        );
    });
    mat_mul_par(&pc.potential_map, &pc.density_map, &mut pc.energy_map);

    let energy: T = pc.energy_map.par_iter().copied().sum();

    time_it("compute_electric_force", || {
        let mutex_grad_x: Mutex<&mut [T]> = Mutex::new(&mut pc.grad_x);
        let mutex_grad_y: Mutex<&mut [T]> = Mutex::new(&mut pc.grad_y);
        compute_electric_force(
            num_movable,
            m,
            n,
            &pc.field_map_x,
            &pc.field_map_y,
            &ps.instances.x,
            &ps.instances.y,
            &pc.instance_clamped_sizes.w,
            &pc.instance_clamped_sizes.h,
            &pc.instance_clamped_offsets.x,
            &pc.instance_clamped_offsets.y,
            &pc.instance_ratios,
            T::zero(),
            T::zero(),
            bins.bin_size.width,
            bins.bin_size.height,
            &mutex_grad_x,
            &mutex_grad_y,
        );
    });

    // Zero density gradients for movable macros — only WL force moves them
    for (i, &is_macro) in pc.movable_macro_mask.iter().enumerate() {
        if is_macro {
            pc.grad_x[i] = T::zero();
            pc.grad_y[i] = T::zero();
        }
    }

    (energy, overflow)
}
