use std::fmt;

use crate::{
    epotential::{compute_potential, PotentialComputation},
    hpwl::HpwlComputation,
    nesterov::Nesterov,
    scheduler::Scheduler,
    utils::time_it,
};
use reda_db::{Numeric, VecCoords, DB};

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Coord<T> {
    pub(crate) x: T,
    pub(crate) y: T,
}
impl<T> Coord<T> {
    fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct Size<T> {
    pub(crate) width: T,
    pub(crate) height: T,
}
impl<T> Size<T> {
    fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

#[derive(Debug)]
pub(crate) struct Bins<T>
where
    T: Numeric + std::fmt::Debug,
{
    pub(crate) bin_size: Size<T>,
    pub(crate) num_bins: Coord<usize>,
    pub(crate) bin_centers: VecCoords<T>,
}
impl<T> Bins<T>
where
    T: Numeric,
{
    pub(crate) fn new_with_size(db: &DB<T>, bins: usize) -> Self {
        let num_bins: Coord<usize> = Coord::new(bins, bins);

        let bin_size_x = db.diearea.width() / T::from(num_bins.x).unwrap();
        let bin_size_y = db.diearea.height() / T::from(num_bins.y).unwrap();
        let zero = T::zero();
        let bin_centers_x = bin_centers(num_bins.x, zero, db.diearea.width(), bin_size_x);
        let bin_centers_y = bin_centers(num_bins.y, zero, db.diearea.height(), bin_size_y);

        let bin_size = Size::new(bin_size_x, bin_size_y);
        let bin_centers = VecCoords::new(bin_centers_x, bin_centers_y);
        Self {
            bin_size,
            num_bins,
            bin_centers,
        }
    }
}

impl<T> fmt::Display for Bins<T>
where
    T: Numeric + std::fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bins: {:?} x {:?} ({:?} x {:?})",
            self.bin_size.width, self.bin_size.height, self.num_bins.x, self.num_bins.y
        )
    }
}

fn bin_centers<T>(num_bins: usize, l: T, h: T, bin_size: T) -> Vec<T>
where
    T: Numeric + std::fmt::Debug,
{
    let mut centers = Vec::with_capacity(num_bins);

    for i in 0..num_bins {
        let bin_l = l + T::from(i).unwrap() * bin_size;
        let bin_h = (bin_l + bin_size).min(h);
        centers.push((bin_l + bin_h) / T::from(2.0).unwrap());
    }

    centers
}

#[derive(Debug)]
pub(crate) struct PlacementSolution<T>
where
    T: Numeric + std::fmt::Debug,
{
    // updated coordinates
    pub(crate) instances: VecCoords<T>,
    pub(crate) target_density: T,
}
impl<T> PlacementSolution<T>
where
    T: Numeric + std::fmt::Debug,
{
    pub(crate) fn new(db: &DB<T>, init_noise: f32) -> Self {
        let num_nodes = db.instances.len();
        let zero = T::zero();
        let two = T::from(2).unwrap();
        let center_x = db.diearea.width() / two;
        let center_y = db.diearea.height() / two;
        let noise_x = db.diearea.width() * T::from(init_noise).unwrap();
        let noise_y = db.diearea.height() * T::from(init_noise).unwrap();
        let mut new_instances_x = vec![zero; num_nodes];
        let mut new_instances_y = vec![zero; num_nodes];

        // Simple deterministic hash-based noise (no RNG dependency)
        let noise_val = |i: usize, seed: usize| -> T {
            let h = ((i.wrapping_mul(2654435761) ^ seed).wrapping_mul(2246822519)) as u32;
            T::from(h as f32 / u32::MAX as f32 * 2.0 - 1.0).unwrap() // -1.0 to 1.0
        };

        for i in 0..num_nodes {
            let new_x = &mut new_instances_x[i];
            let new_y = &mut new_instances_y[i];
            let old_x = db.instances.coords.x[i];
            let old_y = db.instances.coords.y[i];
            let w = db.instances.sizes.w[i];
            let h = db.instances.sizes.h[i];

            if i >= db.num_movable {
                *new_x = old_x;
                *new_y = old_y;
            } else {
                // TODO: using center_x - w / two => much worse results
                // *new_x = center_x - w / two;
                // *new_y = center_y - h / two;
                *new_x = center_x;
                *new_y = center_y;
                if init_noise > 0.0 {
                    *new_x =
                        (*new_x + noise_val(i, 0) * noise_x).clamp(zero, db.diearea.width() - w);
                    *new_y =
                        (*new_y + noise_val(i, 1) * noise_y).clamp(zero, db.diearea.height() - h);
                }
            }
        }
        let new_instances: VecCoords<T> = VecCoords::new(new_instances_x, new_instances_y);
        let target_density = db.cell_utilization;
        Self {
            instances: new_instances,
            target_density,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ComputationResult<T>
where
    T: Numeric,
{
    pub(crate) hpwl: T,
    pub(crate) overflow: T,
}

#[derive(Debug)]
pub(crate) struct Computation<T>
where
    T: Numeric,
{
    pub(crate) pc: PotentialComputation<T>,
    pub(crate) hc: HpwlComputation<T>,
}
impl<T> Computation<T>
where
    T: Numeric,
{
    pub(crate) fn new(pc: PotentialComputation<T>, hc: HpwlComputation<T>) -> Self {
        Computation { pc, hc }
    }

    pub(crate) fn compute(
        &mut self,
        db: &DB<T>,
        bins: &Bins<T>,
        sc: &Scheduler<T>,
        ps: &mut PlacementSolution<T>,
    ) -> ComputationResult<T> {
        let hpwl = time_it("hpwl", || self.hc.compute_hpwl(sc.gamma, db, ps));

        let (_, overflow) = time_it("compute potential", || {
            compute_potential(
                sc.iteration,
                db.num_movable,
                ps.target_density,
                db.movable_area,
                bins,
                ps,
                &mut self.pc,
            )
        });

        ComputationResult { hpwl, overflow }
    }
}

#[derive(Debug)]
pub(crate) struct Placement<'a, T>
where
    T: Numeric,
{
    pub(crate) db: &'a DB<T>,
    pub(crate) bins: Bins<T>,
    pub(crate) sc: Scheduler<T>,
    pub(crate) ne: Nesterov<T>,
    pub(crate) co: Computation<T>,
    pub(crate) ps: PlacementSolution<T>,
    best_hpwl: T,
    best_overflow: T,
    best_coords: Option<VecCoords<T>>,
    best_density_weight: T,
    best_gamma: T,
    target_overflow: T,
}
impl<'a, T> Placement<'a, T>
where
    T: Numeric,
{
    pub(crate) fn new(
        db: &'a DB<T>,
        macro_density_scale: f32,
        gap_boost_radius: i32,
        gap_boost_max: f32,
        num_bins: usize,
        target_overflow: f32,
        init_noise: f32,
        dreamplace: bool,
        initial_density_weight: f32,
        spread_macros: bool,
        macro_colors: &[[u8; 3]],
    ) -> Self {
        let num_threads = rayon::current_num_threads();
        let bins = Bins::new_with_size(db, num_bins);
        let m = bins.num_bins.x;
        let n = bins.num_bins.y;
        let mut ps = PlacementSolution::new(db, init_noise);
        let pc = PotentialComputation::new(
            db,
            &bins,
            ps.target_density,
            m,
            n,
            num_threads,
            macro_density_scale,
            gap_boost_radius,
            gap_boost_max,
            dreamplace,
        );
        let hc = HpwlComputation::new(db);
        let sc = Scheduler::new(&bins, target_overflow, initial_density_weight);
        let ne = Nesterov::new(&db.instances.coords);
        // Compute macro mask by area threshold (10x median), independent of density mask
        let macro_mask = if db.num_movable > 0 {
            let mut sorted: Vec<T> = db.instances.areas[..db.num_movable].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold = sorted[sorted.len() / 2] * T::from(10.0).unwrap();
            (0..db.num_movable)
                .map(|i| db.instances.areas[i] > threshold)
                .collect()
        } else {
            vec![]
        };

        // Spread macros uniformly across die so density/WL gradients are non-zero from start
        // When spread_macros is true but macros are fixed, include fixed instances by area threshold
        let macro_indices: Vec<usize> = if spread_macros && macro_mask.iter().all(|&m| !m) {
            // No movable macros — use fixed instances (index >= num_movable) by area threshold
            let all_areas = &db.instances.areas;
            let mut sorted: Vec<T> = all_areas.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let threshold = sorted[sorted.len() / 2] * T::from(10.0).unwrap();
            (0..db.instances.len())
                .filter(|&i| all_areas[i] > threshold)
                .collect()
        } else {
            (0..db.num_movable)
                .filter(|&i| macro_mask.get(i).copied().unwrap_or(false))
                .collect()
        };
        if spread_macros && !macro_indices.is_empty() {
            // Group macros by cluster (color). If no colors provided, each macro is its own cluster.
            let mut cluster_map: std::collections::HashMap<[u8; 3], Vec<usize>> =
                std::collections::HashMap::new();
            for (midx, &i) in macro_indices.iter().enumerate() {
                let color = macro_colors
                    .get(midx)
                    .copied()
                    .unwrap_or([midx as u8, 0, 0]);
                cluster_map.entry(color).or_default().push(i);
            }
            // Sort clusters deterministically
            let mut clusters: Vec<Vec<usize>> = cluster_map.into_values().collect();
            clusters.sort_by_key(|c| c[0]);

            let num_clusters = clusters.len();
            let grid_cols = (num_clusters as f64).sqrt().ceil() as usize;
            let grid_rows = (num_clusters + grid_cols - 1) / grid_cols;
            let region_w = db.diearea.width() / T::from(grid_cols).unwrap();
            let region_h = db.diearea.height() / T::from(grid_rows).unwrap();
            let offset_x = T::zero();
            let offset_y = T::zero();

            for (ci, cluster) in clusters.iter().enumerate() {
                // Cluster region in the die
                let col = ci % grid_cols;
                let row = ci / grid_cols;
                let rx =
                    offset_x + region_w * T::from(col).unwrap() + region_w / T::from(2).unwrap();
                let ry =
                    offset_y + region_h * T::from(row).unwrap() + region_h / T::from(2).unwrap();

                // Place macros tightly using actual macro size
                let n = cluster.len();
                let sub_cols = (n as f64).sqrt().ceil() as usize;
                let sub_rows = (n + sub_cols - 1) / sub_cols;
                let max_w = cluster
                    .iter()
                    .map(|&i| db.instances.sizes.w[i])
                    .fold(T::zero(), |a, b| if a > b { a } else { b });
                let max_h = cluster
                    .iter()
                    .map(|&i| db.instances.sizes.h[i])
                    .fold(T::zero(), |a, b| if a > b { a } else { b });
                let cell_w = max_w * T::from(1.1).unwrap();
                let cell_h = max_h * T::from(1.1).unwrap();
                let grid_w = cell_w * T::from(sub_cols).unwrap();
                let grid_h = cell_h * T::from(sub_rows).unwrap();
                let ox = rx - grid_w / T::from(2).unwrap();
                let oy = ry - grid_h / T::from(2).unwrap();

                for (k, &i) in cluster.iter().enumerate() {
                    let sc = k % sub_cols;
                    let sr = k / sub_cols;
                    let w = db.instances.sizes.w[i];
                    let h = db.instances.sizes.h[i];
                    let cx = ox + cell_w * T::from(sc).unwrap() + cell_w / T::from(2).unwrap()
                        - w / T::from(2).unwrap();
                    let cy = oy + cell_h * T::from(sr).unwrap() + cell_h / T::from(2).unwrap()
                        - h / T::from(2).unwrap();
                    ps.instances.x[i] = cx.clamp(T::zero(), db.diearea.width() - w);
                    ps.instances.y[i] = cy.clamp(T::zero(), db.diearea.height() - h);
                }
            }
        }

        let co = Computation::new(pc, hc);

        Self {
            db,
            bins,
            sc,
            ne,
            co,
            ps,
            best_hpwl: T::from(f32::MAX).unwrap(),
            best_overflow: T::one(),
            best_coords: None,
            best_density_weight: T::zero(),
            best_gamma: T::zero(),
            target_overflow: T::from(target_overflow).unwrap(),
        }
    }

    pub(crate) fn step(&mut self) -> bool {
        time_it("step", || {
            let ComputationResult { hpwl, overflow } = self.ne.step_bb(
                self.db,
                &self.bins,
                &mut self.sc,
                &mut self.ps,
                &mut self.co,
            );

            // Track best solution — only after warmup and when overflow is below target
            let overflow_threshold = self.target_overflow;
            if self.sc.iteration >= 50 && overflow < overflow_threshold && hpwl < self.best_hpwl {
                self.best_hpwl = hpwl;
                self.best_overflow = overflow;
                self.best_coords = Some(self.ps.instances.clone());
                self.best_density_weight = self.sc.density_weight;
                self.best_gamma = self.sc.gamma;
            }

            let iteration_log = format!(
                "[{:03}] - HPWL {:08e} - Overflow {:08e} - DW {:08e} - G {:08e}\n{}\n",
                self.sc.iteration,
                hpwl,
                overflow,
                self.sc.density_weight,
                self.sc.gamma,
                "=".repeat(40)
            );

            if self.sc.iteration % 100 == 0 {
                log::info!("{}", iteration_log);
            } else if self.sc.iteration % 10 == 0 {
                log::debug!("{}", iteration_log);
            }

            self.sc.update(hpwl, overflow);

            self.sc.should_stop()
        })
    }

    pub(crate) fn restore_best(&mut self) {
        if let Some(coords) = self.best_coords.take() {
            log::info!(
                "Restoring best solution with HPWL {:08e} (overflow {:08e})",
                self.best_hpwl,
                self.best_overflow
            );
            self.ps.instances = coords;
        }
    }
}

impl<T> fmt::Display for Placement<'_, T>
where
    T: Numeric,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.bins)
    }
}
