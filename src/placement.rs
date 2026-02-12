use std::fmt;

use crate::{
    epotential::{compute_potential, PotentialComputation},
    hpwl::{compute_hpwl, HpwlComputation},
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
    T: Numeric + std::fmt::Debug,
{
    pub(crate) fn new(db: &DB<T>) -> Self {
        let num_bins: Coord<usize> = Coord::new(512, 512);

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
    pub(crate) fn new(db: &DB<T>) -> Self {
        let num_nodes = db.instances.len();
        let zero = T::zero();
        let two = T::from(2).unwrap();
        let center_x = db.diearea.width() / two;
        let center_y = db.diearea.height() / two;
        let mut new_instances_x = vec![zero; num_nodes];
        let mut new_instances_y = vec![zero; num_nodes];
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
                *new_x = center_x - w / two;
                *new_y = center_y - h / two;
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
pub(crate) struct Placement<'a, T>
where
    T: Numeric,
{
    pub(crate) db: &'a DB<T>,
    pub(crate) bins: Bins<T>,
    pub(crate) pc: PotentialComputation<T>,
    pub(crate) hc: HpwlComputation<T>,
    pub(crate) ps: PlacementSolution<T>,
    pub(crate) sc: Scheduler<T>,
    pub(crate) ne: Nesterov<T>,
}
impl<'a, T> Placement<'a, T>
where
    T: Numeric,
{
    pub(crate) fn new(db: &'a DB<T>) -> Self {
        let num_threads = rayon::current_num_threads();
        let bins = Bins::new(db);
        let m = bins.num_bins.x;
        let n = bins.num_bins.y;
        let ps = PlacementSolution::new(db);
        let pc = PotentialComputation::new(db, &bins, ps.target_density, m, n, num_threads);
        let hc = HpwlComputation::new(db);
        let sc = Scheduler::new(&bins);
        let ne = Nesterov::new(&db.instances.coords);
        Self {
            db,
            bins,
            pc,
            ps,
            hc,
            sc,
            ne,
        }
    }

    pub(crate) fn step(&mut self) -> bool {
        let hpwl = time_it("hpwl", || {
            compute_hpwl(self.sc.gamma, self.db, &self.ps, &mut self.hc)
        });

        let (_, overflow) = time_it("compute potential", || {
            compute_potential(
                self.sc.iteration,
                self.db.num_movable,
                self.ps.target_density,
                self.db.movable_area,
                &self.bins,
                &self.ps,
                &mut self.pc,
            )
        });

        time_it("nesterov", || {
            self.ne.step_adaptive(
                self.sc.iteration,
                self.db.num_movable,
                &mut self.sc.density_weight,
                &self.hc.grad_x,
                &self.hc.grad_y,
                &self.pc.grad_x,
                &self.pc.grad_y,
                &self.db.instances.sizes,
                &self.db.instances.num_pins,
                &self.db.diearea,
                &mut self.ps.instances,
            );
        });

        if self.sc.iteration % 10 == 0 {
            log::info!(
                "[{:03}] - HPWL {:08e} - Overflow {:08e} - DW {:08e} - G {:08e}",
                self.sc.iteration,
                hpwl,
                overflow,
                self.sc.density_weight,
                self.sc.gamma
            );
            log::info!("{}\n", "=".repeat(40));
        }

        self.sc.update(hpwl, overflow);

        self.sc.should_stop()
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
