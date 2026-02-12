use crate::{placement::PlacementSolution, utils::time_it};
use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::{Numeric, DB};

#[derive(Debug)]
pub(crate) struct HpwlComputation<T>
where
    T: Numeric + std::fmt::Debug,
{
    pins_x: Vec<T>, // used for temp computation
    pins_y: Vec<T>, // used for temp computation

    hpwls: Vec<T>,             // length = num_nets
    grad_pins_x: Vec<T>,       // grad x for pins
    grad_pins_y: Vec<T>,       // grad y for pins
    pub(crate) grad_x: Vec<T>, // grad x for instances
    pub(crate) grad_y: Vec<T>, // grad y for instances
}
impl<T> HpwlComputation<T>
where
    T: Numeric + std::fmt::Debug,
{
    pub(crate) fn new(db: &DB<T>) -> Self {
        let num_pins = db.netlist.pins.len();
        let num_nets = db.netlist.nets.len();
        let num_nodes = db.instances.len();
        let zero = T::zero();

        let mut pins_x = vec![zero; num_pins];
        let mut pins_y = vec![zero; num_pins];

        let hpwls = vec![zero; num_nets];

        let grad_pins_x = vec![zero; num_pins];
        let grad_pins_y = vec![zero; num_pins];
        let grad_x = vec![zero; num_nodes];
        let grad_y = vec![zero; num_nodes];

        // initialize pins (even the fixed ones)
        let instances_x = &db.instances.coords.x;
        let instances_y = &db.instances.coords.y;
        let macro_2_pins = &db.netlist.macro_2_pins;
        let offset_pins_x = &db.netlist.pins.x;
        let offset_pins_y = &db.netlist.pins.y;
        let mutex_pins_x: Mutex<&mut [T]> = Mutex::new(&mut pins_x);
        let mutex_pins_y: Mutex<&mut [T]> = Mutex::new(&mut pins_y);

        // TODO :for some reason biasing the net towards (0, 0) gives better results
        (0..num_nodes).into_par_iter().for_each(|inst_id| {
            let pins_x: &mut [T] = unsafe { &mut *mutex_pins_x.data_ptr() };
            let pins_y: &mut [T] = unsafe { &mut *mutex_pins_y.data_ptr() };

            let inst_x = unsafe { *instances_x.get_unchecked(inst_id) };
            let inst_y = unsafe { *instances_y.get_unchecked(inst_id) };

            for &pin_id in &macro_2_pins[inst_id] {
                let offset_pin_x = unsafe { *offset_pins_x.get_unchecked(pin_id) };
                let offset_pin_y = unsafe { *offset_pins_y.get_unchecked(pin_id) };

                unsafe { *pins_x.get_unchecked_mut(pin_id) = inst_x + offset_pin_x };
                unsafe { *pins_y.get_unchecked_mut(pin_id) = inst_y + offset_pin_y };
            }
        });

        Self {
            pins_x,
            pins_y,
            hpwls,
            grad_pins_x,
            grad_pins_y,
            grad_x,
            grad_y,
        }
    }

    pub(crate) fn reset(&mut self) {
        let zero = T::zero();
        self.hpwls.par_iter_mut().for_each(|v| *v = zero);
        self.grad_pins_x.par_iter_mut().for_each(|v| *v = zero);
        self.grad_pins_y.par_iter_mut().for_each(|v| *v = zero);
        self.grad_x.par_iter_mut().for_each(|v| *v = zero);
        self.grad_y.par_iter_mut().for_each(|v| *v = zero);
    }

    pub(crate) fn compute_pins_position(&mut self, db: &DB<T>, solution: &PlacementSolution<T>) {
        let instances_x = &solution.instances.x;
        let instances_y = &solution.instances.y;
        let offset_pins_x = &db.netlist.pins.x;
        let offset_pins_y = &db.netlist.pins.y;
        let macro_2_pins = &db.netlist.macro_2_pins;

        let mutex_pins_x: Mutex<&mut [T]> = Mutex::new(&mut self.pins_x);
        let mutex_pins_y: Mutex<&mut [T]> = Mutex::new(&mut self.pins_y);

        (0..db.num_movable).into_par_iter().for_each(|inst_id| {
            let pins_x: &mut [T] = unsafe { &mut *mutex_pins_x.data_ptr() };
            let pins_y: &mut [T] = unsafe { &mut *mutex_pins_y.data_ptr() };

            let inst_x = unsafe { *instances_x.get_unchecked(inst_id) };
            let inst_y = unsafe { *instances_y.get_unchecked(inst_id) };

            for &pin_id in &macro_2_pins[inst_id] {
                let offset_pin_x = unsafe { *offset_pins_x.get_unchecked(pin_id) };
                let offset_pin_y = unsafe { *offset_pins_y.get_unchecked(pin_id) };

                unsafe { *pins_x.get_unchecked_mut(pin_id) = inst_x + offset_pin_x };
                unsafe { *pins_y.get_unchecked_mut(pin_id) = inst_y + offset_pin_y };
            }
        });
    }

    pub(crate) fn compute_gradients(&mut self, db: &DB<T>) {
        let macro_2_pins = &db.netlist.macro_2_pins;
        let zero = T::zero();

        let mutex_grad_x: Mutex<&mut [T]> = Mutex::new(&mut self.grad_x);
        let mutex_grad_y: Mutex<&mut [T]> = Mutex::new(&mut self.grad_y);

        (0..db.num_movable).into_par_iter().for_each(|inst_id| {
            let grad_x: &mut [T] = unsafe { &mut *mutex_grad_x.data_ptr() };
            let grad_y: &mut [T] = unsafe { &mut *mutex_grad_y.data_ptr() };

            let mut gx = zero;
            let mut gy = zero;

            for &pin_id in &macro_2_pins[inst_id] {
                gx += unsafe { *self.grad_pins_x.get_unchecked(pin_id) };
                gy += unsafe { *self.grad_pins_y.get_unchecked(pin_id) };
            }

            unsafe { *grad_x.get_unchecked_mut(inst_id) = gx };
            unsafe { *grad_y.get_unchecked_mut(inst_id) = gy };
        });
    }
}

// TODO: many nets have less than 3 pins (optimize based on that)
pub(crate) fn compute_hpwl<T>(
    gamma: T,
    db: &DB<T>,
    solution: &PlacementSolution<T>,
    hc: &mut HpwlComputation<T>,
) -> T
where
    T: Numeric + std::fmt::Debug,
{
    hc.reset();

    time_it("compute pin pos", || {
        hc.compute_pins_position(db, solution);
    });

    let zero = T::zero();
    let one = T::one();
    let max_width = db.diearea.width();
    let max_height = db.diearea.height();
    let nets = &db.netlist.nets;
    let n_2nets = db.netlist.n_2nets;
    let pins_x = &mut hc.pins_x;
    let pins_y = &mut hc.pins_y;

    let def_min_x = max_width + one;
    let def_min_y = max_height + one;

    let inv_gamma = one / gamma;
    let mutex_grad_pins_x: Mutex<&mut [T]> = Mutex::new(&mut hc.grad_pins_x);
    let mutex_grad_pins_y: Mutex<&mut [T]> = Mutex::new(&mut hc.grad_pins_y);

    let hpwls: &mut [T] = &mut hc.hpwls;

    time_it("hpwl main loop", || {
        nets[..n_2nets]
            .par_iter()
            .zip(hpwls[..n_2nets].par_iter_mut())
            .for_each(|(net, hpwl)| {
                let grad_pins_x: &mut [T] = unsafe { &mut *mutex_grad_pins_x.data_ptr() };
                let grad_pins_y: &mut [T] = unsafe { &mut *mutex_grad_pins_y.data_ptr() };

                let p0 = net.pin_ids[0];
                let p1 = net.pin_ids[1];

                let (x0, y0, x1, y1) = unsafe {
                    (
                        *pins_x.get_unchecked(p0),
                        *pins_y.get_unchecked(p0),
                        *pins_x.get_unchecked(p1),
                        *pins_y.get_unchecked(p1),
                    )
                };

                let min_x = x0.min(x1);
                let max_x = x0.max(x1);
                let min_y = y0.min(y1);
                let max_y = y0.max(y1);

                *hpwl = (max_x - min_x) + (max_y - min_y);

                let ex0 = ((x0 - max_x) * inv_gamma).exp();
                let ex1 = ((x1 - max_x) * inv_gamma).exp();

                let enx0 = ((min_x - x0) * inv_gamma).exp();
                let enx1 = ((min_x - x1) * inv_gamma).exp();

                let exp_x_sum = ex0 + ex1;
                let exp_nx_sum = enx0 + enx1;

                let xexp_x_sum = x0 * ex0 + x1 * ex1;
                let xexp_nx_sum = x0 * enx0 + x1 * enx1;

                let ey0 = ((y0 - max_y) * inv_gamma).exp();
                let ey1 = ((y1 - max_y) * inv_gamma).exp();

                let eny0 = ((min_y - y0) * inv_gamma).exp();
                let eny1 = ((min_y - y1) * inv_gamma).exp();

                let exp_y_sum = ey0 + ey1;
                let exp_ny_sum = eny0 + eny1;

                let yexp_y_sum = y0 * ey0 + y1 * ey1;
                let yexp_ny_sum = y0 * eny0 + y1 * eny1;

                let b_x = inv_gamma / exp_x_sum;
                let a_x = (one - b_x * xexp_x_sum) / exp_x_sum;
                let b_nx = -inv_gamma / exp_nx_sum;
                let a_nx = (one - b_nx * xexp_nx_sum) / exp_nx_sum;

                let b_y = inv_gamma / exp_y_sum;
                let a_y = (one - b_y * yexp_y_sum) / exp_y_sum;
                let b_ny = -inv_gamma / exp_ny_sum;
                let a_ny = (one - b_ny * yexp_ny_sum) / exp_ny_sum;

                unsafe {
                    *grad_pins_x.get_unchecked_mut(p0) =
                        (a_x + b_x * x0) * ex0 - (a_nx + b_nx * x0) * enx0;

                    *grad_pins_x.get_unchecked_mut(p1) =
                        (a_x + b_x * x1) * ex1 - (a_nx + b_nx * x1) * enx1;

                    *grad_pins_y.get_unchecked_mut(p0) =
                        (a_y + b_y * y0) * ey0 - (a_ny + b_ny * y0) * eny0;

                    *grad_pins_y.get_unchecked_mut(p1) =
                        (a_y + b_y * y1) * ey1 - (a_ny + b_ny * y1) * eny1;
                }
            });

        nets[n_2nets..]
            .par_iter()
            .zip(hpwls[n_2nets..].par_iter_mut())
            .for_each(|(net, hpwl)| {
                let grad_pins_x: &mut [T] = unsafe { &mut *mutex_grad_pins_x.data_ptr() };
                let grad_pins_y: &mut [T] = unsafe { &mut *mutex_grad_pins_y.data_ptr() };

                let mut min_x = def_min_x;
                let mut max_x: T = zero;
                let mut min_y = def_min_y;
                let mut max_y: T = zero;

                for &pid in &net.pin_ids {
                    let pin_x = unsafe { *pins_x.get_unchecked(pid) };
                    let pin_y = unsafe { *pins_y.get_unchecked(pid) };

                    min_x = min_x.min(pin_x);
                    max_x = max_x.max(pin_x);

                    min_y = min_y.min(pin_y);
                    max_y = max_y.max(pin_y);
                }

                let mut xexp_x_sum = zero;
                let mut xexp_nx_sum = zero;
                let mut exp_x_sum = zero;
                let mut exp_nx_sum = zero;

                let mut yexp_y_sum = zero;
                let mut yexp_ny_sum = zero;
                let mut exp_y_sum = zero;
                let mut exp_ny_sum = zero;

                for &pid in &net.pin_ids {
                    let pin_x = unsafe { *pins_x.get_unchecked(pid) };
                    let pin_y = unsafe { *pins_y.get_unchecked(pid) };

                    let exp_x = ((pin_x - max_x) * inv_gamma).exp();
                    let exp_nx = ((min_x - pin_x) * inv_gamma).exp();

                    xexp_x_sum += pin_x * exp_x;
                    xexp_nx_sum += pin_x * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;

                    let exp_y = ((pin_y - max_y) * inv_gamma).exp();
                    let exp_ny = ((min_y - pin_y) * inv_gamma).exp();

                    yexp_y_sum += pin_y * exp_y;
                    yexp_ny_sum += pin_y * exp_ny;
                    exp_y_sum += exp_y;
                    exp_ny_sum += exp_ny;
                }

                *hpwl = (max_x - min_x) + (max_y - min_y);

                let b_x = inv_gamma / exp_x_sum;
                let a_x = (one - b_x * xexp_x_sum) / exp_x_sum;
                let b_nx = -inv_gamma / exp_nx_sum;
                let a_nx = (one - b_nx * xexp_nx_sum) / exp_nx_sum;

                let b_y = inv_gamma / exp_y_sum;
                let a_y = (one - b_y * yexp_y_sum) / exp_y_sum;
                let b_ny = -inv_gamma / exp_ny_sum;
                let a_ny = (one - b_ny * yexp_ny_sum) / exp_ny_sum;

                for &pid in &net.pin_ids {
                    let pin_x = unsafe { *pins_x.get_unchecked(pid) };
                    let pin_y = unsafe { *pins_y.get_unchecked(pid) };

                    let exp_x = ((pin_x - max_x) * inv_gamma).exp();
                    let exp_nx = ((min_x - pin_x) * inv_gamma).exp();

                    unsafe {
                        *grad_pins_x.get_unchecked_mut(pid) =
                            (a_x + b_x * pin_x) * exp_x - (a_nx + b_nx * pin_x) * exp_nx
                    };

                    let exp_y = ((pin_y - max_y) * inv_gamma).exp();
                    let exp_ny = ((min_y - pin_y) * inv_gamma).exp();

                    unsafe {
                        *grad_pins_y.get_unchecked_mut(pid) =
                            (a_y + b_y * pin_y) * exp_y - (a_ny + b_ny * pin_y) * exp_ny
                    };
                }
            });
    });

    time_it("compute hpwl gradient", || {
        hc.compute_gradients(db);
    });

    let hpwls = &mut hc.hpwls;
    hpwls.par_iter().copied().sum()
}
