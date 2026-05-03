use crate::{placement::PlacementSolution, utils::time_it};
use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::{Net, Numeric, VecCoords, DB};

#[derive(Debug)]
pub(crate) struct HpwlComputation<T>
where
    T: Numeric,
{
    hpwls: Vec<T>,             // length = num_nets
    grad_pins_x: Vec<T>,       // grad x for pins
    grad_pins_y: Vec<T>,       // grad y for pins
    pub(crate) grad_x: Vec<T>, // grad x for instances
    pub(crate) grad_y: Vec<T>, // grad y for instances
}
impl<T> HpwlComputation<T>
where
    T: Numeric,
{
    pub(crate) fn new(db: &DB<T>) -> Self {
        let num_pins = db.netlist.pins.len();
        let num_nets = db.netlist.nets.len();
        let num_nodes = db.instances.len();
        let zero = T::zero();

        let hpwls = vec![zero; num_nets];

        let grad_pins_x = vec![zero; num_pins];
        let grad_pins_y = vec![zero; num_pins];
        let grad_x = vec![zero; num_nodes];
        let grad_y = vec![zero; num_nodes];

        Self {
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
    }

    pub(crate) fn compute_gradients(&mut self, num_movable: usize, macro_2_pins: &Vec<Vec<usize>>) {
        let zero = T::zero();

        let mutex_grad_x: Mutex<&mut [T]> = Mutex::new(&mut self.grad_x);
        let mutex_grad_y: Mutex<&mut [T]> = Mutex::new(&mut self.grad_y);

        (0..num_movable).into_par_iter().for_each(|inst_id| {
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

    fn compute_hpwl_inner(
        &mut self,
        instances: &VecCoords<T>,
        pins: &VecCoords<T>,
        nets: &Vec<Net>,
        n_2nets: usize,
        pin_2_macro: &Vec<usize>,
        gamma: T,
        max_width: T,
        max_height: T,
    ) -> T {
        let zero = T::zero();
        let one = T::one();

        let instances_x = &instances.x;
        let instances_y = &instances.y;
        let offset_pins_x = &pins.x;
        let offset_pins_y = &pins.y;

        let mutex_grad_pins_x: Mutex<&mut [T]> = Mutex::new(&mut self.grad_pins_x);
        let mutex_grad_pins_y: Mutex<&mut [T]> = Mutex::new(&mut self.grad_pins_y);

        let inv_gamma = one / gamma;

        let def_min_x = max_width + one;
        let def_min_y = max_height + one;

        let hpwls: &mut [T] = &mut self.hpwls;

        // Inline helper to get pin position without pre-materialized array
        macro_rules! pin_x_y {
            ($pid:expr) => {
                unsafe {
                    let macro_id = *pin_2_macro.get_unchecked($pid);

                    (
                        *instances_x.get_unchecked(macro_id) + *offset_pins_x.get_unchecked($pid),
                        *instances_y.get_unchecked(macro_id) + *offset_pins_y.get_unchecked($pid),
                    )
                }
            };
        }

        nets[..n_2nets]
            .par_iter()
            .zip(hpwls[..n_2nets].par_iter_mut())
            .for_each(|(net, hpwl)| {
                let grad_pins_x: &mut [T] = unsafe { &mut *mutex_grad_pins_x.data_ptr() };
                let grad_pins_y: &mut [T] = unsafe { &mut *mutex_grad_pins_y.data_ptr() };

                let p0 = net.pin_ids[0];
                let p1 = net.pin_ids[1];

                let (x0, y0) = pin_x_y!(p0);
                let (x1, y1) = pin_x_y!(p1);

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
                    let (px, py) = pin_x_y!(pid);

                    min_x = min_x.min(px);
                    max_x = max_x.max(px);

                    min_y = min_y.min(py);
                    max_y = max_y.max(py);
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
                    let (px, py) = pin_x_y!(pid);

                    let exp_x = ((px - max_x) * inv_gamma).exp();
                    let exp_nx = ((min_x - px) * inv_gamma).exp();

                    xexp_x_sum += px * exp_x;
                    xexp_nx_sum += px * exp_nx;
                    exp_x_sum += exp_x;
                    exp_nx_sum += exp_nx;

                    let exp_y = ((py - max_y) * inv_gamma).exp();
                    let exp_ny = ((min_y - py) * inv_gamma).exp();

                    yexp_y_sum += py * exp_y;
                    yexp_ny_sum += py * exp_ny;
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
                    let (px, py) = pin_x_y!(pid);

                    let exp_x = ((px - max_x) * inv_gamma).exp();
                    let exp_nx = ((min_x - px) * inv_gamma).exp();

                    unsafe {
                        *grad_pins_x.get_unchecked_mut(pid) =
                            (a_x + b_x * px) * exp_x - (a_nx + b_nx * px) * exp_nx
                    };

                    let exp_y = ((py - max_y) * inv_gamma).exp();
                    let exp_ny = ((min_y - py) * inv_gamma).exp();

                    unsafe {
                        *grad_pins_y.get_unchecked_mut(pid) =
                            (a_y + b_y * py) * exp_y - (a_ny + b_ny * py) * exp_ny
                    };
                }
            });

        hpwls.iter().copied().sum()
    }

    pub(crate) fn compute_hpwl(
        &mut self,
        gamma: T,
        db: &DB<T>,
        solution: &PlacementSolution<T>,
    ) -> T
    where
        T: Numeric,
    {
        self.reset();

        let hpwl_sum = time_it("hpwl main loop", || {
            self.compute_hpwl_inner(
                &solution.instances,
                &db.netlist.pins,
                &db.netlist.nets,
                db.netlist.n_2nets,
                &db.netlist.pin_2_macro,
                gamma,
                db.diearea.width(),
                db.diearea.height(),
            )
        });

        time_it("compute hpwl gradient", || {
            self.compute_gradients(db.num_movable, &db.netlist.macro_2_pins);
        });

        hpwl_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reda_db::{Net, VecCoords};

    fn make_coords<T: Numeric>(x: Vec<T>, y: Vec<T>) -> VecCoords<T> {
        VecCoords { x, y }
    }

    fn make_nets(pin_ids: Vec<Vec<usize>>) -> Vec<Net> {
        pin_ids
            .into_iter()
            .map(|ids| Net { pin_ids: ids })
            .collect()
    }

    fn make_comp(num_pins: usize, num_nets: usize, num_nodes: usize) -> HpwlComputation<f64> {
        HpwlComputation {
            hpwls: vec![0.0; num_nets],
            grad_pins_x: vec![0.0; num_pins],
            grad_pins_y: vec![0.0; num_pins],
            grad_x: vec![0.0; num_nodes],
            grad_y: vec![0.0; num_nodes],
        }
    }

    // Two pins at (0,0) and (3,4), no offsets => HPWL = 3 + 4 = 7
    #[test]
    fn test_2pin_net_hpwl() {
        let mut comp = make_comp(2, 1, 2);
        let instances = make_coords(vec![0.0, 3.0], vec![0.0, 4.0]);
        let pins = make_coords(vec![0.0, 0.0], vec![0.0, 0.0]);
        let nets = make_nets(vec![vec![0, 1]]);
        let pin_2_macro = vec![0, 1];

        let hpwl =
            comp.compute_hpwl_inner(&instances, &pins, &nets, 1, &pin_2_macro, 1.0, 10.0, 10.0);

        assert_eq!(hpwl, 7.0_f64);
    }

    // Instance at (1,1) with pin offset (2,3) => effective (3,4)
    // Instance at (0,0) with pin offset (0,0) => effective (0,0)
    // HPWL = 3 + 4 = 7
    #[test]
    fn test_2pin_net_with_pin_offsets() {
        let mut comp = make_comp(2, 1, 2);
        let instances = make_coords(vec![1.0, 0.0], vec![1.0, 0.0]);
        let pins = make_coords(vec![2.0, 0.0], vec![3.0, 0.0]);
        let nets = make_nets(vec![vec![0, 1]]);
        let pin_2_macro = vec![0, 1];

        let hpwl =
            comp.compute_hpwl_inner(&instances, &pins, &nets, 1, &pin_2_macro, 1.0, 10.0, 10.0);

        assert_eq!(hpwl, 7.0_f64);
    }

    // Two pins at the same position => HPWL = 0
    #[test]
    fn test_2pin_net_zero_hpwl() {
        let mut comp = make_comp(2, 1, 2);
        let instances = make_coords(vec![5.0, 5.0], vec![5.0, 5.0]);
        let pins = make_coords(vec![0.0, 0.0], vec![0.0, 0.0]);
        let nets = make_nets(vec![vec![0, 1]]);
        let pin_2_macro = vec![0, 1];

        let hpwl =
            comp.compute_hpwl_inner(&instances, &pins, &nets, 1, &pin_2_macro, 1.0, 10.0, 10.0);

        assert_eq!(hpwl, 0.0_f64);
    }

    // 3-pin net: pins at (0,0), (4,0), (2,3) => HPWL = 4 + 3 = 7
    #[test]
    fn test_3pin_net_hpwl() {
        let mut comp = make_comp(3, 1, 3);
        let instances = make_coords(vec![0.0, 4.0, 2.0], vec![0.0, 0.0, 3.0]);
        let pins = make_coords(vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]);
        let nets = make_nets(vec![vec![0, 1, 2]]);
        let pin_2_macro = vec![0, 1, 2];

        let hpwl =
            comp.compute_hpwl_inner(&instances, &pins, &nets, 0, &pin_2_macro, 1.0, 10.0, 10.0);

        assert_eq!(hpwl, 7.0_f64);
    }

    // Net 0: (0,0)-(3,0) => 3
    // Net 1: (1,1)-(1,5) => 4
    // Total = 7
    #[test]
    fn test_multiple_2pin_nets() {
        let mut comp = make_comp(4, 2, 4);
        let instances = make_coords(vec![0.0, 3.0, 1.0, 1.0], vec![0.0, 0.0, 1.0, 5.0]);
        let pins = make_coords(vec![0.0; 4], vec![0.0; 4]);
        let nets = make_nets(vec![vec![0, 1], vec![2, 3]]);
        let pin_2_macro = vec![0, 1, 2, 3];

        let hpwl =
            comp.compute_hpwl_inner(&instances, &pins, &nets, 2, &pin_2_macro, 1.0, 10.0, 10.0);

        assert_eq!(hpwl, 7.0_f64);
    }

    // One 2-pin net and one 3-pin net processed together
    // Net 0 (2-pin): (0,0)-(3,0) => 3
    // Net 1 (3-pin): (0,0),(4,0),(2,3) => 4 + 3 = 7
    // Total = 10
    #[test]
    fn test_mixed_2pin_and_3pin_nets() {
        let mut comp = make_comp(5, 2, 5);
        let instances = make_coords(vec![0.0, 3.0, 0.0, 4.0, 2.0], vec![0.0, 0.0, 0.0, 0.0, 3.0]);
        let pins = make_coords(vec![0.0; 5], vec![0.0; 5]);
        let nets = make_nets(vec![vec![0, 1], vec![2, 3, 4]]);
        let pin_2_macro = vec![0, 1, 2, 3, 4];

        let hpwl =
            comp.compute_hpwl_inner(&instances, &pins, &nets, 1, &pin_2_macro, 1.0, 10.0, 10.0);

        assert_eq!(hpwl, 10.0_f64);
    }
}
