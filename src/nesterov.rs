use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::{Numeric, VecCoords, DB};

use crate::{
    placement::{Bins, Computation, ComputationResult, PlacementSolution},
    scheduler::Scheduler,
};

#[derive(Debug)]
pub(crate) struct Nesterov<T: Numeric> {
    grads: VecCoords<T>,
    prev_coords: VecCoords<T>,
    prev_grads: VecCoords<T>,

    // Nesterov state
    alpha: T,
    beta: T,
    // DREAMPlace-style adaptive preconditioner alpha
    precond_alpha: T,
    precond_iter: usize,
}
impl<T: Numeric> Nesterov<T> {
    pub(crate) fn new(instances_coords: &VecCoords<T>) -> Self {
        let zero = T::zero();
        let grads = VecCoords::new_zero(instances_coords.len());
        let prev_grads = VecCoords::new_zero(instances_coords.len());
        Nesterov {
            grads,
            prev_coords: instances_coords.clone(),
            prev_grads,
            alpha: T::one(),
            beta: zero,
            precond_alpha: T::one(),
            precond_iter: 0,
        }
    }

    pub(crate) fn step_bb(
        &mut self,
        db: &DB<T>,
        bins: &Bins<T>,
        sc: &mut Scheduler<T>,
        ps: &mut PlacementSolution<T>,
        co: &mut Computation<T>,
    ) -> ComputationResult<T> {
        let one = T::one();
        let zero = T::zero();

        let computation_result = co.compute(db, bins, sc, ps);

        if sc.iteration == 0 {
            let mut num = T::zero();
            let mut denom = T::zero();
            for i in 0..db.num_movable {
                num = num + co.hc.grad_x[i].abs() + co.hc.grad_y[i].abs();
                denom = denom + co.pc.grad_x[i].abs() + co.pc.grad_y[i].abs();
            }
            sc.density_weight = sc.density_weight * (num / denom);
        }

        let mut num = T::zero();
        let mut denom = T::zero();

        {
            let x = &mut ps.instances.x;
            let y = &mut ps.instances.y;
            let x_prev = &mut self.prev_coords.x;
            let y_prev = &mut self.prev_coords.y;
            let grad_x_prev = &mut self.prev_grads.x;
            let grad_y_prev = &mut self.prev_grads.y;

            for i in 0..db.num_movable {
                let w = db.instances.sizes.w[i];
                let h = db.instances.sizes.h[i];
                let area = w * h;
                let num_pins = db.instances.num_pins[i];

                // Split preconditioner: constant for WL, DW-scaled for density
                let wl_precond = (num_pins + area).max(one);
                let d_precond = (num_pins + sc.density_weight * area).max(one);
                let gx =
                    co.hc.grad_x[i] / wl_precond + sc.density_weight * co.pc.grad_x[i] / d_precond;
                let gy =
                    co.hc.grad_y[i] / wl_precond + sc.density_weight * co.pc.grad_y[i] / d_precond;

                self.grads.x[i] = gx;
                self.grads.y[i] = gy;

                let dx = x[i] - x_prev[i];
                let dy = y[i] - y_prev[i];
                let dgx = self.grads.x[i] - grad_x_prev[i];
                let dgy = self.grads.y[i] - grad_y_prev[i];

                num = num + dx * dx + dy * dy;
                denom = denom + dgx * dgx + dgy * dgy;
            }
        }

        // Step 2: Adaptive step size (Barzilai-Borwein)
        if sc.iteration > 0 && denom > zero {
            self.alpha = (num / denom).sqrt();
        }

        // DREAMPlace-style: double precond_alpha every 20 iters when overflow < 0.3
        self.precond_iter += 1;
        if computation_result.overflow < T::from(0.3).unwrap()
            && self.precond_iter % 20 == 0
            && self.precond_alpha < T::from(1024.0).unwrap()
        {
            self.precond_alpha = self.precond_alpha * T::from(2.0).unwrap();
            log::info!(
                "precond_alpha = {:.1e}",
                self.precond_alpha.to_f64().unwrap()
            );
        }
        // fallback: keep previous alpha if denom == 0

        let mutex_x: Mutex<&mut [T]> = Mutex::new(&mut ps.instances.x);
        let mutex_y: Mutex<&mut [T]> = Mutex::new(&mut ps.instances.y);
        let mutex_x_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_coords.x);
        let mutex_y_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_coords.y);
        let mutex_grad_x_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_grads.x);
        let mutex_grad_y_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_grads.y);

        // Step 3: Apply Nesterov update with adaptive alpha
        (0..db.num_movable).into_par_iter().for_each(|i| {
            let x: &mut [T] = unsafe { &mut *mutex_x.data_ptr() };
            let y: &mut [T] = unsafe { &mut *mutex_y.data_ptr() };
            let x_prev: &mut [T] = unsafe { &mut *mutex_x_prev.data_ptr() };
            let y_prev: &mut [T] = unsafe { &mut *mutex_y_prev.data_ptr() };
            let grad_x_prev: &mut [T] = unsafe { &mut *mutex_grad_x_prev.data_ptr() };
            let grad_y_prev: &mut [T] = unsafe { &mut *mutex_grad_y_prev.data_ptr() };

            let w = db.instances.sizes.w[i];
            let h = db.instances.sizes.h[i];

            // Save current position
            let x_old = x[i];
            let y_old = y[i];

            // Lookahead point
            let x_hat = x[i] + self.beta * (x[i] - x_prev[i]);
            let y_hat = y[i] + self.beta * (y[i] - y_prev[i]);

            // Gradient step
            let new_x =
                (x_hat - self.alpha * self.grads.x[i]).clamp(zero, db.diearea.width() - w - one);
            let new_y =
                (y_hat - self.alpha * self.grads.y[i]).clamp(zero, db.diearea.height() - h - one);

            // Update history
            x_prev[i] = x_old;
            y_prev[i] = y_old;

            // Apply new positions
            x[i] = new_x;
            y[i] = new_y;

            // Save gradient for next iteration
            grad_x_prev[i] = self.grads.x[i];
            grad_y_prev[i] = self.grads.y[i];
        });

        computation_result
    }
}
