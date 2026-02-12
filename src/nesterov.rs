use parking_lot::Mutex;
use rayon::prelude::*;
use reda_db::{DieArea, Numeric, VecCoords, VecSizes};

#[derive(Debug)]
pub(crate) struct Nesterov<T: Numeric> {
    prev_coords: VecCoords<T>,
    prev_grads: VecCoords<T>,

    // Nesterov state
    alpha: T,
    beta: T,
}
impl<T: Numeric> Nesterov<T> {
    pub(crate) fn new(instances_coords: &VecCoords<T>) -> Self {
        let zero = T::zero();
        let prev_grads = VecCoords::new_zero(instances_coords.len());
        Nesterov {
            prev_coords: instances_coords.clone(),
            prev_grads,
            alpha: T::one(),
            beta: zero,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step_adaptive(
        &mut self,
        iteration: usize,
        num_movable: usize,
        density_weight: &mut T,
        hpwl_grad_x: &[T],
        hpwl_grad_y: &[T],
        density_grad_x: &[T],
        density_grad_y: &[T],
        instances_sizes: &VecSizes<T>,
        instances_num_pins: &[T],
        diearea: &DieArea<T>,
        curr_coords: &mut VecCoords<T>,
    ) {
        // Update Nesterov momentum
        self.update_momentum();

        let zero = T::zero();
        let one = T::one();

        if iteration == 0 {
            let mut num = T::zero();
            let mut denom = T::zero();
            for i in 0..num_movable {
                num = num + hpwl_grad_x[i].abs() + hpwl_grad_y[i].abs();
                denom = denom + density_grad_x[i].abs() + density_grad_y[i].abs();
            }
            *density_weight = *density_weight * (num / denom);
        }

        let dw = *density_weight;

        // Step 1: Compute combined gradient for each node
        let mut grad_x = vec![zero; num_movable];
        let mut grad_y = vec![zero; num_movable];

        let mut num = T::zero();
        let mut denom = T::zero();

        {
            let x = &mut curr_coords.x;
            let y = &mut curr_coords.y;
            let x_prev = &mut self.prev_coords.x;
            let y_prev = &mut self.prev_coords.y;
            let grad_x_prev = &mut self.prev_grads.x;
            let grad_y_prev = &mut self.prev_grads.y;

            for i in 0..num_movable {
                let w = instances_sizes.w[i];
                let h = instances_sizes.h[i];
                let area = w * h;
                let num_pins = instances_num_pins[i];
                let inv_precond = one / (num_pins + dw * area).min(one);

                let gx = (hpwl_grad_x[i] + dw * density_grad_x[i]) * inv_precond;
                let gy = (hpwl_grad_y[i] + dw * density_grad_y[i]) * inv_precond;

                grad_x[i] = gx;
                grad_y[i] = gy;

                let dx = x[i] - x_prev[i];
                let dy = y[i] - y_prev[i];
                let dgx = grad_x[i] - grad_x_prev[i];
                let dgy = grad_y[i] - grad_y_prev[i];

                num = num + dx * dx + dy * dy;
                denom = denom + dgx * dgx + dgy * dgy;
            }
        }

        // Step 2: Adaptive alpha
        if iteration > 0 && denom > zero {
            self.alpha = (num / denom).sqrt();
        }
        // fallback: keep previous alpha if denom == 0

        let mutex_x: Mutex<&mut [T]> = Mutex::new(&mut curr_coords.x);
        let mutex_y: Mutex<&mut [T]> = Mutex::new(&mut curr_coords.y);
        let mutex_x_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_coords.x);
        let mutex_y_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_coords.y);
        let mutex_grad_x_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_grads.x);
        let mutex_grad_y_prev: Mutex<&mut [T]> = Mutex::new(&mut self.prev_grads.y);

        // Step 3: Apply Nesterov update with adaptive alpha
        (0..num_movable).into_par_iter().for_each(|i| {
            let x: &mut [T] = unsafe { &mut *mutex_x.data_ptr() };
            let y: &mut [T] = unsafe { &mut *mutex_y.data_ptr() };
            let x_prev: &mut [T] = unsafe { &mut *mutex_x_prev.data_ptr() };
            let y_prev: &mut [T] = unsafe { &mut *mutex_y_prev.data_ptr() };
            let grad_x_prev: &mut [T] = unsafe { &mut *mutex_grad_x_prev.data_ptr() };
            let grad_y_prev: &mut [T] = unsafe { &mut *mutex_grad_y_prev.data_ptr() };

            let w = instances_sizes.w[i];
            let h = instances_sizes.h[i];

            // Save current position
            let x_old = x[i];
            let y_old = y[i];

            // Lookahead point
            let x_hat = x[i] + self.beta * (x[i] - x_prev[i]);
            let y_hat = y[i] + self.beta * (y[i] - y_prev[i]);

            // Gradient step
            let new_x = (x_hat - self.alpha * grad_x[i]).clamp(zero, diearea.width() - w - one);
            let new_y = (y_hat - self.alpha * grad_y[i]).clamp(zero, diearea.height() - h - one);

            // Update history
            x_prev[i] = x_old;
            y_prev[i] = y_old;

            // Apply new positions
            x[i] = new_x;
            y[i] = new_y;

            // Save gradient for next iteration
            grad_x_prev[i] = grad_x[i];
            grad_y_prev[i] = grad_y[i];
        });
    }

    fn update_momentum(&mut self) {
        let one = T::one();
        let two = one + one;
        let four = two + two;
        let a_prev = self.alpha;
        let new_alpha = (one + (four * a_prev * a_prev + one).sqrt()) / two;
        let new_beta = (a_prev - one) / new_alpha;

        self.alpha = new_alpha;
        self.beta = new_beta;
    }
}
