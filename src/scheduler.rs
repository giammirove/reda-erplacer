use reda_db::Numeric;

use crate::placement::Bins;

#[derive(Debug)]
pub(crate) struct Metric<T>
where
    T: Numeric + std::fmt::Debug,
{
    initial: Option<T>,
    current: T,
    previous: T,
}
impl<T> Metric<T>
where
    T: Numeric,
{
    pub(crate) fn new() -> Self {
        let zero = T::zero();
        Self {
            initial: None,
            current: zero,
            previous: zero,
        }
    }

    pub(crate) fn update(&mut self, metric: T) {
        if self.initial.is_none() {
            self.initial = Some(metric);
        }
        self.previous = self.current;
        self.current = metric;
    }
}

#[derive(Debug)]
pub(crate) struct Scheduler<T>
where
    T: Numeric,
{
    pub(crate) iteration: usize,
    pub(crate) gamma: T,
    pub(crate) base_gamma: T,
    pub(crate) density_weight: T,
    hpwl_metrics: Metric<T>,
    overflow_metrics: Metric<T>,
    converged_iter: Option<usize>,
    target_overflow: T,
}
impl<T> Scheduler<T>
where
    T: Numeric,
{
    pub(crate) fn new(bins: &Bins<T>, target_overflow: f32, initial_density_weight: f32) -> Self {
        let hpwl_metrics = Metric::new();
        let overflow_metrics = Metric::new();
        let iteration = 0;
        let gamma = T::from(4.0).unwrap() * (bins.bin_size.width + bins.bin_size.height);
        let density_weight = T::from(initial_density_weight).unwrap();
        Self {
            iteration,
            gamma,
            base_gamma: gamma,
            density_weight,
            hpwl_metrics,
            overflow_metrics,
            converged_iter: None,
            target_overflow: T::from(target_overflow).unwrap(),
        }
    }

    pub(crate) fn update(&mut self, hpwl: T, overflow: T) {
        self.increase_iteration();
        self.update_metrics(hpwl, overflow);
        self.check_convergence();
        self.update_density_weight();
        self.update_gamma();
    }

    pub(crate) fn should_stop(&self) -> bool {
        if self.iteration < 50 {
            return false;
        }

        match self.converged_iter {
            Some(ci) => self.iteration >= ci + 50,
            None => false,
        }
    }

    fn check_convergence(&mut self) {
        let overflow_threshold = self.target_overflow;
        if self.converged_iter.is_none()
            && self.overflow_metrics.current < overflow_threshold
            && self.overflow_metrics.current > T::zero()
        {
            self.converged_iter = Some(self.iteration);
        }
    }

    fn increase_iteration(&mut self) {
        self.iteration += 1;
    }

    fn update_metrics(&mut self, hpwl: T, overflow: T) {
        // Delay initial HPWL capture to avoid inflated values from noisy starts
        if self.hpwl_metrics.initial.is_none() && self.iteration >= 10 {
            self.hpwl_metrics.initial = Some(hpwl);
        }
        self.hpwl_metrics.previous = self.hpwl_metrics.current;
        self.hpwl_metrics.current = hpwl;
        self.overflow_metrics.update(overflow);
    }

    fn update_density_weight(&mut self) {
        let ref_hpwl = match self.hpwl_metrics.initial {
            Some(v) if v > T::zero() => v,
            _ => return, // skip update until we have a valid initial HPWL
        };

        let lower = T::from(0.95).unwrap();
        let upper = T::from(1.05).unwrap();
        let zero = T::zero();

        let delta_hpwl = self.hpwl_metrics.current - self.hpwl_metrics.previous;

        let power = if delta_hpwl < zero {
            T::from(0.9999)
                .unwrap()
                .powi(self.iteration as i32)
                .max(T::from(0.98).unwrap())
        } else {
            upper.powf(-delta_hpwl / ref_hpwl).clamp(lower, upper)
        };

        let mu = upper * power;

        self.density_weight = self.density_weight * mu;
    }

    fn update_gamma(&mut self) {
        let overflow = self.overflow_metrics.current;

        let coef = T::from(10).unwrap().powf(
            (overflow - T::from(0.1).unwrap()) * T::from(20).unwrap() / T::from(9).unwrap()
                - T::one(),
        );

        let min_coef = T::from(0.05).unwrap();
        let max_coef = T::from(10.0).unwrap();

        self.gamma = self.base_gamma * coef.clamp(min_coef, max_coef);
    }
}
