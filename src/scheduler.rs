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
    T: Numeric + std::fmt::Debug,
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
    T: Numeric + std::fmt::Debug,
{
    pub(crate) iteration: usize,
    pub(crate) gamma: T,
    pub(crate) base_gamma: T,
    pub(crate) density_weight: T,
    hpwl_metrics: Metric<T>,
    overflow_metrics: Metric<T>,
}
impl<T> Scheduler<T>
where
    T: Numeric + std::fmt::Debug,
{
    pub(crate) fn new(bins: &Bins<T>) -> Self {
        let hpwl_metrics = Metric::new();
        let overflow_metrics = Metric::new();
        let iteration = 0;
        let gamma = T::from(4.0).unwrap() * (bins.bin_size.width + bins.bin_size.height);
        let density_weight = T::from(8e-5).unwrap();
        Self {
            iteration,
            gamma,
            base_gamma: gamma,
            density_weight,
            hpwl_metrics,
            overflow_metrics,
        }
    }

    pub(crate) fn update(&mut self, hpwl: T, overflow: T) {
        self.increase_iteration();
        self.update_metrics(hpwl, overflow);
        self.update_density_weight();
        self.update_gamma();
    }

    pub(crate) fn should_stop(&self) -> bool {
        let hpwl_curr = self.hpwl_metrics.current;
        let hpwl_prev = self.hpwl_metrics.previous;
        let zero = T::zero();
        let two = T::from(1.5).unwrap();
        hpwl_prev != zero && hpwl_curr != zero && hpwl_prev > hpwl_curr * two
    }

    fn increase_iteration(&mut self) {
        self.iteration += 1;
    }

    fn update_metrics(&mut self, hpwl: T, overflow: T) {
        self.hpwl_metrics.update(hpwl);
        self.overflow_metrics.update(overflow);
    }

    fn update_density_weight(&mut self) {
        let ref_hpwl = self
            .hpwl_metrics
            .initial
            .unwrap_or_else(|| T::from(350000).unwrap());

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
