//! Holds helper functionality for Teacher algorithms

/// Calculates annealed learning rate
///
/// Smaller `t` will decrease the learning rate faster
/// After `t` events the start learning rate will be a half of `start`,
/// after two times `t` events the learning rate will be one third of `start`,
/// and so on.
pub fn annealed_learning_rate(num_events : usize, start : f64, t : f64) -> f64{
    start / (1.0 + num_events as f64 / t)
}
