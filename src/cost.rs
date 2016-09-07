use Cost;

use std::marker::PhantomData;

/// Pass an instance of this Type to a training algorithm to optimize for C=Error^2
///
/// Optimizing a `model::Constant` for LeastSquares should yield the mean value
pub struct LeastSquares;

impl Cost for LeastSquares{

    type Truth = f64;
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: f64, gradient_error_by_coefficent : f64) -> f64
    {
        let error = prediction - truth;
        2.0 * error * gradient_error_by_coefficent
    }
}

/// Pass an instance of this Type to a training algorithm to optimize for C=|Error|
///
/// Optimizing a `model::Constant` for LeastSquares should yield the median
/// Gradient for error == 0 is set to 0
pub struct LeastAbsoluteDeviation;

impl Cost for LeastAbsoluteDeviation{

    type Truth = f64;
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: f64, gradient_error_by_coefficent: f64) -> f64
    {
        let error = prediction - truth;
        if error > 0.0 {
            gradient_error_by_coefficent
        } else if error < 0.0{
            - gradient_error_by_coefficent
        } else {
            0.0
        }
    }
}

/// Maximies the likelihood function `L` by defining `C=-ln(L)`
///
/// This cost function is best used to optimize propabilities
pub struct MaxLikelihood<Truth>{
    _truth_type : PhantomData<Truth>
}

impl<T> MaxLikelihood<T>{
    /// Returns a `MaxLikelihood` instance
    pub fn new() -> Self{
        MaxLikelihood{ _truth_type : PhantomData::<T>{} }
    }
}

impl Cost for MaxLikelihood<f64>{

    type Truth = f64;
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: f64, gradient_error_by_coefficent: f64) -> f64{
        gradient_error_by_coefficent * ((1.0 - truth) / (1.0 - prediction) - truth / prediction)
    }
}

impl Cost for MaxLikelihood<bool>{

    type Truth = bool;
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: bool, gradient_error_by_coefficent: f64) -> f64{
        let outer_derivation = if truth { - 1.0 / prediction } else { 1.0 / (1.0 - prediction)};
        gradient_error_by_coefficent * outer_derivation
    }
}