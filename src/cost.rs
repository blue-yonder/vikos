use Cost;

/// Pass an instance of this type to a training algorithm to optimize for C=Error^2
///
/// Optimizing a `model::Constant` for `LeastSquares` should yield the mean value
pub struct LeastSquares;

impl Cost<f64> for LeastSquares {
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: f64, gradient_error_by_coefficent: f64) -> f64 {
        let error = prediction - truth;
        2.0 * error * gradient_error_by_coefficent
    }
}

/// Pass an instance of this type to a training algorithm to optimize for C=|Error|
///
/// Optimizing a `model::Constant` for `LeastAbsoluteDeviation` should yield the median.
/// Gradient for error == 0 is set to 0
pub struct LeastAbsoluteDeviation;

impl Cost<f64> for LeastAbsoluteDeviation {
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: f64, gradient_error_by_coefficent: f64) -> f64 {
        let error = prediction - truth;
        if error > 0.0 {
            gradient_error_by_coefficent
        } else if error < 0.0 {
            -gradient_error_by_coefficent
        } else {
            0.0
        }
    }
}

/// Maximizes the likelihood function `L` by defining `C=-ln(L)`
///
/// You can use this function if your truth is a probability
/// (i.e., a value betwenn 0 and 1). Maximizing the likelihood
/// function is equivalent to minimizing the least square error,
/// yet this cost function has shown itself to converge quicker
/// for some problems.
pub struct MaxLikelihood;

impl Cost<f64> for MaxLikelihood {
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: f64, gradient_error_by_coefficent: f64) -> f64 {
        gradient_error_by_coefficent * ((1.0 - truth) / (1.0 - prediction) - truth / prediction)
    }
}

impl Cost<bool> for MaxLikelihood {
    type Error = f64;

    fn gradient(&self, prediction: f64, truth: bool, gradient_error_by_coefficent: f64) -> f64 {
        gradient_error_by_coefficent / if truth { -prediction } else { 1.0 - prediction }
    }
}
