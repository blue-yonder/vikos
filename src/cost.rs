use Cost;

/// Pass an instance of this Type to a training algorithm to optimize for C=Error^2
///
/// Optimizing a `model::Constant` for LeastSquares should yield the mean value
pub struct LeastSquares;

impl Cost for LeastSquares{

    type Error = f64;

    fn gradient(&self, error : f64, gradient_error_by_coefficent : f64) -> f64
    {
        2.0 * error * gradient_error_by_coefficent
    }
}

/// Pass an instance of this Type to a training algorithm to optimize for C=|Error|
///
/// Optimizing a `model::Constant` for LeastSquares should yield the median
/// Gradient for error == 0 is set to 0
pub struct LeastAbsoluteDeviation;

impl Cost for LeastAbsoluteDeviation{

    type Error = f64;

    fn gradient(&self, error : f64, gradient_error_by_coefficent : f64) -> f64
    {
        if error > 0.0 {
            gradient_error_by_coefficent
        } else if error < 0.0{
            - gradient_error_by_coefficent
        } else {
            0.0
        }
    }
}