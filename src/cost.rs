use Cost;

/// Pass an instance of this Type to a training algorithm to optimize for C=Error^2
///
/// Uses f64 as Error type
pub struct LeastSquares;

impl Cost for LeastSquares{

    type Error = f64;

    fn gradient(&self, error : f64, gradient_error_by_coefficent : f64) -> f64
    {
        2.0 * error * gradient_error_by_coefficent
    }
}