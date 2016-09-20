use Cost;

/// Pass an instance of this type to a training algorithm to optimize for C=Error^2
///
/// Optimizing a `model::Constant` for `LeastSquares` should yield the mean value
pub struct LeastSquares;

impl Cost<f64> for LeastSquares {
    fn outer_derivative(&self, prediction: f64, truth: f64) -> f64 {
        let error = prediction - truth;
        2.0 * error
    }

    fn cost(&self, prediction: f64, truth: f64) -> f64 {
        (prediction - truth).powi(2)
    }
}

/// Pass an instance of this type to a training algorithm to optimize for C=|Error|
///
/// Optimizing a `model::Constant` for `LeastAbsoluteDeviation` should yield the median.
/// Gradient for error == 0 is set to 0
pub struct LeastAbsoluteDeviation;

impl Cost<f64> for LeastAbsoluteDeviation {
    fn outer_derivative(&self, prediction: f64, truth: f64) -> f64 {
        let error = prediction - truth;
        if error > 0.0 {
            1.0
        } else if error < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
    fn cost(&self, prediction: f64, truth: f64) -> f64 {
        (prediction - truth).abs()
    }
}

/// Maximizes the likelihood function `L` by defining `C=-ln(L)`
///
/// You can use this function if your truth is a probability
/// (i.e., a value between 0 and 1). Maximizing the likelihood
/// function is equivalent to minimizing the least square error,
/// yet this cost function has shown itself to converge quicker
/// for some problems.
///
/// #Examples
///
/// ```
/// use vikos::{learn_history, Model, teacher, cost};
/// use vikos::model::Logistic;
/// use std::default::Default;
///
/// let history = [([2.7, 2.5], false),
///                ([1.4, 2.3], false),
///                ([3.3, 4.4], false),
///                ([1.3, 1.8], false),
///                ([3.0, 3.0], false),
///                ([7.6, 2.7], true),
///                ([5.3, 2.0], true),
///                ([6.9, 1.7], true),
///                ([8.6, -0.2], true),
///                ([7.6, 3.5], true)];
///
/// let mut model = Logistic::default();
/// let teacher = teacher::GradientDescent { learning_rate: 0.3 };
/// let cost = cost::MaxLikelihood {};
///
/// learn_history(&teacher,
///               &cost,
///               &mut model,
///               history.iter().cycle().take(20).cloned());
/// ```
pub struct MaxLikelihood;

impl Cost<f64> for MaxLikelihood {
    fn outer_derivative(&self, prediction: f64, truth: f64) -> f64 {
        ((1.0 - truth) / (1.0 - prediction) - truth / prediction)
    }
    fn cost(&self, prediction: f64, truth: f64) -> f64 {
        -truth * prediction.ln() - (1.0 - truth) * (1.0 - prediction).ln()
    }
}

impl Cost<bool> for MaxLikelihood {
    fn outer_derivative(&self, prediction: f64, truth: bool) -> f64 {
        1. / if truth { -prediction } else { 1.0 - prediction }
    }
    fn cost(&self, prediction: f64, truth: bool) -> f64 {
        -(if truth { prediction } else { 1.0 - prediction }).ln()
    }
}

#[cfg(test)]
mod test{

    use super::super::Cost;
    use super::{LeastSquares, LeastAbsoluteDeviation, MaxLikelihood};

    // Approximates the derivation of the cost function
    fn approx_derivate<T : Copy>(cost : &Cost<T>, prediction : f64, truth : T) -> f64{
        let epsilon = 0.00001;
        let f_plus_epsilon = cost.cost(prediction + epsilon, truth);
        let f_minus_epsilon = cost.cost(prediction - epsilon, truth);
        println!("f_x_plus_epsilon: {}, f_x_minus_epsilon:: {}", f_plus_epsilon, f_minus_epsilon);
        (f_plus_epsilon - f_minus_epsilon) / (2.0 * epsilon)
    }

    // Returns absolute difference between derivate and approximation
    fn check_derivate<T : Copy>(cost : &Cost<T>, prediction : f64, truth : T) -> f64{
        let derivate = cost.outer_derivative(prediction, truth);
        let approx = approx_derivate(cost, prediction, truth);
        println!("derivation: {}, approximation: {}", derivate, approx);
        (derivate - approx).abs()
    }

    #[test]
    fn least_squares_derivation(){

        let cost = LeastSquares{};
        assert!(check_derivate(&cost, 10.0, 12.0) < 0.001);
    }

    #[test]
    fn least_absolute_derivation(){

        let cost = LeastAbsoluteDeviation{};
        assert!(check_derivate(&cost, 0.0, 0.0) < 0.001);
        assert!(check_derivate(&cost, 1.0, 0.0) < 0.001);
        assert!(check_derivate(&cost, -1.0, 0.0) < 0.001);
    }

    #[test]
    fn neg_log_likelihood_derivation(){

        let cost = MaxLikelihood{};
        assert!(check_derivate(&cost, 0.2, false) < 0.001);
        assert!(check_derivate(&cost, 0.8, true) < 0.001);
        assert!(check_derivate(&cost, 0.2, 0.0) < 0.001);
        assert!(check_derivate(&cost, 0.8, 1.0) < 0.001);
        assert_eq!(cost.outer_derivative(0.2, false), cost.outer_derivative(0.2, 0.0));
        assert_eq!(cost.outer_derivative(0.8, true), cost.outer_derivative(0.8, 1.0));
    }
}