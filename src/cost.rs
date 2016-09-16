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
/// use vikos::model::{Logistic, Linear};
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
/// let mut model = Logistic {
///     linear: Linear {
///         m: [0.0, 0.0],
///         c: 0.0,
///     },
/// };
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
