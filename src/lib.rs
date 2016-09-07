//! A machine learning library for supervised regression trainings
//!
//! This library wants to enable its user to write training algorithms
//! independent of the model trained or the cost function tried to
//! minimize.
//! Consequently its two main traits are currently `Model` and `Cost`.
//! The two submodules `model` and `cost` provide ready to use
//! implementations of said traits.

#![warn(missing_docs)]

extern crate num;

use std::iter::Iterator;
use num::{Zero, One, Float};

/// A Model is defines how to predict a target from an input
///
/// A model usually depends on several coefficents whose values
/// are derived using a training algorithm
pub trait Model : Clone{
    /// Input features
    type Input;
    /// Target type
    type Target : Float;

    /// Predicts a target for the inputs based on the internal coefficents
    fn predict(&self, &Self::Input) -> Self::Target;

    /// The number of internal coefficents this model depends on
    fn num_coefficents(&self) -> usize;

    /// Value predict derived by the n-th `coefficent` at `input`
    fn gradient(&self, coefficent : usize, input : &Self::Input) -> Self::Target;

    /// Mutable reference to the n-th `coefficent`
    fn coefficent(& mut self, coefficent : usize) -> & mut Self::Target;
}

/// Cost functions those value is supposed be minimized by the training algorithm
pub trait Cost{

    /// Type used for truth by the cost function
    ///
    /// Usually `f64` or `bool`
    type Truth : Copy;

    /// Error type used by the cost function
    ///
    /// Usually `f64` or `f32`
    type Error : Float;

    /// Value of the cost function derived by the n-th coefficent at x expressed in Error(x) and dY(x)/dx
    ///
    /// This method is called by SGD based training algorithm in order to
    /// determine the delta of the coefficents
    fn gradient(&self, prediction : Self::Error, truth : Self::Truth, gradient_error_by_coefficent : Self::Error) -> Self::Error;
}

/// Changes all coefficents of model based on their derivation of the cost function at features
///
/// Will not get stuck on saddle points as easily as a plain SGD and will converge quicker in general.
/// A good default for `inertia` is 0.9
pub fn inert_gradient_descent_step<C, M>(
    cost : &C,
    model : &mut M,
    features : &M::Input,
    truth : C::Truth,
    learning_rate : M::Target,
    inertia : M::Target,
    velocity : & mut Vec<M::Target>
)
    where C : Cost, M : Model<Target=C::Error>
{
    let inv_inertia = M::Target::one() - inertia;
    let prediction = model.predict(&features);

    for ci in 0..model.num_coefficents(){

        velocity[ci] = inertia * velocity[ci] - inv_inertia * learning_rate * cost.gradient(prediction, truth, model.gradient(ci, features));
        *model.coefficent(ci) = *model.coefficent(ci) + velocity[ci];
    }
}

/// An SGD training step with a velocity term
///
/// Can be used to implement stochastic or batch gradient descent
pub fn gradient_descent_step<C, M>(cost : &C, model : &mut M, features : &M::Input, truth : C::Truth, learning_rate : M::Target)
    where C : Cost, M : Model<Target=C::Error>
{
    let prediction = model.predict(&features);

    for ci in 0..model.num_coefficents(){
        *model.coefficent(ci) = *model.coefficent(ci) - learning_rate * cost.gradient(prediction, truth, model.gradient(ci, features));
    }
}

/// Applies a plain SGD training step to model once for every event in history using a constant learning rate
pub fn stochastic_gradient_descent<C, M, H>(cost : &C, start : M, history : H, learning_rate : M::Target) -> M
    where C : Cost,
    M : Model<Target=C::Error>,
    H : Iterator<Item=(M::Input, C::Truth)>
{

    let mut next = start.clone();
    for (features, truth) in history{

        gradient_descent_step(cost, & mut next, &features, truth, learning_rate);
    }

    next
}

/// SGD tranining with constant learning rate and velocity
///
/// Velocity avoids being stuck on saddle points during optimization
/// A good default for `inertia` is 0.9
pub fn inert_stochastic_gradient_descent<C, M, H>(
    cost : &C,
    start : M,
    history : H,
    learning_rate : M::Target,
    inertia : M::Target
) -> M
    where C : Cost,
    M : Model<Target=C::Error>,
    H : Iterator<Item=(M::Input, C::Truth)>
{

    let mut velocity = Vec::new();
    velocity.resize(start.num_coefficents(), M::Target::zero());
    let mut next = start.clone();
    for (features, truth) in history{

        inert_gradient_descent_step(cost, & mut next, &features, truth, learning_rate, inertia, & mut velocity);
    }

    next
}

/// Implementations of `Model` trait
pub mod model;
/// Implementations of `Cost` trait
pub mod cost;
/// Defines linear algebra traits used for some model parameters
pub mod linear_algebra;

#[cfg(test)]
mod tests {

    #[test]
    fn estimate_median() {

        use model::Constant;
        use cost::LeastAbsoluteDeviation;
        use gradient_descent_step;

        let features = ();
        let history = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //median is seven

        let cost = LeastAbsoluteDeviation{};
        let mut model = Constant::new(0.0);

        let learning_rate_start = 0.4;
        let learning_rate_stop = 0.001;
        let num_steps = 200;
        let learning_rate_gradient = (learning_rate_start - learning_rate_stop) / (num_steps as f64);

        for (count_step, &truth) in history.iter().cycle().take(num_steps).enumerate(){

            let adapted_learning_rate = learning_rate_stop + learning_rate_gradient * (num_steps - count_step) as f64;
            gradient_descent_step(&cost, & mut model, &features, truth, adapted_learning_rate);
            println!("model: {:?}, learning_rate: {:?}", model, adapted_learning_rate);
        }

        assert!(model.c < 7.1);
        assert!(model.c > 6.9);
    }

    #[test]
    fn estimate_mean() {

        use model::Constant;
        use cost::LeastSquares;
        use gradient_descent_step;

        let features = ();
        let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9

        let cost = LeastSquares{};
        let mut model = Constant::new(0.0);

        let learning_rate_start = 0.4;
        let learning_rate_stop = 0.001;
        let num_steps = 60;
        let learning_rate_gradient = (learning_rate_start - learning_rate_stop) / (num_steps as f64);

        for (count_step, &truth) in history.iter().cycle().take(num_steps).enumerate(){

            let adapted_learning_rate = learning_rate_stop + learning_rate_gradient * (num_steps - count_step) as f64;
            gradient_descent_step(&cost, & mut model, &features, truth, adapted_learning_rate);
            println!("model: {:?}, learning_rate: {:?}", model, adapted_learning_rate);
        }

        assert!(model.c < 9.1);
        assert!(model.c > 8.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent() {

        use cost::LeastSquares;
        use model::Linear;
        use stochastic_gradient_descent;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let start = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        let cost = LeastSquares{};
        let model = stochastic_gradient_descent(&cost, start, history.iter().cycle().take(20).cloned(), learning_rate);

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent_iter() {

        use model::Linear;
        use gradient_descent_step;
        use cost::LeastSquares;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let cost = LeastSquares{};
        let mut model = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        for &(features, truth) in history.iter().cycle().take(20){

            gradient_descent_step(&cost, & mut model, &features, truth, learning_rate);
            println!("model: {:?}", model);
        }

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_sgd_2d(){
        use cost::LeastSquares;
        use model::Linear;
        use inert_stochastic_gradient_descent;

        let history = [([0.0, 7.0], 17.0), ([1.0, 2.0], 8.0), ([2.0, -2.0], 1.0)];

        let start = Linear{m : [0.0, 0.0], c : 0.0};

        let learning_rate = 0.1;

        let cost = LeastSquares{};
        let model = inert_stochastic_gradient_descent(
            &cost, start,
            history.iter().cycle().take(15000).cloned(),
            learning_rate, 0.9
        );

        println!("{:?}", model);

        assert!(model.m[0] < 1.1);
        assert!(model.m[0] > 0.9);
        assert!(model.m[1] < 2.1);
        assert!(model.m[1] > 1.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn logistic_sgd_2d_least_squaers(){
        use cost::LeastSquares;
        use model::{Logicstic, Linear};
        use stochastic_gradient_descent;

        use Model;

        let history = [
            ([2.7810836, 2.550537003], 0.0),
            ([1.465489372, 2.362125076], 0.0),
            ([3.396561688, 4.400293529], 0.0),
            ([1.38807019, 1.850220317], 0.0),
            ([3.06407232, 3.005305973], 0.0),
            ([7.627531214, 2.759262235], 1.0),
            ([5.332441248, 2.088626775], 1.0),
            ([6.922596716, 1.77106367], 1.0),
            ([8.675418651, -0.242068655], 1.0),
            ([7.673756466, 3.508563011], 1.0)
        ];

        let start = Logicstic{ linear: Linear{m : [0.0, 0.0], c : 0.0}};

        let learning_rate = 0.3;

        let cost = LeastSquares{};
        let model = stochastic_gradient_descent(
            &cost, start,
            history.iter().cycle().take(40).cloned(),
            learning_rate
        );

        println!("{:?}", model.linear);

        let classification_errors = history.iter()
            .map(|&(input, truth)| model.predict(&input).round() == truth)
            .fold(0, |errors, correct| if correct { errors } else { errors + 1 });

        assert_eq!(0, classification_errors);
    }

    #[test]
    fn logistic_sgd_2d_max_likelihood(){
        use cost::MaxLikelihood;
        use model::{Logicstic, Linear};
        use stochastic_gradient_descent;

        use Model;

        let history = [
            ([2.7810836, 2.550537003], false),
            ([1.465489372, 2.362125076], false),
            ([3.396561688, 4.400293529], false),
            ([1.38807019, 1.850220317], false),
            ([3.06407232, 3.005305973], false),
            ([7.627531214, 2.759262235], true),
            ([5.332441248, 2.088626775], true),
            ([6.922596716, 1.77106367], true),
            ([8.675418651, -0.242068655], true),
            ([7.673756466, 3.508563011], true)
        ];

        let start = Logicstic{ linear: Linear{m : [0.0, 0.0], c : 0.0}};

        let learning_rate = 0.3;

        let cost = MaxLikelihood::<bool>::new();
        let model = stochastic_gradient_descent(
            &cost, start,
            history.iter().cycle().take(20).cloned(),
            learning_rate
        );

        println!("{:?}", model.linear);

        let classification_errors = history.iter()
            .map(|&(input, truth)| {
                model.predict(&input).round() == if truth { 1.0 } else { 0.0 }
            })
            .fold(0, |errors, correct| if correct { errors } else { errors + 1 });

        assert_eq!(0, classification_errors);
    }
}
