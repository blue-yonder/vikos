//! A machine learning library for supervised regression trainings
//!
//! This library wants to enable its user to write teachers
//! independent of the model trained or the cost function tried to
//! minimize.
//! Consequently its three most important traits are `Model`, `Cost`
//! and `Training`.
//!
//! # Examples
//!
//! Estimate mean
//!
//! ```
//! use vikos::{model, cost, teacher, teach_history, Model};
//! //We are only training a constant so our feature vector is empty
//! //Any other vector would be fine, too.
//! let features = ();
//! let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9
//! // Give each 'truth' its feature vector and stretch our history to 100 elements
//! let history = history.iter().cycle().map(|&y|((),y)).take(100);
//!
//! let cost = cost::LeastSquares{};
//! let mut model = model::Constant::new(0.0);
//!
//! let teacher = teacher::GradientDescentAl{ l0 : 0.3, t : 4.0 };
//! teach_history(&teacher, &cost, &mut model, history);
//! println!("{}", model.predict(&features));
//! ```
#![warn(missing_docs)]

extern crate num;

use std::iter::IntoIterator;
use num::Float;

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

    /// Error type used by the cost function
    ///
    /// Usually `f64` or `f32`
    type Error : Float;

    /// Value of the cost function derived by the n-th coefficent at x expressed in Error(x) and dY(x)/dx
    ///
    /// This method is called by SGD based training algorithm in order to
    /// determine the delta of the coefficents
    fn gradient(&self, prediction : Self::Error, truth : Self::Error, gradient_error_by_coefficent : Self::Error) -> Self::Error;
}

/// Teaches event to a `Model`
pub trait Training{

    /// `Model` changed by this `Training`
    ///
    /// A `Training` is strictly associated with a `Model` type. One could
    /// even argue that an instance of `Training` strictly belongs to an
    /// instance of `Model`
    type Model : Model;

    /// Changes `model`s coefficents so they minimize the `cost` function (hopefully)
    fn teach_event<C>(
        &mut self, cost : &C, model : &mut Self::Model,
        features : &<Self::Model as Model>::Input,
        truth : <Self::Model as Model>::Target
    ) where
        C : Cost<Error=<Self::Model as Model>::Target>;
}

/// `Teachers` are used to train `Models`, based on events and a `Cost` function
pub trait Teacher<M : Model>{

    /// Contains state which changes during the training, but is not required by the expert
    ///
    /// Examples are the velocity of the coefficents (in SGD) or the number of events already learned.
    /// This may also be empty
    type Training : Training<Model=M>;

    /// Creates a new `Training` object to hold training state
    fn new_training(&self, model : &M) -> Self::Training;
}

/// Teaches `model` all events in `history`
pub fn teach_history<M, C, T, H>(teacher : &T, cost : &C, model : &mut M, history : H)
    where M : Model,
    C : Cost<Error=M::Target>,
    T : Teacher<M>,
    H : IntoIterator<Item=(M::Input, M::Target)>
{
    let mut training = teacher.new_training(&model);
    for (features, truth) in history{

        training.teach_event(cost, model, &features, truth);
    }
}

/// Implementations of `Model` trait
pub mod model;
/// Implementations of `Cost` trait
pub mod cost;
/// Implementations of `Training` trait
///
/// You can use `Teacher`s to create `Training` instances
pub mod training;
/// Implementatios of `Teacher` trait
pub mod teacher;
/// Defines linear algebra traits used for some model parameters
pub mod linear_algebra;

#[cfg(test)]
mod tests {

    #[test]
    fn estimate_median() {

        use model::Constant;
        use cost::LeastAbsoluteDeviation;
        use training::GradientDescentAl;
        use Training;

        let features = ();
        let history = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //median is seven

        let cost = LeastAbsoluteDeviation{};
        let mut model = Constant::new(0.0);

        let mut training = GradientDescentAl{ l0 : 0.9, t : 9.0, learned_events : 0.0 };

        for &truth in history.iter().cycle().take(150){

            training.teach_event(&cost, &mut model, &features, truth);
            println!("model: {:?}, learning_rate: {:?}", model, training.learning_rate());
        }

        assert!(model.c < 7.1);
        assert!(model.c > 6.9);
    }

    #[test]
    fn estimate_mean() {

        use model::Constant;
        use cost::LeastSquares;
        use training::GradientDescentAl;
        use Training;

        let features = ();
        let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9

        let cost = LeastSquares{};
        let mut model = Constant::new(0.0);

        let mut training = GradientDescentAl{ l0 : 0.3, t : 4.0, learned_events : 0.0 };

        for &truth in history.iter().cycle().take(100){

            training.teach_event(&cost, &mut model, &features, truth);
            println!("model: {:?}, learning_rate: {:?}", model, training.learning_rate());
        }

        assert!(model.c < 9.1);
        assert!(model.c > 8.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent() {

        use cost::LeastSquares;
        use model::Linear;
        use teacher::GradientDescent;
        use teach_history;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let mut model = Linear{m : 0.0, c : 0.0};

        let teacher = GradientDescent{ learning_rate : 0.2 };

        let cost = LeastSquares{};
        teach_history(&teacher, &cost, &mut model, history.iter().cycle().take(20).cloned());

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent_iter() {

        use model::Linear;
        use cost::LeastSquares;
        use teacher::GradientDescent;
        use Teacher;
        use Training;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let cost = LeastSquares{};
        let mut model = Linear{m : 0.0, c : 0.0};

        let teacher = GradientDescent{ learning_rate : 0.2 };
        let mut training = teacher.new_training(&model);

        for &(features, truth) in history.iter().cycle().take(20){

            training.teach_event(&cost, &mut model, &features, truth);
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
        use teacher::Momentum;
        use teach_history;

        let history = [([0.0, 7.0], 17.0), ([1.0, 2.0], 8.0), ([2.0, -2.0], 1.0)];
        let mut model = Linear{m : [0.0, 0.0], c : 0.0};
        let cost = LeastSquares{};
        let teacher = Momentum{ l0 : 0.01, t : 10000000.0, inertia : 0.9 };

        teach_history(&teacher, &cost, &mut model, history.iter().cycle().take(15000).cloned());

        println!("{:?}", model);

        assert!(model.m[0] < 1.1);
        assert!(model.m[0] > 0.9);
        assert!(model.m[1] < 2.1);
        assert!(model.m[1] > 1.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn logistic_sgd_2d_least_squares(){
        use cost::LeastSquares;
        use model::{Logicstic, Linear};
        use teacher::GradientDescent;
        use teach_history;

        use Model;

        let history = [
            ([2.7, 2.5], 0.0),
            ([1.4, 2.3], 0.0),
            ([3.3, 4.4], 0.0),
            ([1.3, 1.8], 0.0),
            ([3.0, 3.0], 0.0),
            ([7.6, 2.7], 1.0),
            ([5.3, 2.0], 1.0),
            ([6.9, 1.7], 1.0),
            ([8.6, -0.2], 1.0),
            ([7.6, 3.5], 1.0)
        ];

        let mut model = Logicstic{ linear: Linear{m : [0.0, 0.0], c : 0.0}};
        let teacher = GradientDescent{ learning_rate : 0.3 };
        let cost = LeastSquares{};

        teach_history(
            &teacher, &cost, &mut model,
            history.iter().cycle().take(40).cloned(),
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
        use teacher::GradientDescent;
        use teach_history;
        use Model;

        let history = [
            ([2.7, 2.5], 0.0),
            ([1.4, 2.3], 0.0),
            ([3.3, 4.4], 0.0),
            ([1.3, 1.8], 0.0),
            ([3.0, 3.0], 0.0),
            ([7.6, 2.7], 1.0),
            ([5.3, 2.0], 1.0),
            ([6.9, 1.7], 1.0),
            ([8.6, -0.2], 1.0),
            ([7.6, 3.5], 1.0)
        ];

        let mut model = Logicstic{ linear: Linear{m : [0.0, 0.0], c : 0.0}};
        let teacher = GradientDescent{ learning_rate : 0.3 };
        let cost = MaxLikelihood{};

        teach_history(
            &teacher, &cost, &mut model,
            history.iter().cycle().take(20).cloned(),
        );

        println!("{:?}", model.linear);

        let classification_errors = history.iter()
            .map(|&(input, truth)| model.predict(&input).round() == truth)
            .fold(0, |errors, correct| if correct { errors } else { errors + 1 });

        assert_eq!(0, classification_errors);
    }
}
