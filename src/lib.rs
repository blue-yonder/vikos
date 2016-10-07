//! A machine learning library for supervised regression trainings
//!
//! This library wants to enable its users to write teachers
//! independently of the model trained or the cost function that is meant to
//! be minimized. To get started right away, you may want to
//! have a look at the [tutorial](./tutorial/index.html).

#![warn(missing_docs)]
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate rustc_serialize;
extern crate num;

use std::iter::IntoIterator;

/// Allows accessing and changing coefficients
pub trait Model {
    /// The number of internal coefficients this model depends on
    fn num_coefficients(&self) -> usize;

    /// Mutable reference to the n-th `coefficient`
    fn coefficient(&mut self, coefficient: usize) -> &mut f64;
}

/// A parameterized expert algorithm
///
/// Implementations of this trait can be found in
/// [models](./model/index.html)
pub trait Expert<X>: Model {
    /// Type returned by the expert algorithm
    type Prediction;
    /// Type holding the gradients of the coefficents
    type Gradient: linear_algebra::Vector;

    /// Predicts a target for the inputs based on the internal coefficents
    fn predict(&self, &X) -> Self::Prediction;

    /// Value predict derived by the n-th `coefficent` at `input`
    fn gradient(&self, input: &X) -> Self::Gradient;
}

/// Representing a cost function whose value is supposed be minimized by the
/// training algorithm.
///
/// The cost function is a quantity that describes how deviations of the
/// prediction from the true, observed target values should be penalized during
/// the optimization of the prediction.
///
/// Algorithms like stochastic gradient descent use the gradient of the cost
/// function. When calculating the gradient, it is important to apply the
/// outer-derivative of the cost function to the prediction, with the
/// inner-derivative of the model to the coefficient changes (chain-rule of
/// calculus). This inner-derivative must be supplied as the argument
/// `derivative_of_model` to `Cost::gradient`.
///
/// Implementations of this trait can be found in
/// [cost](./cost/index.html)
pub trait Cost<P, T = P> {

    /// The outer derivative of the cost function with respect to the prediction.
    fn outer_derivative(&self, prediction: P, truth: T) -> f64;

    /// Value of the cost function.
    fn cost(&self, prediction: P, truth: T) -> f64;
}

/// Algorithms used to adapt [Model](./trait.Model.html) coefficients
pub trait Teacher<M: Model> {
    /// Contains state which changes during the training, but is not part of the expertise
    ///
    /// Examples are the velocity of the coefficients (in stochastic gradient
    /// descent) or the number of events already learned.
    /// This may also be empty
    type Training;

    /// Creates an instance holding all mutable state of the algorithm
    fn new_training(&self, model: &M) -> Self::Training;

    /// Changes `model`s coefficients so they minimize the `cost` function (hopefully)
    fn teach_event<X, Y, C>(&self,
                            training: &mut Self::Training,
                            model: &mut M,
                            cost: &C,
                            features: &X,
                            truth: Y)
        where C: Cost<M::Prediction, Y>,
              Y: Copy,
              M: Expert<X>;
}

/// Teaches `model` all events in `history`
pub fn learn_history<X, M, C, T, H, Truth>(teacher: &T, cost: &C, model: &mut M, history: H)
    where M: Expert<X>,
          C: Cost<M::Prediction, Truth>,
          T: Teacher<M>,
          H: IntoIterator<Item = (X, Truth)>,
          Truth: Copy
{
    let mut training = teacher.new_training(model);
    for (features, truth) in history {

        teacher.teach_event(&mut training, model, cost, &features, truth);
    }
}

/// Implementations of `Model` trait
pub mod model;
/// Implementations of `Cost` trait
pub mod cost;
pub mod training;
pub mod teacher;
/// Defines linear algebra traits used for some model parameters
pub mod linear_algebra;
pub mod tutorial;
