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

/// A parameterized expert algorithm
///
/// Implementations of this trait can be found in
/// [models](./model/index.html)
pub trait Model{
    /// Input from which to predict the target
    type Features;

    /// The number of internal coefficients this model depends on
    fn num_coefficients(&self) -> usize;

    /// Mutable reference to the n-th `coefficient`
    fn coefficient(&mut self, coefficient: usize) -> &mut f64;

    /// Predicts a target for the inputs based on the internal coefficients
    fn predict(&self, &Self::Features) -> f64;

    /// Value predict derived by the n-th `coefficient` at `input`
    fn gradient(&self, coefficient: usize, input: &Self::Features) -> f64;
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
pub trait Cost<Truth> {
    /// Value of the gradient of the cost function (i.e. the cost function
    /// derived by the n-th coefficient at x expressed in Error(x) and dY(x)/dx
    ///
    /// This method is called by stochastic gradient descent (SGD)-based
    /// training algorithm in order to determine the delta of the coefficients
    ///
    /// Implementors of this trait should implement `Cost::outer_derivative` and not overwrite this
    /// method.
    fn gradient(&self, prediction: f64, truth: Truth, derivative_of_model: f64) -> f64 {
        self.outer_derivative(prediction, truth) * derivative_of_model
    }

    /// The outer derivative of the cost function with respect to the prediction.
    fn outer_derivative(&self, prediction: f64, truth: Truth) -> f64;

    /// Value of the cost function.
    fn cost(&self, prediction: f64, truth: Truth) -> f64;
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
    fn teach_event<Y, C>(&self,
                            training: &mut Self::Training,
                            model: &mut M,
                            cost: &C,
                            features: &M::Features,
                            truth: Y)
        where C: Cost<Y>,
              Y: Copy;
}

/// Teaches `model` all events in `history`
pub fn learn_history<M, C, T, H, Truth>(teacher: &T, cost: &C, model: &mut M, history: H)
    where M: Model,
          C: Cost<Truth>,
          T: Teacher<M>,
          H: IntoIterator<Item = (M::Features, Truth)>,
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
