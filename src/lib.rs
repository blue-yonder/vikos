//! A machine learning library for supervised regression trainings
//!
//! This library wants to enable its users to write teachers
//! independently of the model trained or the cost function that is meant to
//! be minimized. To get started right away, you may want to
//! have a look at the [tutorial](./tutorial/index.html).
//!
//! # Design
//! The three most important traits are [Model], [Cost],
//! and [Training]. [Teacher]s act as factories for [Training]s and
//! hold parameters which do not change during learning.
//!
//! [Model]: ./trait.Model.html
//! [Cost]: ./trait.Cost.html
//! [Training]: ./trait.Cost.html
//! [Teacher]: ./trait.Teacher.html

#![warn(missing_docs)]
#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate num;

use std::iter::IntoIterator;
use num::Float;

/// A Model is a parameterized expert algorithm
///
/// Implementations of this trait can be found in
/// [models](./model/index.html)
pub trait Model: Clone {
    /// Input features
    type Input;
    /// Target type
    type Target: Float;

    /// Predicts a target for the inputs based on the internal coefficents
    fn predict(&self, &Self::Input) -> Self::Target;

    /// The number of internal coefficents this model depends on
    fn num_coefficents(&self) -> usize;

    /// Value predict derived by the n-th `coefficent` at `input`
    fn gradient(&self, coefficent: usize, input: &Self::Input) -> Self::Target;

    /// Mutable reference to the n-th `coefficent`
    fn coefficent(&mut self, coefficent: usize) -> &mut Self::Target;
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
    /// Error type used by the cost function
    ///
    /// Usually `f64` or `f32`
    type Error: Float;

    /// Value of the gradient of the cost function (i.e. the cost function
    /// derived by the n-th coefficent at x expressed in Error(x) and dY(x)/dx
    ///
    /// This method is called by stochastic gradient descent (SGD)-based
    /// training algorithm in order to determine the delta of the coefficents
    ///
    /// Implementors of this trait should implement `Cost::outer_derivative` and not overwrite this
    /// method.
    fn gradient(&self,
                prediction: Self::Error,
                truth: Truth,
                derivative_of_model: Self::Error)
                -> Self::Error {
        self.outer_derivative(prediction, truth) * derivative_of_model
    }

    /// The outer derivative of the cost function with respect to the prediction.
    fn outer_derivative(&self, prediction: Self::Error, truth: Truth) -> Self::Error;

    /// Value of the cost function.
    fn cost(&self, prediction: Self::Error, truth: Truth) -> Self::Error;
}

/// Algorithms used to adapt [Model](./trait.Model.html) coefficents
///
/// Implementations of this trait may hold mutable state during
/// learning. You find training algorithms in [training](./training/index.html)
pub trait Training {
    /// `Model` changed by this `Training`
    ///
    /// A `Training` is strictly associated with a `Model` type. One could
    /// even argue that an instance of `Training` strictly belongs to an
    /// instance of `Model`
    type Model: Model;

    /// Changes `model`s coefficents so they minimize the `cost` function (hopefully)
    fn teach_event<C, Truth>(&mut self,
                             cost: &C,
                             model: &mut Self::Model,
                             features: &<Self::Model as Model>::Input,
                             truth: Truth)
        where C: Cost<Truth, Error = <Self::Model as Model>::Target>,
              Truth: Copy;
}

/// Factories for [Training](./trait.Training.html)
pub trait Teacher<M: Model> {
    /// Contains state which changes during the training, but is not required by the expert
    ///
    /// Examples are the velocity of the coefficents (in stochastic gradient
    /// descent) or the number of events already learned.
    /// This may also be empty
    type Training: Training<Model = M>;

    /// Creates a new `Training` object to hold training state
    fn new_training(&self, model: &M) -> Self::Training;
}

/// Teaches `model` all events in `history`
pub fn learn_history<M, C, T, H, Truth>(teacher: &T, cost: &C, model: &mut M, history: H)
    where M: Model,
          C: Cost<Truth, Error = M::Target>,
          T: Teacher<M>,
          H: IntoIterator<Item = (M::Input, Truth)>,
          Truth: Copy
{
    let mut training = teacher.new_training(model);
    for (features, truth) in history {

        training.teach_event(cost, model, &features, truth);
    }
}

/// Implementations of `Model` trait
pub mod model;
/// Implementations of `Cost` trait
pub mod cost;
/// Implementations of `Training` trait
pub mod training;
/// Implementations of `Teacher` trait
pub mod teacher;
/// Defines linear algebra traits used for some model parameters
pub mod linear_algebra;
pub mod tutorial;
