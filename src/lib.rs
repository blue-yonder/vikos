//! A machine learning library for supervised regression and classifaction
//!
//! This library wants to enable its users to write models independently of the teacher used for
//! training or the cost function that is meant to be minimized. To get started right away, you may
//! want to have a look at the [tutorial](./tutorial/index.html).

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
pub trait Model {
    /// Input from which to predict the target
    type Features;

    /// f64 for Regressors or binary classifiers. For multi classification an array of `f64` with a
    /// dimension equal to the number of classes.
    type Target;

    /// The number of internal coefficients this model depends on
    fn num_coefficients(&self) -> usize;

    /// Mutable reference to the n-th `coefficient`
    fn coefficient(&mut self, coefficient: usize) -> &mut f64;

    /// Predicts a target for the inputs based on the internal coefficients
    fn predict(&self, &Self::Features) -> Self::Target;

    /// Value predict derived by the n-th `coefficient` at `input`
    fn gradient(&self, coefficient: usize, input: &Self::Features) -> Self::Target;
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
pub trait Cost<Truth, Target = f64> {
    /// The outer derivative of the cost function with respect to the prediction.
    fn outer_derivative(&self, prediction: &Target, truth: Truth) -> Target;

    /// Value of the cost function.
    fn cost(&self, prediction: Target, truth: Truth) -> f64;
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
        where C: Cost<Y, M::Target>,
              Y: Copy;
}

/// Define this trait over the target type of a classifier, to convert it into its truth type
///
/// i.e. for a binary classifier returning a `f64` this returns a `bool` which is true if the
/// prediction is greater `0.5`. For a classifier returning an `[f64;n]` it returns the index of
/// the largest discriminant, which should be equal to the class index.
pub trait Crisp {
    /// The crisp type of the prediction.
    ///
    /// Called `Truth` since it is identical to the `Truth` type used during learning. Although
    /// the instances returned by crisp are obviously still a prediction, just their type is
    /// identical to that of the truth.
    type Truth;

    /// Return crisp prediction
    fn crisp(&self) -> Self::Truth;
}

/// Teaches `model` all events in `history`
pub fn learn_history<M, C, T, H, Truth>(teacher: &T, cost: &C, model: &mut M, history: H)
    where M: Model,
          C: Cost<Truth, M::Target>,
          T: Teacher<M>,
          H: IntoIterator<Item = (M::Features, Truth)>,
          Truth: Copy
{
    let mut training = teacher.new_training(model);
    for (features, truth) in history {

        teacher.teach_event(&mut training, model, cost, &features, truth);
    }
}
mod array;
/// Implementations of `Model` trait
pub mod model;
/// Implementations of `Cost` trait
pub mod cost;
pub mod teacher;
pub mod crisp;
/// Defines linear algebra traits used for some model parameters
pub mod linear_algebra;
pub mod tutorial;
