//! A machine learning library for supervised regression trainings
//!
//! This library wants to enable its user to write teachers
//! independent of the model trained or the cost function tried to
//! minimize.
//!
//! # Tutorial
//!
//! Look, a bunch of data! Let's do something with it.
//!
//! ```
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//! ```
//! The first element of the tuple is our feature vector,
//! the second elements represents the truth. We want to
//! use a [Training] to find the coefficents of a [Model]
//! which minimize a `Cost` function. Let's start with
//! finding the mean value of the truth.
//!
//! ## Estimating mean
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! //mean is 9, but we don't know that yet of course
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! // The mean is just a simple number ...
//! let mut model = model::Constant::new(0.0);
//! // ... which minimizes the square error
//! let cost = cost::LeastSquares{};
//! // Use Stochasic Gradient Descent with an annealed learning rate
//! let teacher = teacher::GradientDescentAl{ l0 : 0.3, t : 4.0 };
//! // Train 100 (admitettly repetitive) events
//! learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(100).cloned());
//! // We need an input vector for predictions, the 42 won't influence the mean
//! println!("{}", model.predict(&42.0));
//! // Since we know models type is `Constant` we could just access the members
//! println!("{}", model.c);
//! ```
//! As far as the mean is concerned the first element is just
//! ignored. The code would also compile if the first
//! element would be an empty tuple or any other type for
//! that matter.
//!
//! ## Estimating median
//!
//! If we want to estimate the median instead we only need to change
//! our cost function
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! //median is 7, but we don't know that yet of course
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! // The median is just a simple number ...
//! let mut model = model::Constant::new(0.0);
//! // ... which minimizes the absolute error
//! let cost = cost::LeastAbsoluteDeviation{};
//! let teacher = teacher::GradientDescentAl{ l0 : 1.0, t : 9.0 };
//! learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(100).cloned());
//! ```
//! Most notably we changed the cost function to train for the median. We also had to
//! increase our learning rate to be able to converge to `7` more quickly. Maybe we
//! should try a slightly more sophisticated `Training` algorithm.
//!
//! ## Estimating median again
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! //median is 7, but we don't know that yet of course
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! // The median is just a simple number ...
//! let mut model = model::Constant::new(0.0);
//! // ... which minimizes the absolute error
//! let cost = cost::LeastAbsoluteDeviation{};
//! // Use Stochasic Gradient Descent with an annealed learning rate and momentum
//! let teacher = teacher::Momentum{ l0 : 1.0, t : 3.0, inertia : 0.9};
//! learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(100).cloned());
//! println!("{}", model.predict(&42.0));
//! ```
//! The momentum term allowed us to drop our learning rate way quicker and to retrieve a
//! more precise result in the same number of iterations. The algorithms and their
//! parameters are not the point however, the important thing is we could switch them
//! quite easily and independent of our cost function and our model. Speaking of which,
//! it is time to fit a straight line through our data points.
//!
//! ## Line of best fit
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! //Best described by 2 * m - 3
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! let mut model = model::Linear{ m : 0.0, c : 0.0 };
//! let cost = cost::LeastSquares{};
//! let teacher = teacher::Momentum{ l0 : 0.001, t : 1000.0, inertia : 0.9};
//! learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(500).cloned());
//! for &(input, truth) in history.iter(){
//!     println!("Input: {}, Truth: {}, Prediction: {}", input, truth, model.predict(&input));
//! }
//! println!("slope: {}, intercept: {}", model.m, model.c);
//! ```
//! # Desgin
//! Consequently its three most important traits are [Model], [Cost]
//! and [Training]. [Teacher]s act as factories for [Trainings] and
//! as of now are largely convinience.
//! [Model]: ./trait.Model.html
//! [Cost]: ./trait.Cost.html
//! [Training]: ./trait.Cost.html
//! [Teacher]: ./trait.Teacher.html
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
pub fn learn_history<M, C, T, H>(teacher : &T, cost : &C, model : &mut M, history : H)
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
