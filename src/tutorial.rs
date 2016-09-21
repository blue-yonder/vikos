//! A short tutorial on how to use vikos to solve the problem of supervised
//! machine learning: We want to predict values for a quantity (the target), and
//! we have some data that we can base our inference on (features). We have a
//! data set (a history), that consists of features and corresponding, *true* target values, so
//! that we have a base to learn about how the target relates to the feature data.
//!
//! # Tutorial
//! Look, a bunch of data! Let us do something with it.
//!
//! ```
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//! ```
//! The first elements of each tuple represent our *feature* vector,
//! the second elements represents the true (observed) *target* value
//! (aka *the truth*). We want to use a [Training](../trait.Training.html) to
//! find the coefficients of a  [Model](../trait.Model.html)
//! which minimizes a [Cost](../trait.Cost.html) function. Let us start with
//! finding the mean value of the truth.
//!
//! ## Estimating the mean target value
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! // mean is 9, but of course we do not know that yet
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! // The mean is just a simple number ...
//! let mut model = model::Constant::new(0.0);
//! // ... which minimizes the square error
//! let cost = cost::LeastSquares {};
//! // Use stochastic gradient descent with an annealed learning rate
//! let teacher = teacher::GradientDescentAl { l0: 0.3, t: 4.0 };
//! // Train 100 (admittedly repetitive) events
//! learn_history(&teacher,
//!               &cost,
//!               &mut model,
//!               history.iter().cycle().take(100).cloned());
//! // We need an input vector for predictions, the 42 will not influence the mean
//! println!("{}", model.predict(&42.0));
//! // Since we know the model's type is `Constant`, we could just access the members
//! println!("{}", model.c);
//! ```
//! As far as the mean is concerned, the first element of each tuple, i.e.,
//! the feature, is just ignored (because we use the
//! [Constant](../model/struct.Constant.html) model).  The code would also
//! compile if the first element would be an empty tuple or any other type for
//! that matter.
//!
//! ## Estimating the median target value
//!
//! If we want to estimate the median instead, we only need to change
//! our cost function, to that of an absolute error:
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//! // median is 7, but we don't know that yet of course
//!
//! // The median is just a simple number ...
//! let mut model = model::Constant::new(0.0);
//! // ... which minimizes the absolute error
//! let cost = cost::LeastAbsoluteDeviation {};
//! let teacher = teacher::GradientDescentAl { l0: 1.0, t: 9.0 };
//! learn_history(&teacher,
//!               &cost,
//!               &mut model,
//!               history.iter().cycle().take(100).cloned());
//! ```
//! Most notably we changed the cost function to train for the median. We also had to
//! increase our learning rate to be able to converge to `7` more quickly. Maybe we
//! should try a slightly more sophisticated `Teacher` algorithm.
//!
//! ## Estimating median again
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! // median is 7, but of course we do not know that yet
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! // The median is just a simple number ...
//! let mut model = model::Constant::new(0.0);
//! // ... which minimizes the absolute error
//! let cost = cost::LeastAbsoluteDeviation {};
//! // Use stochasic gradient descent with an annealed learning rate and momentum
//! let teacher = teacher::Momentum {
//!     l0: 1.0,
//!     t: 3.0,
//!     inertia: 0.9,
//! };
//! learn_history(&teacher,
//!               &cost,
//!               &mut model,
//!               history.iter().cycle().take(100).cloned());
//! println!("{}", model.predict(&42.0));
//! ```
//! The momentum term allowed us to drop our learning rate way quicker and to retrieve a
//! more precise result in the same number of iterations. The algorithms and their
//! parameters are not the point however — the important thing is we could switch them
//! quite easily and independently of both cost function and model. Speaking of which:
//! it is time to fit a straight line through our data points.
//!
//! ## Line of best fit
//! We now use a linear model
//!
//! ```
//! use vikos::{model, cost, teacher, learn_history, Model};
//! // Best described by 2 * m - 3
//! let history = [
//!    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
//!    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
//!    (16.0, 29.0)
//! ];
//!
//! let mut model = model::Linear { m: 0.0, c: 0.0 };
//! let cost = cost::LeastSquares {};
//! let teacher = teacher::Momentum {
//!     l0: 0.0001,
//!     t: 1000.0,
//!     inertia: 0.99,
//! };
//! learn_history(&teacher,
//!               &cost,
//!               &mut model,
//!               history.iter().cycle().take(500).cloned());
//! for &(input, truth) in history.iter() {
//!     println!("Input: {}, Truth: {}, Prediction: {}",
//!              input,
//!              truth,
//!              model.predict(&input));
//! }
//! println!("slope: {}, intercept: {}", model.m, model.c);
//! ```
//! # Summary
//!
//! Using Vikos, we can build a machine-learning model by composing
//! implementations of three aspects:
//!
//!  * the [Model](../trait.Model.html) describes how features and target
//!    relate to each other (and what kind of estimated parameters/coefficients
//!    mediate among the target and the feature space), the model is  fitted by
//!  * the training algorithm, modelled with the
//!    [Teacher](../trait.Teacher.html) trait, that contains the optimization algorithm minimizing
//!    the model coefficents.
//!  * the [Cost](../trait.Cost.html) "function" describes the function that
//!    should be minimized by the algorithm.
