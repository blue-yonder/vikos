//! A short tutorial on how to use vikos
//!
//! # Tutorial
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
//! let teacher = teacher::Momentum{ l0 : 0.0001, t : 1000.0, inertia : 0.99};
//! learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(500).cloned());
//! for &(input, truth) in history.iter(){
//!     println!("Input: {}, Truth: {}, Prediction: {}", input, truth, model.predict(&input));
//! }
//! println!("slope: {}, intercept: {}", model.m, model.c);
//! ```