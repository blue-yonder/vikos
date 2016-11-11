//! Learning algorithms implementing `Teacher` trait

use Teacher;
use Model;
use Cost;

/// Calculates annealed learning rate
///
/// Smaller `t` will decrease the learning rate faster
/// After `t` events the start learning rate will be a half of `start`,
/// after two times `t` events the learning rate will be one third of `start`,
/// and so on.
fn annealed_learning_rate(num_events: usize, start: f64, t: f64) -> f64 {
    start / (1.0 + num_events as f64 / t)
}

/// Value of the gradient of the cost function (i.e. the cost function
/// derived by the n-th coefficient at x expressed in Error(x) and dY(x)/dx
///
/// This method is called by stochastic gradient descent (SGD)-based
/// training algorithm in order to determine the delta of the coefficients
fn gradient<T, C>(cost: &C, prediction: f64, truth: T, derivative_of_model: f64) -> f64
    where C: Cost<T>
{
    cost.outer_derivative(&prediction, truth) * derivative_of_model
}

/// Gradient descent
///
/// Simplest possible implementation of gradient descent with fixed learning rate
pub struct GradientDescent {
    /// Defines how fast the coefficients of the trained `Model` will change
    pub learning_rate: f64,
}

impl<M> Teacher<M> for GradientDescent
    where M: Model<Target = f64>
{
    type Training = ();

    fn new_training(&self, _: &M) -> () {
        ()
    }

    fn teach_event<Y, C>(&self,
                         _training: &mut (),
                         model: &mut M,
                         cost: &C,
                         features: &M::Features,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        let prediction = model.predict(features);

        for ci in 0..model.num_coefficients() {
            *model.coefficient(ci) =
                *model.coefficient(ci) -
                self.learning_rate *
                gradient(cost, prediction, truth, model.gradient(ci, features));
        }
    }
}

/// Gradient descent with annealing learning rate
///
/// For the i-th event the learning rate is `l = l0 * (1 + i/t)`
pub struct GradientDescentAl {
    /// Start learning rate
    pub l0: f64,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`
    /// and so on.
    pub t: f64,
}

impl<M> Teacher<M> for GradientDescentAl
    where M: Model<Target = f64>
{
    type Training = usize;

    fn new_training(&self, _: &M) -> usize {
        0
    }

    fn teach_event<Y, C>(&self,
                         num_events: &mut usize,
                         model: &mut M,
                         cost: &C,
                         features: &M::Features,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        let prediction = model.predict(features);
        let learning_rate = annealed_learning_rate(*num_events, self.l0, self.t);

        for ci in 0..model.num_coefficients() {
            *model.coefficient(ci) =
                *model.coefficient(ci) -
                learning_rate * gradient(cost, prediction, truth, model.gradient(ci, features));
        }
        *num_events += 1;
    }
}

/// Gradient descent with annealing learning rate and momentum
///
/// For the i-th event the learning rate is `l = l0 * (1 + i/t)`
pub struct Momentum {
    /// Start learning rate
    pub l0: f64,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`,
    /// and so on.
    pub t: f64,
    /// To simulate friction, please select a value smaller than 1 (recommended)
    pub inertia: f64,
}

impl<M> Teacher<M> for Momentum
    where M: Model<Target = f64>
{
    type Training = (usize, Vec<f64>);

    fn new_training(&self, model: &M) -> (usize, Vec<f64>) {

        let mut velocity = Vec::with_capacity(model.num_coefficients());
        velocity.resize(model.num_coefficients(), 0.0);

        (0, velocity)
    }

    fn teach_event<Y, C>(&self,
                         training: &mut (usize, Vec<f64>),
                         model: &mut M,
                         cost: &C,
                         features: &M::Features,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        // let (ref mut num_events, ref mut velocity) = *training; ok?
        let mut num_events = &mut training.0;
        let mut velocity = &mut training.1;
        let prediction = model.predict(features);
        let learning_rate = annealed_learning_rate(*num_events, self.l0, self.t);

        for ci in 0..model.num_coefficients() {
            velocity[ci] = self.inertia * velocity[ci] -
                           learning_rate *
                           gradient(cost, prediction, truth, model.gradient(ci, features));
            *model.coefficient(ci) = *model.coefficient(ci) + velocity[ci];
        }
        *num_events += 1;
    }
}

/// Nesterov accelerated gradient descent
///
/// Like accelerated gradient descent, Nesterov accelerated gradient descent
/// includes a momentum term. In contrast to regular gradient descent, the
/// acceleration is not calculated with respect to the current position, but to
/// the estimated new one.
/// Source:
/// [G. Hinton's lecture 6c]
/// (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
pub struct Nesterov {
    /// Start learning rate
    pub l0: f64,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`,
    /// and so on.
    pub t: f64,
    /// To simulate friction, please select a value smaller than 1 (recommended)
    pub inertia: f64,
}

impl<M> Teacher<M> for Nesterov
    where M: Model<Target = f64>
{
    type Training = (usize, Vec<f64>);

    fn new_training(&self, model: &M) -> (usize, Vec<f64>) {

        let mut velocity = Vec::with_capacity(model.num_coefficients());
        velocity.resize(model.num_coefficients(), 0.0);

        (0, velocity)
    }

    fn teach_event<Y, C>(&self,
                         training: &mut (usize, Vec<f64>),
                         model: &mut M,
                         cost: &C,
                         features: &M::Features,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        let mut num_events = &mut training.0;
        let mut velocity = &mut training.1;
        let prediction = model.predict(features);
        let learning_rate = annealed_learning_rate(*num_events, self.l0, self.t);

        for ci in 0..model.num_coefficients() {
            *model.coefficient(ci) = *model.coefficient(ci) + velocity[ci];
        }
        for ci in 0..model.num_coefficients() {
            let delta = -learning_rate *
                        gradient(cost, prediction, truth, model.gradient(ci, features));
            *model.coefficient(ci) = *model.coefficient(ci) + delta;
            velocity[ci] = self.inertia * velocity[ci] + delta;
        }
        *num_events += 1;
    }
}

/// Adagard learning algorithm
///
/// Adagard divides the learning rate through the square root of the square sum of gradients for
/// each coefficient. In effect the learning rate is smaller for frequent and larger for infrequent
/// features.
/// See [this paper](http://jmlr.org/papers/v12/duchi11a.html) for more information.
pub struct Adagard {
    /// The larger this parameter is, the more the coefficients will change with each iteration
    pub learning_rate: f64,
    /// Small smoothing term, to avoid division by zero in first iteration
    pub epsilon: f64,
}

impl<M> Teacher<M> for Adagard
    where M: Model<Target = f64>
{
    type Training = Vec<f64>;

    fn new_training(&self, model: &M) -> Vec<f64> {

        let mut squared_gradients = Vec::with_capacity(model.num_coefficients());
        squared_gradients.resize(model.num_coefficients(), self.epsilon);
        squared_gradients
    }

    fn teach_event<Y, C>(&self,
                         squared_gradients: &mut Vec<f64>,
                         model: &mut M,
                         cost: &C,
                         features: &M::Features,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {

        let prediction = model.predict(features);
        for ci in 0..model.num_coefficients() {
            let gradient = gradient(cost, prediction, truth, model.gradient(ci, features));
            let delta = -self.learning_rate * gradient / squared_gradients[ci].sqrt();
            *model.coefficient(ci) += delta;
            squared_gradients[ci] += gradient.powi(2);
        }
    }
}