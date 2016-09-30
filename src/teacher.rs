//! Learning algorithms implementing `Teacher` trait

use ::training;
use linear_algebra::Vector;
use Teacher;
use Model;
use Expert;
use Cost;

/// Value of the gradient of the cost function (i.e. the cost function derived by the coefficents
/// at x)
fn cost_gradient<X,Y,C,E>(x: &X, y: &Y, cost: &C, expert: &E) -> E::Gradient
    where C: Cost<Y>,
          E: Expert<X>,
          Y: Clone
{
    // Chain rule of calculus
    expert.gradient(x).mul_scalar(cost.outer_derivative(expert.predict(x), y.clone()))
}

/// Gradient descent
///
/// Simplest possible implementation of gradient descent with fixed learning rate
pub struct GradientDescent {
    /// Defines how fast the coefficents of the trained `Model` will change
    pub learning_rate: f64,
}

impl<M> Teacher<M> for GradientDescent
    where M: Model
{
    type Training = ();

    fn new_training(&self, _: &M) -> () {
        ()
    }

    fn teach_event<X, Y, C>(&self,
                            _training: &mut (),
                            model: &mut M,
                            cost: &C,
                            features: &X,
                            truth: Y)
        where C: Cost<Y>,
              Y: Copy,
              M: Expert<X>
    {
        let gradient = cost_gradient(features, &truth, cost, model);
        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) -
                                    self.learning_rate * gradient.at(ci);
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
    where M: Model
{
    type Training = usize;

    fn new_training(&self, _: &M) -> usize {
        0
    }

    fn teach_event<X, Y, C>(&self,
                            num_events: &mut usize,
                            model: &mut M,
                            cost: &C,
                            features: &X,
                            truth: Y)
        where C: Cost<Y>,
              Y: Copy,
              M: Expert<X>
    {
        let learning_rate = training::annealed_learning_rate(*num_events, self.l0, self.t);
        let gradient = cost_gradient(features, &truth, cost, model);
        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) -
                                    learning_rate * gradient.at(ci);
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
    where M: Model
{
    type Training = (usize, Vec<f64>);

    fn new_training(&self, model: &M) -> (usize, Vec<f64>) {

        let mut velocity = Vec::with_capacity(model.num_coefficents());
        velocity.resize(model.num_coefficents(), 0.0);

        (0, velocity)
    }

    fn teach_event<X, Y, C>(&self,
                            training: &mut (usize, Vec<f64>),
                            model: &mut M,
                            cost: &C,
                            features: &X,
                            truth: Y)
        where C: Cost<Y>,
              Y: Copy,
              M: Expert<X>
    {
        let (ref mut num_events, ref mut velocity) = *training;
        let learning_rate = training::annealed_learning_rate(*num_events, self.l0, self.t);
        let gradient = cost_gradient(features, &truth, cost, model);

        for ci in 0..model.num_coefficents() {
            velocity[ci] = self.inertia * velocity[ci] -
                           learning_rate * gradient.at(ci);

            *model.coefficent(ci) = *model.coefficent(ci) + velocity[ci];
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
    where M: Model
{
    type Training = (usize, Vec<f64>);

    fn new_training(&self, model: &M) -> (usize, Vec<f64>) {

        let mut velocity = Vec::with_capacity(model.num_coefficents());
        velocity.resize(model.num_coefficents(), 0.0);

        (0, velocity)
    }

    fn teach_event<X, Y, C>(&self,
                            training: &mut (usize, Vec<f64>),
                            model: &mut M,
                            cost: &C,
                            features: &X,
                            truth: Y)
        where C: Cost<Y>,
              Y: Copy,
              M: Expert<X>
    {
        let mut num_events = &mut training.0;
        let mut velocity = &mut training.1;
        let learning_rate = training::annealed_learning_rate(*num_events, self.l0, self.t);
        let gradient = cost_gradient(features, &truth, cost, model);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) + velocity[ci];
            let delta = -learning_rate * gradient.at(ci);

            *model.coefficent(ci) = *model.coefficent(ci) + delta;
            velocity[ci] = self.inertia * velocity[ci] + delta;
        }
        *num_events += 1;
    }
}

/// Adagard learning algorithm
///
/// Adagard divides the learning rate through the square root of the square sum of gradients for
/// each coefficent. In effect the learning rate is smaller for frequent and larger for infrequent
/// features.
/// See [this paper](http://jmlr.org/papers/v12/duchi11a.html) for more information.
pub struct Adagard{
    /// The larger this parameter is, the more the coefficents will change with each iteration
    pub learning_rate : f64,
    /// Small smoothing term, to avoid division by zero in first iteration
    pub epsilon : f64
}

impl<M> Teacher<M> for Adagard
    where M: Model
{
    type Training = Vec<f64>;

    fn new_training(&self, model: &M) -> Vec<f64> {

        let mut squared_gradients = Vec::with_capacity(model.num_coefficents());
        squared_gradients.resize(model.num_coefficents(), self.epsilon);
        squared_gradients
    }

    fn teach_event<X, Y, C>(&self,
                        squared_gradients: &mut Vec<f64>,
                        model: &mut M,
                        cost: &C,
                        features: &X,
                        truth: Y)
    where C: Cost<Y>,
            Y: Copy,
            M: Expert<X>
    {
        let gradient = cost_gradient(features, &truth, cost, model);
        for ci in 0..model.num_coefficents() {
            let delta = -self.learning_rate * gradient.at(ci) / squared_gradients[ci].sqrt();
            *model.coefficent(ci) += delta;
            squared_gradients[ci] += gradient.at(ci).powi(2);
        }
    }
}