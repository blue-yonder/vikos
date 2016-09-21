//! Learning algorithms implementing `Teacher` trait

use ::training;
use Teacher;
use Model;
use Cost;

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

    fn teach_event<Y, C>(&self,
                         _training: &mut (),
                         model: &mut M,
                         cost: &C,
                         features: &M::Input,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        let prediction = model.predict(features);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) -
                                    self.learning_rate *
                                    cost.gradient(prediction, truth, model.gradient(ci, features));
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

    fn teach_event<Y, C>(&self,
                         num_events: &mut usize,
                         model: &mut M,
                         cost: &C,
                         features: &M::Input,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        let prediction = model.predict(features);
        let learning_rate = training::annealed_learning_rate(*num_events, self.l0, self.t);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) -
                                    learning_rate *
                                    cost.gradient(prediction, truth, model.gradient(ci, features));
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

    fn teach_event<Y, C>(&self,
                         training: &mut (usize, Vec<f64>),
                         model: &mut M,
                         cost: &C,
                         features: &M::Input,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
        // let (ref mut num_events, ref mut velocity) = *training; ok?
        let mut num_events = &mut training.0;
        let mut velocity = &mut training.1;
        let prediction = model.predict(features);
        let learning_rate = training::annealed_learning_rate(*num_events, self.l0, self.t);

        for ci in 0..model.num_coefficents() {
            velocity[ci] = self.inertia * velocity[ci] -
                           learning_rate *
                           cost.gradient(prediction, truth, model.gradient(ci, features));
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

    fn teach_event<Y, C>(&self,
                         training: &mut (usize, Vec<f64>),
                         model: &mut M,
                         cost: &C,
                         features: &M::Input,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {

        let mut num_events = &mut training.0;
        let mut velocity = &mut training.1;
        let prediction = model.predict(features);
        let learning_rate = training::annealed_learning_rate(*num_events, self.l0, self.t);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) + velocity[ci];
        }
        for ci in 0..model.num_coefficents() {
            let delta = -learning_rate *
                        cost.gradient(prediction, truth, model.gradient(ci, features));
            *model.coefficent(ci) = *model.coefficent(ci) + delta;
            velocity[ci] = self.inertia * velocity[ci] + delta;
        }
        *num_events += 1;
    }
}
