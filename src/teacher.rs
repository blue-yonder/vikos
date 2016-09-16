use ::training;
use Teacher;
use Model;
use std::marker::PhantomData;

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
    type Training = training::GradientDescent<M>;

    fn new_training(&self, _: &M) -> training::GradientDescent<M> {
        training::GradientDescent{ learning_rate : self.learning_rate, model_type : PhantomData{}}
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
    type Training = training::GradientDescentAl<M>;

    fn new_training(&self, _: &M) -> training::GradientDescentAl<M> {
        training::GradientDescentAl::<M> {
            l0: self.l0,
            t: self.t,
            learned_events: 0.0,
            model_type: PhantomData{}
        }
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
    type Training = training::Momentum<M>;

    fn new_training(&self, model: &M) -> training::Momentum<M> {

        let mut velocity = Vec::with_capacity(model.num_coefficents());
        velocity.resize(model.num_coefficents(), 0.0);

        training::Momentum {
            l0: self.l0,
            t: self.t,
            inertia: self.inertia,
            learned_events: 0.0,
            velocity: velocity,
            model_type: PhantomData{}
        }
    }
}

/// Nesterov accelerated gradient descent
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
    type Training = training::Nesterov<M>;

    fn new_training(&self, model: &M) -> training::Nesterov<M> {

        let mut velocity = Vec::with_capacity(model.num_coefficents());
        velocity.resize(model.num_coefficents(), 0.0);

        training::Nesterov {
            l0: self.l0,
            t: self.t,
            inertia: self.inertia,
            learned_events: 0.0,
            velocity: velocity,
            model_type: PhantomData{}
        }
    }
}
