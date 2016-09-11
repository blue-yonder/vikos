use ::training;
use Teacher;
use Model;
use num::Float;

/// Gradient Descent
///
/// Simplest possible implementation of gradient descent with fixed learning rate
pub struct GradientDescent<F : Float>{
    /// Defines how fast the coefficents of the trained `Model` will change
    pub learning_rate: F
}

impl<F,M> Teacher<M> for GradientDescent<F>
    where F : Float,
    M : Model<Target=F>
{
    type Training = training::GradientDescent<M>;

    fn new_training(&self, _ : &M) -> training::GradientDescent<M>{
        training::GradientDescent::<M>{ learning_rate: self.learning_rate }
    }
}

/// Gradient Descent with annealing learning rate
///
/// For the i-th event the learning rate is `l = l0 * (1 + i/t)`
pub struct GradientDescentAl<F : Float>{
    /// Start learning rate
    pub l0 : F,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`
    /// and so on.
    pub t : F
}

impl<F,M> Teacher<M> for GradientDescentAl<F>
    where F : Float,
    M : Model<Target=F>
{
    type Training = training::GradientDescentAl<M>;

    fn new_training(&self, _ : &M) -> training::GradientDescentAl<M>{
        training::GradientDescentAl::<M>{ l0: self.l0, t: self.t, learned_events: M::Target::zero() }
    }
}

/// Gradient Descent with annealing learning rate and momentum
///
/// For the i-th event the learning rate is `l = l0 * (1 + i/t)`
pub struct Momentum<F : Float>{
    /// Start learning rate
    pub l0 : F,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`
    /// and so on.
    pub t : F,
    /// Too simulate friction select a value smaller than 1 (recommended)
    pub inertia : F,
}

impl<F,M> Teacher<M> for Momentum<F>
    where F : Float,
    M : Model<Target=F>
{
    type Training = training::Momentum<M>;

    fn new_training(&self, model : &M) -> training::Momentum<M>{

        let mut velocity = Vec::with_capacity(model.num_coefficents());
        velocity.resize(model.num_coefficents(), M::Target::zero());

        training::Momentum::<M>{
            l0: self.l0, t: self.t,
            inertia: self.inertia,
            learned_events: M::Target::zero(),
            velocity : velocity
        }
    }
}

/// Nesterov accelerated Gradient Descent
pub struct Nesterov<F : Float>{
    /// Start learning rate
    pub l0 : F,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`
    /// and so on.
    pub t : F,
    /// Too simulate friction select a value smaller than 1 (recommended)
    pub inertia : F,
}

impl<F,M> Teacher<M> for Nesterov<F>
    where F : Float,
    M : Model<Target=F>
{
    type Training = training::Nesterov<M>;

    fn new_training(&self, model : &M) -> training::Nesterov<M>{

        let mut velocity = Vec::with_capacity(model.num_coefficents());
        velocity.resize(model.num_coefficents(), M::Target::zero());

        training::Nesterov::<M>{
            l0: self.l0, t: self.t,
            inertia: self.inertia,
            learned_events: M::Target::zero(),
            velocity : velocity
        }
    }
}