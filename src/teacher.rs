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
    
    fn new_training(&self) -> training::GradientDescent<M>{
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

    fn new_training(&self) -> training::GradientDescentAl<M>{
        training::GradientDescentAl::<M>{ l0: self.l0, t: self.l0, learned_events: M::Target::zero() }
    }   
}