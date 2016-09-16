use Model;
use Cost;
use Training;
use std::marker::PhantomData;

/// Stochastic gradient descent (SGD) with constant learning rate and no momentum
pub struct GradientDescent<M: Model> {
    /// Defines how fast the coefficients of the trained `Model` will change
    pub learning_rate: f64,

    /// Associated `Model` type
    pub model_type: PhantomData<M>,
}

impl<M> Training for GradientDescent<M>
    where M: Model
{
    type Model = M;

    fn teach_event<C, Truth>(&mut self, cost: &C, model: &mut M, features: &M::Input, truth: Truth)
        where C: Cost<Truth>,
              Truth: Copy
    {
        let prediction = model.predict(features);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) -
                                    self.learning_rate *
                                    cost.gradient(prediction, truth, model.gradient(ci, features));
        }
    }
}

/// Trains a model with an annealing learning rate
pub struct GradientDescentAl<M: Model> {
    /// Start learning rate
    pub l0: f64,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`,
    /// and so on.
    pub t: f64,

    /// number of learned events
    pub learned_events: f64,

    /// associated model type
    pub model_type: PhantomData<M>,
}

impl<M: Model> GradientDescentAl<M> {
    /// Returns current learning rate
    ///
    /// While this could be calculated directly by `teach_event`,
    /// it is useful for debugging or monitoring purposes to have a
    /// look at the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.l0 / (1.0 + self.learned_events / self.t)
    }
}

impl<M> Training for GradientDescentAl<M>
    where M: Model
{
    type Model = M;

    fn teach_event<C, Truth>(&mut self, cost: &C, model: &mut M, features: &M::Input, truth: Truth)
        where C: Cost<Truth>,
              Truth: Copy
    {
        let prediction = model.predict(features);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) -
                                    self.learning_rate() *
                                    cost.gradient(prediction, truth, model.gradient(ci, features));
        }

        self.learned_events = self.learned_events + 1.0;
    }
}

/// Stochastic gradient descent (SGD) training with adaptive learning rate and momentum term
#[derive(Debug)]
pub struct Momentum<M: Model> {
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

    /// Number of learned events
    pub learned_events: f64,

    /// Current velocity of coefficients (in delta per iteration);
    pub velocity: Vec<f64>,

    /// Associated `Model` type
    pub model_type: PhantomData<M>,
}

impl<M: Model> Momentum<M> {
    /// Returns current learning rate
    ///
    /// While this could be calculated directly by `teach_event`,
    /// it is useful for debugging or monitoring purposes to have a
    /// look at the current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.l0 / (1.0 + self.learned_events / self.t)
    }
}

impl<M> Training for Momentum<M>
    where M: Model
{
    type Model = M;

    fn teach_event<C, Truth>(&mut self, cost: &C, model: &mut M, features: &M::Input, truth: Truth)
        where C: Cost<Truth>,
              Truth: Copy
    {
        let prediction = model.predict(features);

        for ci in 0..model.num_coefficents() {
            self.velocity[ci] = self.inertia * self.velocity[ci] -
                                self.learning_rate() *
                                cost.gradient(prediction, truth, model.gradient(ci, features));
            *model.coefficent(ci) = *model.coefficent(ci) + self.velocity[ci];
        }

        self.learned_events = self.learned_events + 1.0;
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
#[derive(Debug)]
pub struct Nesterov<M: Model> {
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

    /// Number of learned events
    pub learned_events: f64,

    /// Current velocity of coefficents (in delta per iteration);
    pub velocity: Vec<f64>,

    /// Associated `Model` type
    pub model_type: PhantomData<M>,
}

impl<M: Model> Nesterov<M> {
    /// Returns current learning rate
    ///
    /// While this could be calculated directly by `teach_event`,
    /// it is useful for debugging or monitoring purposes to have a
    /// look at the current learning rate
    pub fn learning_rate(&self) -> f64 {
        self.l0 / (1.0 + self.learned_events / self.t)
    }
}

impl<M> Training for Nesterov<M>
    where M: Model
{
    type Model = M;

    fn teach_event<C, Truth>(&mut self, cost: &C, model: &mut M, features: &M::Input, truth: Truth)
        where C: Cost<Truth>,
              Truth: Copy
    {
        let prediction = model.predict(features);

        for ci in 0..model.num_coefficents() {
            *model.coefficent(ci) = *model.coefficent(ci) + self.velocity[ci];
        }

        for ci in 0..model.num_coefficents() {
            let delta = -self.learning_rate() *
                        cost.gradient(prediction, truth, model.gradient(ci, features));
            *model.coefficent(ci) = *model.coefficent(ci) + delta;
            self.velocity[ci] = self.inertia * self.velocity[ci] + delta;
        }

        self.learned_events = self.learned_events + 1.0;
    }
}
