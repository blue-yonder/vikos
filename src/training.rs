use Model;
use Cost;
use Training;

use num::One;

/// SGD with constant learning rate and no momentum 
pub struct GradientDescent<M : Model>{

    /// Defines how fast the coefficents of the trained `Model` will change
    pub learning_rate: M::Target
}

impl<M> Training for GradientDescent<M> where M : Model{

    type Model = M;

    fn teach_event<C>(&mut self, cost : &C, model : &mut M, features : &M::Input, truth : M::Target)
        where C : Cost<Error=M::Target>
    {
        let prediction = model.predict(&features);

        for ci in 0..model.num_coefficents(){
            *model.coefficent(ci) = *model.coefficent(ci) - self.learning_rate * cost.gradient(prediction, truth, model.gradient(ci, features));
        }
    }
}

/// Trains a Model with an annealing learning rate
pub struct GradientDescentAl<M : Model>{
    /// Start learning rate
    pub l0 : M::Target,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`
    /// and so on.
    pub t : M::Target,

    /// number of learned events
    pub learned_events : M::Target
}

impl<M : Model> GradientDescentAl<M>{

    /// Returns current learning rate
    ///
    /// While this could be calculated directly by `teach_event`
    /// its useful for debugging or monitoring purposes to have a
    /// look at the current learning rate
    pub fn learning_rate(&self) -> M::Target{
        self.l0 / (M::Target::one() + self.learned_events / self.t)
    } 
}

impl<M> Training for GradientDescentAl<M>
    where M : Model
{
    type Model = M;

    fn teach_event<C>(&mut self, cost : &C, model : &mut M, features : &M::Input, truth : M::Target)
        where C : Cost<Error=M::Target>
    {
        let prediction = model.predict(&features);

        for ci in 0..model.num_coefficents(){
            *model.coefficent(ci) = *model.coefficent(ci) - self.learning_rate() * cost.gradient(prediction, truth, model.gradient(ci, features));
        }

        self.learned_events = self.learned_events + M::Target::one();
    }
}

/// SGD training with adaptive learning rate and momentum term
pub struct Momentum<M : Model>{

    /// Start learning rate
    pub l0 : M::Target,

    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`
    /// and so on.
    pub t : M::Target,

    /// Too simulate friction select a value smaller than 1 (recommended)
    pub inertia : M::Target,

    /// Number of learned events
    pub learned_events : M::Target,

    /// Current velocity of coefficents (in delta per iteration);
    pub velocity : Vec<M::Target>
}

impl<M : Model> Momentum<M>{

    /// Returns current learning rate
    ///
    /// While this could be calculated directly by `teach_event`
    /// its useful for debugging or monitoring purposes to have a
    /// look at the current learning rate
    pub fn learning_rate(&self) -> M::Target{
        self.l0 / (M::Target::one() + self.learned_events / self.t)
    } 
}

impl<M> Training for Momentum<M> where M : Model{

    type Model = M;

    fn teach_event<C>(&mut self, cost : &C, model : &mut M, features : &M::Input, truth : M::Target)
        where C : Cost<Error=M::Target>
    {
        let prediction = model.predict(&features);

        for ci in 0..model.num_coefficents(){
            self.velocity[ci] =
                self.inertia * self.velocity[ci]
                - self.learning_rate() * cost.gradient(prediction, truth, model.gradient(ci, features));
            *model.coefficent(ci) = *model.coefficent(ci) + self.velocity[ci];
        }

        self.learned_events = self.learned_events + M::Target::one();
    }
}