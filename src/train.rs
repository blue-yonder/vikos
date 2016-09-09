use Model;
use Cost;
use Teacher;
use Training;

use num::Float;

/// SGD with constant learning rate and no momentum 
pub struct GradientDescentTraining<M : Model>{

    /// Defines how fast the coefficents of the trained `Model` will change
    pub learning_rate: M::Target
}

/// Gradient Descent
///
/// Simplest possible implementation of gradient descent with fixed learning rate
pub struct GradientDescent<F : Float>{

    /// Defines how fast the coefficents of the trained `Model` will change
    pub learning_rate: F
}

impl<M> Training for GradientDescentTraining<M> where M : Model{

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

impl<F,M> Teacher<M> for GradientDescent<F>
    where F : Float,
    M : Model<Target=F>
{
    type Training = GradientDescentTraining<M>;
    
    fn new_training(&self) -> GradientDescentTraining<M>{
        GradientDescentTraining::<M>{ learning_rate: self.learning_rate }
    }

    // fn teach_event<C>(&self, cost : &C, model : &mut M, features : &M::Input, truth : M::Target)
    //     where C : Cost<Error=M::Target>
    // {
    //     let prediction = model.predict(&features);

    //     for ci in 0..model.num_coefficents(){
    //         *model.coefficent(ci) = *model.coefficent(ci) - self.learning_rate * cost.gradient(prediction, truth, model.gradient(ci, features));
    //     }
    // }
}

// /// Gradient Descent with annealing learning rate
// ///
// /// For the i-th event the learning rate is `l = l0 * (1 + i/t)`
// pub struct GradientDescentAl<F : Float>{
//     /// Start learning rate
//     pub l0 : F,
//     /// Smaller t will decrease the learning rate faster
//     ///
//     /// After t events the start learning rate will be a half `l0`,
//     /// after two t events the learning rate will be one third `l0`
//     /// and so on.
//     pub t : F
// }

// pub struct GradientDescentAlTraining<F>{
//     /// Start learning rate
//     pub l0 : F,
//     /// Smaller t will decrease the learning rate faster
//     ///
//     /// After t events the start learning rate will be a half `l0`,
//     /// after two t events the learning rate will be one third `l0`
//     /// and so on.
//     pub t : F,

//     /// number of learned events
//     pub learned_events : usize
// }

// impl<F,M> Teacher<M> for GradientDescentAl<F>
//     where F : Float,
//     M : Model<Target=F>
// {
//     type Training = GradientDescentAlTraining<F>;

//     fn new_training(&self) -> GradientDescentAlTraining<F>{
//         GradientDescentAlTraining{ l0: self.l0, t: self.l0, learned_events: 0 }
//     }   

//     fn teach_event<C>(&self, cost : &C, model : &mut M, features : &M::Input, truth : M::Target)
//         where C : Cost<Error=M::Target>
//     {
//         let prediction = model.predict(&features);

//         for ci in 0..model.num_coefficents(){
//             *model.coefficent(ci) = *model.coefficent(ci) - self.l0 * cost.gradient(prediction, truth, model.gradient(ci, features));
//         }
//     }
// }