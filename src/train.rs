use Model;
use Cost;
use Teacher;

use num::Float;

/// Gradient Descent
pub struct GradientDescent<F : Float>{

    /// Defines how fast the coefficents of the trained `Model` will change
    pub learning_rate: F
}

impl<F,M> Teacher<M> for GradientDescent<F>
    where F : Float,
    M : Model<Target=F> {

    fn teach_event<C>(&self, cost : &C, model : &mut M, features : &M::Input, truth : M::Target)
        where C : Cost<Error=M::Target>
    {
        let prediction = model.predict(&features);

        for ci in 0..model.num_coefficents(){
            *model.coefficent(ci) = *model.coefficent(ci) - self.learning_rate * cost.gradient(prediction, truth, model.gradient(ci, features));
        }
    }
}