//! A machine learning library for supervised regression trainings
//!
//! This library wants to enable its user to write training algorithms
//! independent of the model trained or the cost function tried to
//! minimize.
//! Consequently its two main traits are currently `Model` and `Cost`.
//! The two submodules `model` and `cost` provide ready to use
//! implementations of said traits.

extern crate num;

use std::iter::Iterator;
use num::Num;

/// A Model is defines how to predict a target from an input
///
/// A model usually depends on several coefficents whose values
/// are derived using a training algorithm 
pub trait Model : Clone{
    /// Input features
    type Input;
    /// Target type
    type Target : Num + Copy;

    /// Predicts a target for the inputs based on the internal coefficents
    fn predict(&self, &Self::Input) -> Self::Target;

    /// The number of internal coefficents this model depends on
    fn num_coefficents(&self) -> u32;

    /// Value predict derived by the n-th `coefficent` at `input`
    fn gradient(&self, coefficent : u32, input : &Self::Input) -> Self::Target;

    /// Mutable reference to the n-th `coefficent`
    fn coefficent(& mut self, coefficent : u32) -> & mut Self::Target;
}

/// Cost functions those value is supposed be minimized by the training algorithm
pub trait Cost{

    type Error : Num + Copy;

    /// Value of the cost function derived by the n-th coefficent at x expressed in E(x) and dY(x)/dx
    ///
    /// This method is called by SGD based training algorithm in order to
    /// determine the delta of the coefficents
    fn gradient(&self, error : Self::Error, gradient_error_by_coefficent : Self::Error) -> Self::Error;
}

/// Changes all coefficents of model based on their derivation of the cost function at features
///
/// Can be used to implement stochastic or batch gradient descent
pub fn gradient_descent_step<C : Cost, M : Model<Target=C::Error>>(cost : &C, model : &mut M, features : &M::Input, truth : M::Target, learning_rate : M::Target)
{
    let prediction = model.predict(&features);
    let error = prediction - truth;

    for ci in 0..model.num_coefficents(){
        *model.coefficent(ci) = *model.coefficent(ci) - learning_rate * cost.gradient(error, model.gradient(ci, features));
    }
}

/// Trains a model
pub fn stochastic_gradient_descent<C, M, H>(cost : &C, start : M, history : H, learning_rate : M::Target) -> M
    where C : Cost,
    M : Model<Target=C::Error>,
    H : Iterator<Item=(M::Input, M::Target)>
{

    let mut next = start.clone();        
    for (features, truth) in history{

        gradient_descent_step(cost, & mut next, &features, truth, learning_rate);
    }

    next
}

pub mod model;
pub mod cost;
pub mod linear_algebra;

#[cfg(test)]
mod tests {

    #[test]
    fn estimate_median() {

        use model::Constant;
        use cost::LeastAbsoluteDeviation;
        use gradient_descent_step;

        let features = ();
        let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //median is seven

        let cost = LeastAbsoluteDeviation{};
        let mut model = Constant{c : 0.0};

        let learning_rate_start = 0.4;
        let learning_rate_stop = 0.001;
        let num_steps = 200;
        let learning_rate_gradient = (learning_rate_start - learning_rate_stop) / (num_steps as f64);

        for (count_step, &truth) in history.iter().cycle().take(num_steps).enumerate(){

            let adapted_learning_rate = learning_rate_stop + learning_rate_gradient * (num_steps - count_step) as f64;
            gradient_descent_step(&cost, & mut model, &features, truth, adapted_learning_rate);
            println!("model: {:?}, learning_rate: {:?}", model, adapted_learning_rate);
        }

        assert!(model.c < 7.1);
        assert!(model.c > 6.9);
    }

    #[test]
    fn estimate_mean() {

        use model::Constant;
        use cost::LeastSquares;
        use gradient_descent_step;

        let features = ();
        let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9

        let cost = LeastSquares{};
        let mut model = Constant{c : 0.0};

        let learning_rate_start = 0.4;
        let learning_rate_stop = 0.001;
        let num_steps = 60;
        let learning_rate_gradient = (learning_rate_start - learning_rate_stop) / (num_steps as f64);

        for (count_step, &truth) in history.iter().cycle().take(num_steps).enumerate(){

            let adapted_learning_rate = learning_rate_stop + learning_rate_gradient * (num_steps - count_step) as f64;
            gradient_descent_step(&cost, & mut model, &features, truth, adapted_learning_rate);
            println!("model: {:?}, learning_rate: {:?}", model, adapted_learning_rate);
        }

        assert!(model.c < 9.1);
        assert!(model.c > 8.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent() {

        use cost::LeastSquares;
        use model::Linear;
        use stochastic_gradient_descent;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let start = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        let cost = LeastSquares{};
        let model = stochastic_gradient_descent(&cost, start, history.iter().cycle().take(20).cloned(), learning_rate); 

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent_iter() {

        use model::Linear;
        use gradient_descent_step;
        use cost::LeastSquares;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let cost = LeastSquares{};
        let mut model = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        for &(features, truth) in history.iter().cycle().take(20){

            gradient_descent_step(&cost, & mut model, &features, truth, learning_rate);
            println!("model: {:?}", model);
        }

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_sgd_2d()
    {
        use cost::LeastSquares;
        use model::Linear;
        use stochastic_gradient_descent;

        let history = [([0.0, 7.0], 17.0), ([1.0, 2.0], 8.0), ([2.0, -2.0], 1.0)];

        let start = Linear{m : [0.0, 0.0], c : 0.0};

        let learning_rate = 0.05;

        let cost = LeastSquares{};
        let model = stochastic_gradient_descent(&cost, start, history.iter().cycle().take(20000).cloned(), learning_rate); 

        println!("{:?}", model);

        assert!(model.m[0] < 1.1);
        assert!(model.m[0] > 0.9);
        assert!(model.m[1] < 2.1);
        assert!(model.m[1] > 1.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }
}
