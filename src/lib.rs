extern crate num;

mod thea{

    use std::iter::Iterator;
    use num::Num;
    use num::One;

    /// A Model is defines how to predict a target from an input
    ///
    /// A model usually depends on several coefficents whose values
    /// are derived using a training algorithm 
    pub trait Model : Clone{
        /// Input features
        type Input;
        /// Target type
        type Target : Num + One + Copy;
 
        /// Predicts a target for the inputs based on the internal coefficents
        fn predict(&self, &Self::Input) -> Self::Target;

        /// The number of internal coefficents this model depends on
        fn num_coefficents(&self) -> u32;

        /// Value predict derived by the n-th `coefficent` at `input`
        fn gradient(&self, coefficent : u32, input : &Self::Input) -> Self::Target;

        /// Mutable reference to the n-th `coefficent`
        fn coefficent(& mut self, coefficent : u32) -> & mut Self::Target;
    }

    /// Changes all coefficents of model based on their derivation of the cost function at features
    ///
    /// Can be used to implement stochastic or batch gradient descent
    pub fn gradient_descent_step<M : Model>(model : &mut M, features : &M::Input, truth : M::Target, learning_rate : M::Target)
    {
        let two = M::Target::one() + M::Target::one();
        let prediction = model.predict(&features);
        let error = prediction - truth;

        for ci in 0..model.num_coefficents(){
            *model.coefficent(ci) = *model.coefficent(ci) - learning_rate * two * error * model.gradient(ci, features);
        }
    }

    /// Trains a model
    pub fn stochastic_gradient_descent<M, H>(start : M, history : H, learning_rate : M::Target) -> M
        where M : Model,
        H : Iterator<Item=(M::Input, M::Target)>
    {
        let mut last = start;
        let mut next = last.clone();
        
        for (features, truth) in history{

            gradient_descent_step(& mut next, &features, truth, learning_rate);
            last = next.clone();
        }

        next
    }

    pub mod model{

        use thea::Model;

        /// Constant Model
        ///
        /// This model predicts a number. The cost function used during training decides
        /// wether this number is a mean, median or something else.
        #[derive(Debug, Clone)]
        pub struct Constant{
            /// Any prediction made by this model will have the value of `c`
            pub c : f64
        }

        impl Model for Constant{
            type Input = ();
            type Target = f64;

            fn predict(&self, input: &()) -> f64{
                self.c
            }

            fn num_coefficents(&self) -> u32{
                1
            }

            fn gradient(&self, coefficent : u32, input : &()) -> f64{
                match coefficent{
                    0 => 1.0,
                    _ => panic!("coefficent index out of range")
                }
            }

            fn coefficent(& mut self, coefficent : u32) -> & mut f64{
                match coefficent{
                    0 => & mut self.c,
                    _ => panic!("coefficent index out of range")
                }
            }
        }

        /// Linear model
        ///
        /// Models the target as `y = m * x + c`
        #[derive(Debug, Clone)]
        pub struct Linear{
            /// Slope
            pub m : f64,
            /// Offset
            pub c : f64
        }

        impl Model for Linear{

            type Input = f64;
            type Target = f64;

            fn predict(&self, input : &f64) -> f64{
                self.m * *input + self.c
            }

            fn num_coefficents(&self) -> u32{
                2
            }

            fn gradient(&self, coefficent : u32, input : &f64) -> f64{
                match coefficent{
                    0 => *input,
                    1 => 1.0,
                    _ => panic!("coefficent index out of range")
                }
            }

            fn coefficent(& mut self, coefficent : u32) -> & mut f64{
                match coefficent{
                    0 => & mut self.m,
                    1 => & mut self.c,
                    _ => panic!("coefficent index out of range")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn estimate_mean() {

        use thea::model::Constant;
        use thea::gradient_descent_step
;

        let history = [((), 3f64), ((), 4.0), ((), 5.0)];

        let mut model = Constant{c : 0.0};

        let learning_rate = 0.05;

        for &(features, truth) in history.iter().cycle().take(60){

            gradient_descent_step
    (& mut model, &features, truth, learning_rate);
            println!("model: {:?}", model);
        }

        assert!(model.c < 4.1);
        assert!(model.c > 3.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent() {

        use thea::model::Linear;
        use thea::stochastic_gradient_descent;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let start = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        let model = stochastic_gradient_descent(start, history.iter().cycle().take(20).cloned(), learning_rate); 

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent_iter() {

        use thea::model::Linear;
        use thea::gradient_descent_step
;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let mut model = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        for &(features, truth) in history.iter().cycle().take(20){

            gradient_descent_step
    (& mut model, &features, truth, learning_rate);

            println!("model: {:?}", model);
        }

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }
}
