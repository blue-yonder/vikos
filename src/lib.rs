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

    /// Trains a model
    pub fn stochastic_gradient_descent<M, H>(start : M, history : H, learning_rate : M::Target) -> M
        where M : Model,
        H : Iterator<Item=(M::Input, M::Target)>
    {
        let two = M::Target::one() + M::Target::one();
        let mut last = start;
        let mut next = last.clone();
        
        for event in history{
            let prediction = next.predict(&event.0);
            let error = prediction - event.1;

            for ci in 0..next.num_coefficents(){
                *next.coefficent(ci) = *last.coefficent(ci) - learning_rate * two * error * last.gradient(ci, &event.0);
            }

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
        use thea::Model;

        let history = [((), 3f64), ((), 4.0), ((), 5.0)];

        let mut model = Constant{c : 0.0};

        let learning_rate = 0.05;

        for &event in history.iter().cycle().take(60){
            let prediction = model.predict(&event.0);
            let error = prediction - event.1;

            for ci in 0..model.num_coefficents(){
                * model.coefficent(ci) -= learning_rate * 2. * error * model.gradient(ci, &event.0);
            }

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

        let model = stochastic_gradient_descent(start, history.iter().cycle().take(20).cloned(), 0.2); 

        assert!(model.m < 1.1);
        assert!(model.m > 0.9);
        assert!(model.c < 3.1);
        assert!(model.c > 2.9);
    }

    #[test]
    fn linear_stochastic_gradient_descent_iter() {

        use thea::model::Linear;
        use thea::Model;

        let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

        let mut model = Linear{m : 0.0, c : 0.0};

        let learning_rate = 0.2;

        // for &event in history.iter().cycle().take(20){
        //     let prediction = model.predict(&event.0);
        //     let error = prediction - event.1;

        //     for ci in 0..model.num_coefficents(){
        //         * model.coefficent(ci) -= learning_rate * 2. * error * model.gradient(ci, &event.0);
        //     }

        //     println!("model: {:?}", model);
        // }

        let model_it = history.iter().cycle().take(20).scan(& mut model, |m, &(features, truth)|{
            let prediction = m.predict(&features);
            let error = prediction - truth;

            for ci in 0..m.num_coefficents(){
                * m.coefficent(ci) -= learning_rate * 2. * error * m.gradient(ci, &features);
            }

            Some(m.clone())
        });

        for model in model_it{
            println!("model: {:?}", model);
        }

        // assert!(model.m < 1.1);
        // assert!(model.m > 0.9);
        // assert!(model.c < 3.1);
        // assert!(model.c > 2.9);
    }
}
