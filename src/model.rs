use Model;
use linear_algebra::Vector;

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

    fn predict(&self, _: &()) -> f64{
        self.c
    }

    fn num_coefficents(&self) -> usize{
        1
    }

    fn gradient(&self, coefficent : usize, _ : &()) -> f64{
        match coefficent{
            0 => 1.0,
            _ => panic!("coefficent index out of range")
        }
    }

    fn coefficent(& mut self, coefficent : usize) -> & mut f64{
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
pub struct Linear<V : Vector>{
    /// Slope
    pub m : V,
    /// Offset
    pub c : V::Scalar
}

impl<V : Vector> Model for Linear<V>{

    type Input = V;
    type Target = V::Scalar;

    fn predict(&self, input : &V) -> V::Scalar{
        self.m.dot(input) + self.c
    }

    fn num_coefficents(&self) -> usize{
        self.m.dimension() + 1
    }

    fn gradient(&self, coefficent : usize, input : &V) -> V::Scalar{

        use num::One;

        if coefficent == self.m.dimension(){
            V::Scalar::one() //c
        } else {
            input.at(coefficent)
        }
    }

    fn coefficent(& mut self, coefficent : usize) -> & mut V::Scalar{
        if coefficent == self.m.dimension(){
            & mut self.c
        } else {
            self.m.mut_at(coefficent)
        }
    }
}