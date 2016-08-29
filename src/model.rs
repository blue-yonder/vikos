use Model;

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

    fn num_coefficents(&self) -> u32{
        1
    }

    fn gradient(&self, coefficent : u32, _ : &()) -> f64{
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