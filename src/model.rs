use Model;
use linear_algebra::Vector;
use num::{Float, One};
use std::marker::PhantomData;

/// Models the target as `c`
///
/// This model predicts a number. The cost function used during training decides
/// wether this number is a mean, median or something else.
///
/// # Examples
///
/// Estimate mean
///
/// ```
/// use vikos::model::Constant;
/// use vikos::cost::LeastSquares;
/// use vikos::train::GradientDescentTraining;
/// use vikos::Training;
///
/// let features = ();
/// let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9
///
/// let cost = LeastSquares{};
/// let mut model = Constant::new(0.0);
///
/// let learning_rate_start = 0.3;
/// let decay = 4;
///
/// for (count_step, &truth) in history.iter().cycle().take(100).enumerate(){
///
/// let mut training = GradientDescentTraining{ learning_rate: learning_rate_start / ( 1.0 + count_step as f64 /decay as f64) as f64 };
///     training.teach_event(&cost, &mut model, &features, truth);
/// }
/// ```
#[derive(Debug)]
pub struct Constant<Input>{
    /// Any prediction made by this model will have the value of `c`
    pub c: f64,
    _phantom: PhantomData<Input>
}

impl<I> Constant<I> {
    /// Creates a new Constant from a `f64`
    pub fn new(c: f64) -> Constant<I>{
        Constant{c: c, _phantom: PhantomData::<I>{}}
    }
}

impl<I> Clone for Constant<I>{
    fn clone(&self) -> Self{
        Constant::new(self.c)
    }
}

impl<I> Model for Constant<I>{
    type Input = I;
    type Target = f64;

    fn predict(&self, _: &I) -> f64{
        self.c
    }

    fn num_coefficents(&self) -> usize{
        1
    }

    fn gradient(&self, coefficent : usize, _ : &I) -> f64{
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

/// Models the target as `y = m * x + c`
#[derive(Debug, Clone)]
pub struct Linear<V : Vector>{
    /// Slope
    pub m : V,
    /// Offset
    pub c : V::Scalar
}

impl<V : Vector> Model for Linear<V> where V::Scalar : Float{

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

/// Models target as `y = 1/1+e^(m * x + c)`
#[derive(Clone)]
pub struct Logicstic<V: Vector>{
    /// Linear term of `Logistic` model
    pub linear : Linear<V>
}

impl<V : Vector> Model for Logicstic<V> where V::Scalar : Float{
    type Input = V;
    type Target = V::Scalar;

    fn predict(&self, input : &V) -> V::Scalar{
        V::Scalar::one() / (V::Scalar::one() + self.linear.predict(input).exp())
    }

    fn num_coefficents(&self) -> usize{
        self.linear.num_coefficents()
    }

    fn gradient(&self, coefficent : usize, input : &V) -> V::Scalar{
        let p = self.predict(input);
        - p * (V::Scalar::one() - p) * self.linear.gradient(coefficent, input)
    }

    fn coefficent(& mut self, coefficent : usize) -> & mut V::Scalar{
        self.linear.coefficent(coefficent)
    }
}