use Model;
use linear_algebra::Vector;
use std::marker::PhantomData;

/// Models the target as a constant `c`
///
/// This model predicts a number. The cost function used during training decides
/// whether this number is a mean, median, or something else.
///
/// # Examples
///
/// Estimate mean
///
/// ```
/// use vikos::model::Constant;
/// use vikos::cost::LeastSquares;
/// use vikos::teacher::GradientDescentAl;
/// use vikos::learn_history;
///
/// let features = ();
/// let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9
///
/// let cost = LeastSquares{};
/// let mut model = Constant::new(0.0);
///
/// let teacher = GradientDescentAl{ l0 : 0.3, t : 4.0 };
/// learn_history(&teacher, &cost, &mut model, history.iter().cycle().map(|&y|((),y)).take(100));
/// println!("{}", model.c);
/// ```
#[derive(Debug, Default, RustcDecodable, RustcEncodable)]
pub struct Constant<Input> {
    /// Any prediction made by this model will have the value of `c`
    pub c: f64,
    _phantom: PhantomData<Input>,
}

impl<I> Constant<I> {
    /// Creates a new Constant from a `f64`
    pub fn new(c: f64) -> Constant<I> {
        Constant {
            c: c,
            _phantom: PhantomData::<I> {},
        }
    }
}

impl<I> Clone for Constant<I> {
    fn clone(&self) -> Self {
        Constant::new(self.c)
    }
}

impl<I> Model for Constant<I> {
    type Input = I;

    fn predict(&self, _: &I) -> f64 {
        self.c
    }

    fn num_coefficents(&self) -> usize {
        1
    }

    fn gradient(&self, coefficent: usize, _: &I) -> f64 {
        match coefficent {
            0 => 1.0,
            _ => panic!("coefficent index out of range"),
        }
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut f64 {
        match coefficent {
            0 => &mut self.c,
            _ => panic!("coefficent index out of range"),
        }
    }
}

/// Models the target as `y = m * x + c`
#[derive(Debug, Clone, Default, RustcDecodable, RustcEncodable)]
pub struct Linear<V: Vector> {
    /// Slope
    pub m: V,
    /// Offset
    pub c: V::Scalar,
}

impl<V> Model for Linear<V>
    where V: Vector<Scalar = f64>
{
    type Input = V;

    fn predict(&self, input: &V) -> V::Scalar {
        self.m.dot(input) + self.c
    }

    fn num_coefficents(&self) -> usize {
        self.m.dimension() + 1
    }

    fn gradient(&self, coefficent: usize, input: &V) -> V::Scalar {

        use num::One;

        if coefficent == self.m.dimension() {
            V::Scalar::one() //c
        } else {
            input.at(coefficent)
        }
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut V::Scalar {
        if coefficent == self.m.dimension() {
            &mut self.c
        } else {
            self.m.mut_at(coefficent)
        }
    }
}

/// Models target as `y = 1/(1+e^(m * x + c))`
#[derive(Debug, Clone, Default, RustcDecodable, RustcEncodable)]
pub struct Logistic<V: Vector>(Linear<V>);

impl<V> Model for Logistic<V>
    where V: Vector<Scalar = f64>
{
    type Input = V;

    fn predict(&self, input: &V) -> f64 {
        1.0 / (1.0 + self.0.predict(input).exp())
    }

    fn num_coefficents(&self) -> usize {
        self.0.num_coefficents()
    }

    fn gradient(&self, coefficent: usize, input: &V) -> f64 {
        let p = self.predict(input);
        -p * (1.0 - p) * self.0.gradient(coefficent, input)
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut f64 {
        self.0.coefficent(coefficent)
    }
}

/// Models the target as `y = g(m*x + c)`
#[derive(Clone)]
pub struct GeneralizedLinearModel<V: Vector, G, Dg> {
    /// `Linear` term of the generalized linear `Model`
    pub linear: Linear<V>,
    /// Outer function applied to the result of `linear`
    pub g: G,
    /// Derivation of `g`
    pub g_derivate: Dg,
}

impl<V, G, Dg> GeneralizedLinearModel<V, G, Dg>
    where V: Vector
{
    /// Creates new model with the coefficents set to zero
    pub fn new(g: G, g_derivate: Dg) -> GeneralizedLinearModel<V, G, Dg>
        where V: Default
    {
        GeneralizedLinearModel {
            linear: Linear::default(),
            g: g,
            g_derivate: g_derivate,
        }
    }
}

impl<V, F, Df> Model for GeneralizedLinearModel<V, F, Df>
    where F: Fn(f64) -> f64 + Clone,
          Df: Fn(f64) -> f64 + Clone,
          V: Vector<Scalar = f64>
{
    type Input = V;

    fn predict(&self, input: &V) -> f64 {
        let f = &self.g;
        f(self.linear.predict(&input))
    }

    fn num_coefficents(&self) -> usize {
        self.linear.num_coefficents()
    }

    fn gradient(&self, coefficent: usize, input: &V) -> f64 {
        let f = &self.g_derivate;
        f(self.linear.predict(&input)) * self.linear.gradient(coefficent, input)
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut f64 {
        self.linear.coefficent(coefficent)
    }
}
