use Model;
use Expert;
use linear_algebra::Vector;

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
pub struct Constant {
    /// Any prediction made by this model will have the value of `c`
    pub c: f64
}

impl Constant {
    /// Creates a new Constant from a `f64`
    pub fn new(c: f64) -> Constant {
        Constant {
            c: c
        }
    }
}

impl Clone for Constant {
    fn clone(&self) -> Self {
        Constant::new(self.c)
    }
}

impl Model for Constant{

    fn num_coefficents(&self) -> usize {
        1
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut f64 {
        match coefficent {
            0 => &mut self.c,
            _ => panic!("coefficent index out of range"),
        }
    }

}

impl<I> Expert<I> for Constant {

    fn predict(&self, _: &I) -> f64 {
        self.c
    }

    fn gradient(&self, coefficent: usize, _: &I) -> f64 {
        match coefficent {
            0 => 1.0,
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
    fn num_coefficents(&self) -> usize {
        self.m.dimension() + 1
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut V::Scalar {
        if coefficent == self.m.dimension() {
            &mut self.c
        } else {
            self.m.mut_at(coefficent)
        }
    }
}

impl<V> Expert<V> for Linear<V>
    where V: Vector<Scalar = f64>
{
    fn predict(&self, input: &V) -> V::Scalar {
        self.m.dot(input) + self.c
    }

    fn gradient(&self, coefficent: usize, input: &V) -> V::Scalar {

        use num::One;

        if coefficent == self.m.dimension() {
            V::Scalar::one() //c
        } else {
            input.at(coefficent)
        }
    }
}

/// Models target as `y = 1/(1+e^(m * x + c))`
#[derive(Debug, Clone, Default, RustcDecodable, RustcEncodable)]
pub struct Logistic<V: Vector>(Linear<V>);

impl<V> Model for Logistic<V>
    where V: Vector<Scalar = f64>
{
    fn num_coefficents(&self) -> usize {
        self.0.num_coefficents()
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut f64 {
        self.0.coefficent(coefficent)
    }
}

impl<V> Expert<V> for Logistic<V>
    where V: Vector<Scalar = f64>
{
    fn predict(&self, input: &V) -> f64 {
        1.0 / (1.0 + self.0.predict(input).exp())
    }

    fn gradient(&self, coefficent: usize, input: &V) -> f64 {
        let p = self.predict(input);
        -p * (1.0 - p) * self.0.gradient(coefficent, input)
    }
}

/// Models the target as `y = g(m*x + c)`
///
/// # Example
///
/// Logistic regression implemented using a generalized linear model. This is just for
/// demonstration purposes. For this usecase you would usally use `Logistic`.
///
/// ```
/// # use vikos::{model, teacher, cost, learn_history};
/// # let history = [(0.0, true)];
/// let mut model = model::GeneralizedLinearModel::new(|x| 1.0 / (1.0 + x.exp()),
///                                                    |x| -x.exp() / (1.0 + x.exp()).powi(2) );
/// let teacher = teacher::GradientDescent { learning_rate: 0.3 };
/// let cost = cost::MaxLikelihood {};
///
/// learn_history(&teacher,
///               &cost,
///               &mut model,
///               history.iter().cloned());
/// ```
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
    where V: Vector,
          G: Fn(f64) -> f64,
          Dg: Fn(f64) -> f64
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
    where F: Fn(f64) -> f64,
          Df: Fn(f64) -> f64,
          V: Vector<Scalar = f64>
{
    fn num_coefficents(&self) -> usize {
        self.linear.num_coefficents()
    }

    fn coefficent(&mut self, coefficent: usize) -> &mut f64 {
        self.linear.coefficent(coefficent)
    }
}

impl<V, F, Df> Expert<V> for GeneralizedLinearModel<V, F, Df>
    where F: Fn(f64) -> f64,
          Df: Fn(f64) -> f64,
          V: Vector<Scalar = f64>
{
    fn predict(&self, input: &V) -> f64 {
        let f = &self.g;
        f(self.linear.predict(&input))
    }

    fn gradient(&self, coefficent: usize, input: &V) -> f64 {
        let f = &self.g_derivate;
        f(self.linear.predict(&input)) * self.linear.gradient(coefficent, input)
    }
}
