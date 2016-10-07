use Model;
use Expert;
use linear_algebra::Vector;

impl Model for f64 {
    fn num_coefficients(&self) -> usize {
        1
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        match coefficient {
            0 => self,
            _ => panic!("coefficient index out of range"),
        }
    }
}

impl<I> Expert<I> for f64 {

    type Gradient = f64;

    fn predict(&self, _: &I) -> f64 {
        *self
    }

    fn gradient(&self, _: &I) -> f64 {
        1.0
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
    fn num_coefficients(&self) -> usize {
        self.m.dimension() + 1
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut V::Scalar {
        if coefficient == 0 {
            &mut self.c
        } else {
            self.m.mut_at(coefficient - 1)
        }
    }
}

impl<V> Expert<V> for Linear<V>
    where V: Vector<Scalar = f64>
{
    type Gradient = (V::Scalar, V);

    fn predict(&self, input: &V) -> V::Scalar {
        self.m.dot(input) + self.c
    }

    fn gradient(&self, input: &V) -> (V::Scalar, V) {
        use num::One;
        (V::Scalar::one(), input.clone())
    }
}

/// Models target as `y = 1/(1+e^(m * x + c))`
#[derive(Debug, Clone, Default, RustcDecodable, RustcEncodable)]
pub struct Logistic<V: Vector>(Linear<V>);

impl<V> Model for Logistic<V>
    where V: Vector<Scalar = f64>
{
    fn num_coefficients(&self) -> usize {
        self.0.num_coefficients()
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        self.0.coefficient(coefficient)
    }
}

impl<V> Expert<V> for Logistic<V>
    where V: Vector<Scalar = f64>
{
    type Gradient = (V::Scalar, V);

    fn predict(&self, input: &V) -> f64 {
        1.0 / (1.0 + self.0.predict(input).exp())
    }

    fn gradient(&self, input: &V) -> (V::Scalar, V) {
        let p = self.predict(input);
        self.0.gradient(input).mul_scalar(-p * (1.0 - p))
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
    /// Creates new model with the coefficients set to zero
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
    fn num_coefficients(&self) -> usize {
        self.linear.num_coefficients()
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        self.linear.coefficient(coefficient)
    }
}

impl<V, F, Df> Expert<V> for GeneralizedLinearModel<V, F, Df>
    where F: Fn(f64) -> f64,
          Df: Fn(f64) -> f64,
          V: Vector<Scalar = f64>
{
    type Gradient = (V::Scalar, V);

    fn predict(&self, input: &V) -> f64 {
        let f = &self.g;
        f(self.linear.predict(&input))
    }

    fn gradient(&self, input: &V) -> (V::Scalar, V) {
        let f = &self.g_derivate;
        self.linear.gradient(input).mul_scalar(f(self.linear.predict(&input)))
    }
}
