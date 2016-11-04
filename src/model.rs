use Model;
use linear_algebra::Vector;

impl Model for f64 {
    type Features = ();
    type Target = f64;

    fn num_coefficients(&self) -> usize {
        1
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        match coefficient {
            0 => self,
            _ => panic!("coefficient index out of range"),
        }
    }

    fn predict(&self, _: &()) -> f64 {
        *self
    }

    fn gradient(&self, coefficient: usize, _: &()) -> f64 {
        match coefficient {
            0 => 1.0,
            _ => panic!("coefficient index out of range"),
        }
    }
}

/// Models the target as `y = m * x + c`
#[derive(Debug, Clone, Default, RustcDecodable, RustcEncodable)]
pub struct Linear<V> {
    /// Slope
    pub m: V,
    /// Offset
    pub c: f64,
}

impl Model for Linear<f64> {
    type Features = f64;
    type Target = f64;

    fn num_coefficients(&self) -> usize {
        2
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        if coefficient == 1 {
            &mut self.c
        } else {
            &mut self.m
        }
    }

    fn predict(&self, input: &f64) -> f64 {
        self.m * input + self.c
    }

    fn gradient(&self, coefficient: usize, input: &f64) -> f64 {

        if coefficient == 1 {
            1.0 //derive by c
        } else {
            *input //derive by m
        }
    }
}

impl<V> Model for Linear<V>
    where V: Vector
{
    type Features = V;
    type Target = f64;

    fn num_coefficients(&self) -> usize {
        self.m.dimension() + 1
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        if coefficient == self.m.dimension() {
            &mut self.c
        } else {
            self.m.mut_at(coefficient)
        }
    }

    fn predict(&self, input: &V) -> f64 {
        self.m.dot(input) + self.c
    }

    fn gradient(&self, coefficient: usize, input: &V) -> f64 {

        if coefficient == self.m.dimension() {
            1.0 //derive by c
        } else {
            input.at(coefficient) //derive by m
        }
    }
}

/// Models target as `y = 1/(1+e^(m * x + c))`
#[derive(Debug, Clone, Default, RustcDecodable, RustcEncodable)]
pub struct Logistic<V>(Linear<V>);

impl<V> Model for Logistic<V>
    where Linear<V>: Model<Features = V, Target = f64>
{
    type Features = V;
    type Target = f64;

    fn num_coefficients(&self) -> usize {
        self.0.num_coefficients()
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        self.0.coefficient(coefficient)
    }

    fn predict(&self, input: &V) -> f64 {
        1.0 / (1.0 + self.0.predict(input).exp())
    }

    fn gradient(&self, coefficient: usize, input: &V) -> f64 {
        let p = self.predict(input);
        -p * (1.0 - p) * self.0.gradient(coefficient, input)
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
pub struct GeneralizedLinearModel<V, G, Dg> {
    /// `Linear` term of the generalized linear `Model`
    pub linear: Linear<V>,
    /// Outer function applied to the result of `linear`
    pub g: G,
    /// Derivation of `g`
    pub g_derivate: Dg,
}

impl<V, G, Dg> GeneralizedLinearModel<V, G, Dg>
    where G: Fn(f64) -> f64,
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
          Linear<V>: Model<Features = V, Target = f64>
{
    type Features = V;
    type Target = f64;

    fn num_coefficients(&self) -> usize {
        self.linear.num_coefficients()
    }

    fn coefficient(&mut self, coefficient: usize) -> &mut f64 {
        self.linear.coefficient(coefficient)
    }

    fn predict(&self, input: &V) -> f64 {
        let f = &self.g;
        f(self.linear.predict(&input))
    }

    fn gradient(&self, coefficient: usize, input: &V) -> f64 {
        let f = &self.g_derivate;
        f(self.linear.predict(&input)) * self.linear.gradient(coefficient, input)
    }
}
