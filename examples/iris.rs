/// Multiclass analyses of the iris dataset build on binary logistic classifaction
///
/// Also demonstrates online training, and usage of custom feature type
extern crate csv;
extern crate vikos;
extern crate rustc_serialize;

use vikos::{Teacher, Model, Cost};
use vikos::linear_algebra::Vector;
use std::default::Default;

const PATH: &'static str = "examples/data/iris.csv";

type Features = [f64; 4];

#[derive(Default)]
struct IrisModel {
    // We train three binary classifiers. One for each specis of iris.
    // Each of this classifiers will tell us the propability of an
    // observation belonging to its particular class
    class_models: [vikos::model::Logistic<Features>; 3],
}

impl Model for IrisModel {
    type Features = Features;
    type Target = [f64; 3];

    fn num_coefficients(&self) -> usize {
        // self.class_models.map(|m| m.num_coefficients()).sum()
        3 * (4 + 1)
    }

    fn coefficient(&mut self, index: usize) -> &mut f64 {
        self.class_models[index / (4 + 1)].coefficient(index % (4 + 1))
    }

    fn predict(&self, input: &Self::Features) -> Self::Target {
        [self.class_models[0].predict(input),
         self.class_models[1].predict(input),
         self.class_models[2].predict(input)]
    }

    fn gradient(&self, coefficient: usize, input: &Self::Features) -> [f64; 3] {
        let index = coefficient / (4 + 1);
        let mut result = [0.0; 3];
        result[index] = self.class_models[index].gradient(coefficient % (4 + 1), input);
        result
    }
}

struct IrisTeacher {
    /// Start learning rate
    pub l0: f64,
    /// Smaller t will decrease the learning rate faster
    ///
    /// After t events the start learning rate will be a half `l0`,
    /// after two t events the learning rate will be one third `l0`,
    /// and so on.
    pub t: f64,
    /// To simulate friction, please select a value smaller than 1 (recommended)
    pub inertia: f64,
}

impl<M> Teacher<M> for IrisTeacher
    where M: Model<Target = [f64; 3]>
{
    type Training = (usize, Vec<f64>);

    fn new_training(&self, model: &M) -> (usize, Vec<f64>) {

        let mut velocity = Vec::with_capacity(model.num_coefficients());
        velocity.resize(model.num_coefficients(), 0.0);

        (0, velocity)
    }

    fn teach_event<Y, C>(&self,
                         training: &mut Self::Training,
                         model: &mut M,
                         cost: &C,
                         features: &M::Features,
                         truth: Y)
        where C: Cost<Y, [f64; 3]>,
              Y: Copy
    {
        let mut num_events = &mut training.0;
        let mut velocity = &mut training.1;
        let prediction = model.predict(features);
        let learning_rate = self.l0 / (1.0 + *num_events as f64 / self.t);

        for ci in 0..model.num_coefficients() {
            *model.coefficient(ci) = *model.coefficient(ci) + velocity[ci];
        }
        for ci in 0..model.num_coefficients() {
            let delta = -learning_rate *
                        cost.outer_derivative(&prediction, truth)
                .dot(&model.gradient(ci, features));
            *model.coefficient(ci) = *model.coefficient(ci) + delta;
            velocity[ci] = self.inertia * velocity[ci] + delta;
        }
        *num_events += 1;
    }
}

fn main() {

    let teacher = IrisTeacher {
        l0: 0.0001,
        t: 1000.0,
        inertia: 0.99,
    };
    let cost = vikos::cost::MaxLikelihood {};

    let mut model = IrisModel::default();

    // Each of the classifieres has its own training state
    let mut training = teacher.new_training(&model);

    // Read iris Data
    for epoch in 0..300 {
        let mut rdr = csv::Reader::from_file(PATH).expect("File is ok");
        let mut hit = 0;
        let mut miss = 0;

        for row in rdr.decode() {

            // Learn event
            let (truth, features): (String, Features) = row.unwrap();

            let class = match truth.as_ref() {
                "setosa" => 0,
                "versicolor" => 1,
                "virginica" => 2,
                _ => panic!("unknown Iris class: {}", truth),
            };

            teacher.teach_event(&mut training, &mut model, &cost, &features, class);

            // Make prediction using current expertise
            let p = model.predict(&features);
            let prediction = (0..3).fold(0, |m, c| if p[c] > p[m] { c } else { m });

            if prediction == class {
                hit += 1;
            } else {
                miss += 1;
            }
        }
        let accuracy = hit as f64 / (hit + miss) as f64;

        println!("epoch: {}, accuracy: {}", epoch, accuracy);
    }
}
