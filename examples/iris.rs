/// Multiclass analyses of the iris dataset build on binary logistic classifaction
///
/// Also demonstrates online training, and usage of custom feature type
extern crate csv;
extern crate vikos;
extern crate rustc_serialize;

use vikos::{Teacher, Model};
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

fn main() {

    let teacher = vikos::teacher::Nesterov {
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
