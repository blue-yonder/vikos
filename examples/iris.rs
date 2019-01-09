/// Multiclass analyses of the iris dataset build on binary logistic classifaction
///
/// Also demonstrates online training, and usage of custom feature type
extern crate csv;
extern crate serde;
extern crate vikos;

use std::default::Default;
use vikos::{Crisp, Model, Teacher};

const PATH: &'static str = "examples/data/iris.csv";

type Features = [f64; 4];

fn main() {
    let teacher = vikos::teacher::Nesterov {
        l0: 0.0001,
        t: 1000.0,
        inertia: 0.99,
    };
    let cost = vikos::cost::MaxLikelihood {};

    // Train three individual three Logistic models, one for each class of Iris.
    let mut model = vikos::model::OneVsRest::<[vikos::model::Logistic<_>; 3]>::default();

    // Each of the classifieres has its own training state
    let mut training = teacher.new_training(&model);

    // Read iris Data
    for epoch in 0..300 {
        let mut rdr = csv::Reader::from_path(PATH).expect("File is ok");
        let mut hit = 0;
        let mut miss = 0;

        for row in rdr.deserialize() {
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
            let prediction = model.predict(&features).crisp();

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
