/// Multiclass analyses of the iris dataset build on binary logistic classifaction
///
/// Also demonstrates online training, and usage of custom feature type
extern crate csv;
extern crate vikos;
extern crate rustc_serialize;

use vikos::{Teacher, Model};
use std::default::Default;

const PATH : &'static str = "examples/data/iris.csv";

/// A custom type used to hold all features
#[derive(Clone, Default, RustcDecodable)]
struct Features{
    sepal_length : f64,
    sepal_width : f64,
    petal_length : f64,
    petal_width : f64
}

// Trait required by `vikos::model::Logistic`
impl vikos::linear_algebra::Vector for Features{
    type Scalar = f64;

    fn dimension(&self) -> usize{
        4
    }

    fn at(&self, index : usize) -> f64{
        match index{
            0 => self.sepal_length,
            1 => self.sepal_width,
            2 => self.petal_length,
            3 => self.petal_width,
            _ => panic!("out of bounds")
        }
    }

    fn mut_at(&mut self, index : usize) -> & mut f64{
        match index{
            0 => &mut self.sepal_length,
            1 => &mut self.sepal_width,
            2 => &mut self.petal_length,
            3 => &mut self.petal_width,
            _ => panic!("out of bounds")
        }
    }
}

fn main() {

    let teacher = vikos::teacher::Nesterov{l0 : 0.0001, t: 1000.0, inertia: 0.99};
    let cost = vikos::cost::MaxLikelihood{};

    // We train three binary classifiers. One for each specis of iris.
    // Each of this classifiers will tell us the propability of an
    // observation belonging to its particular class
    let mut setosa = vikos::model::Logistic::default();
    let mut versicolor = vikos::model::Logistic::default();
    let mut virginica = vikos::model::Logistic::default();

    // Each of the classifieres has its own training state
    let mut train_setosa = teacher.new_training(&setosa, &cost);
    let mut train_versicolor = teacher.new_training(&versicolor, &cost);
    let mut train_virginica = teacher.new_training(&virginica, &cost);

    // Read iris Data
    for epoch in 0..300{
        let mut rdr = csv::Reader::from_file(PATH).expect("File is ok");
        let mut hit = 0;
        let mut miss = 0;

        for row in rdr.decode() {

            // Learn event
            let (truth, features) : (String, Features) = row.unwrap();
            teacher.teach_event(&mut train_setosa, &mut setosa, &cost, &features, truth == "setosa");
            teacher.teach_event(&mut train_versicolor, &mut versicolor, &cost, &features, truth == "setosa");
            teacher.teach_event(&mut train_virginica, &mut virginica, &cost, &features, truth == "setosa");

            // Make prediction using current expertise
            let p_setosa = setosa.predict(&features);
            let p_versicolor = versicolor.predict(&features);
            let p_virginica = virginica.predict(&features);
            let prediction = if p_setosa > p_versicolor {
                if p_setosa > p_virginica { "setosa" } else { "virginica" }
            } else {
                if p_versicolor > p_virginica { "versicolor" } else { "virginica" }
            };

            if prediction == truth{
                hit += 1;
            } else {
                miss += 1;
            }
        }
        let accuracy = hit as f64 / (hit + miss) as f64;

        println!("epoch: {}, accuracy: {}", epoch, accuracy);
    }
}
