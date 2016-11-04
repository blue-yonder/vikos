/// Multiclass analyses of the iris dataset build on binary logistic classifaction
///
/// Also demonstrates online training, and usage of custom feature type
extern crate csv;
extern crate vikos;
extern crate rustc_serialize;

use vikos::{Teacher, Model, Cost};
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

    fn gradient(&self, coefficient: usize, input: &Self::Features) -> f64 {
        0.0
    }
}

struct IrisTeacher {
    teacher: vikos::teacher::Nesterov,
}

impl Teacher<IrisModel> for IrisTeacher {
    type Training = [<vikos::teacher::Nesterov as Teacher<vikos::model::Logistic<Features>>>::Training; 3];

    fn new_training(&self, model: &IrisModel) -> Self::Training {
        // Each of the classifieres has its own training state
        [self.teacher.new_training(&model.class_models[0]),
         self.teacher.new_training(&model.class_models[1]),
         self.teacher.new_training(&model.class_models[2])]
    }

    fn teach_event<Y, C>(&self,
                         training: &mut Self::Training,
                         model: &mut IrisModel,
                         cost: &C,
                         features: &Features,
                         truth: Y)
        where C: Cost<Y>,
              Y: Copy
    {
    }
}

fn main() {

    let teacher = IrisTeacher {
        teacher: vikos::teacher::Nesterov {
            l0: 0.0001,
            t: 1000.0,
            inertia: 0.99,
        },
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

            for i in 0..3 {
                teacher.teacher.teach_event(&mut training[i],
                                            &mut model.class_models[i],
                                            &cost,
                                            &features,
                                            class == i);
            }

            // Make prediction using current expertise
            let p = model.predict(&features);
            let prediction = if p[0] > p[1] {
                if p[0] > p[2] { "setosa" } else { "virginica" }
            } else {
                if p[1] > p[2] {
                    "versicolor"
                } else {
                    "virginica"
                }
            };

            if prediction == truth {
                hit += 1;
            } else {
                miss += 1;
            }
        }
        let accuracy = hit as f64 / (hit + miss) as f64;

        println!("epoch: {}, accuracy: {}", epoch, accuracy);
    }
}
