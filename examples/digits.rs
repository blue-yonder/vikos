//! Train a classifier to distinguish between hand written digits
//! This example uses the MINST datas et for training. Since it is kind of large users are
//! encouraged to download it themselves from http://yann.lecun.com/exdb/mnist/. The data needs to
//! be unpacked into the `examples/data` for this example to work. Since the minst dataset comes in
//! its own special file format we have to write the parsing logic ourselves. It's implemented in
//! the label and image iterator types.
extern crate byteorder;
extern crate rand;
extern crate vikos;

use std::fs::File;
use std::path::Path;
use std::error::Error;
use std::io::Read;
use byteorder::{BigEndian, ReadBytesExt};
use vikos::{Model, Teacher, Crisp};

fn main() {

    let mut model = DigitsClassifier::new();
    let teacher = vikos::teacher::GradientDescent { learning_rate: 0.0001 };
    let cost = vikos::cost::MaxLikelihood {};
    let mut training = teacher.new_training(&model);

    let labels = LabelIterator::new(&Path::new("examples/data/train-labels.idx1-ubyte"));
    let images = ImageIterator::new(&Path::new("examples/data/train-images.idx3-ubyte"));

    let events = labels.zip(images);

    let mut hit = 0;
    let mut miss = 0;

    for (truth, features) in events {
        let truth = truth as usize;
        // print current image
        for (index, pixel) in features.iter().enumerate() {
            if index % 28 == 0 {
                println!("");
            }
            print!("{:>4}", pixel);
        }

        teacher.teach_event(&mut training, &mut model, &cost, &features, truth);

        // Make prediction using current expertise
        let prediction = model.predict(&features).crisp();

        // calculate rolling accuracy
        if prediction == truth {
            hit += 1;
        } else {
            miss += 1;
        }
        let accuracy = hit as f64 / (hit + miss) as f64;

        println!("\ntruth: {}, prediction: {}, accuracy: {}",
                 truth,
                 prediction,
                 accuracy);
    }
}

/// A neural network with two layers designed to classify digits from a 28 by 28 image
///
/// Input layer consists of 28 by 28 nodes
/// Hidden layer consits of 15 hidden nodes
/// Output layer consits of 10 output nodes
struct DigitsClassifier {
    input_to_hidden: [f64; 28 * 28 * 15],
    hidden_biases: [f64; 15],
    hidden_to_output: [f64; 15 * 10],
    output_biases: [f64; 10],
}

impl DigitsClassifier {
    /// Builds a new Neural Network with randomly choosen weights between -1.0 and 1.0
    fn new() -> DigitsClassifier {
        let layer_one = [0.0; 28 * 28 * 15];
        let bias_one = [0.0; 15];
        let layer_two = [0.0; 15 * 10];
        let bias_two = [0.0; 10];

        let mut result = DigitsClassifier {
            input_to_hidden: layer_one,
            hidden_to_output: layer_two,
            hidden_biases: bias_one,
            output_biases: bias_two,
        };

        let between = rand::distributions::Normal::new(0.0, 1.0);
        let mut rng = rand::thread_rng();

        use rand::distributions::IndependentSample;

        for i in 0..result.num_coefficients() {
            *result.coefficient(i) = between.ind_sample(&mut rng);
        }

        result
    }

    /// Returns the activation values for the n-th hidden neuron
    fn activate_hidden_n(&self, features: &[u8; 28 * 28], n: usize) -> f64 {
        (&self.input_to_hidden[(n * features.len())..((n + 1) * features.len())])
            .iter()
            .zip(features.iter().map(|&i| i as f64 / 255.0))
            .map(|(&w, f)| f * w)
            .sum::<f64>() + self.hidden_biases[n]
    }

    /// Returns the activation values for the hidden layer
    fn activate_hidden(&self, features: &[u8; 28 * 28]) -> [f64; 15] {
        let mut hidden = [0.0; 15];
        for i in 0..15 {
            hidden[i] = self.activate_hidden_n(features, i);
        }
        hidden
    }
}

impl Model for DigitsClassifier {
    type Features = [u8; 28 * 28];
    type Target = [f64; 10];

    fn num_coefficients(&self) -> usize {
        28 * 28 * 15 + 15 * 10 + 15 + 10
    }

    fn coefficient(&mut self, index: usize) -> &mut f64 {
        if index < 28 * 28 * 15 {
            &mut self.input_to_hidden[index]
        } else if index < 28 * 28 * 15 + 15 {
            &mut self.hidden_biases[index - 28 * 28 * 15]
        } else if index < 28 * 28 * 15 + 15 + 15 * 10 {
            &mut self.hidden_to_output[index - 28 * 28 * 15 - 15]
        } else {
            &mut self.output_biases[index - 28 * 28 * 15 - 15 * 10 - 15]
        }
    }

    fn predict(&self, features: &Self::Features) -> [f64; 10] {
        let hidden = self.activate_hidden(features);
        let mut output = [0.0; 10];
        for i in 0..10 {
            output[i] = (&self.hidden_to_output[(i * hidden.len())..((i + 1) * hidden.len())])
                .iter()
                .zip(hidden.iter())
                .map(|(&w, &h)| h * w)
                .sum::<f64>() + self.output_biases[i];
        }
        output
    }

    fn gradient(&self, index: usize, input: &Self::Features) -> [f64; 10] {
        let mut output = [0.0; 10];
        if index < 28 * 28 * 15 {
            let hidden = index / input.len();
            for i in 0..output.len() {
                output[i] = (input[hidden] as f64 / 255.0) * self.hidden_to_output[i * 15 + hidden];
            }
        } else if index < 28 * 28 * 15 + 15 {
            let index = index - 28 * 28 * 15;
            for i in 0..output.len() {
                output[i] = self.hidden_to_output[i * 15 + index % 15];
            }
        } else if index < 28 * 28 * 15 + 15 * 10 + 15 {
            let index = index - 28 * 28 * 15 - 15;
            output[index / 15] = self.activate_hidden_n(input, index % 15);
        } else {
            let index = index - 28 * 28 * 15 - 15 * 10 - 15;
            output[index % 10] = 1.0;
        }
        output
    }
}

/// Reads the labels from the file
struct LabelIterator {
    file: File,
    items_left: u32,
}

impl LabelIterator {
    fn new(path: &Path) -> LabelIterator {
        match File::open(&path) {
            Err(why) => {
                panic!("unable to open file {}. Reason: '{}' Please download minst data set from \
                        http://yann.lecun.com/exdb/mnist/ and unpack it into 'examples/data'.",
                       path.display(),
                       why.description())
            }
            Ok(mut file) => {
                let magic = file.read_u32::<BigEndian>().unwrap();
                assert!(magic == 2049);
                let number_of_items = file.read_u32::<BigEndian>().unwrap();
                LabelIterator {
                    file: file,
                    items_left: number_of_items,
                }
            }
        }
    }
}

impl Iterator for LabelIterator {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        match self.items_left {
            0 => None,
            _ => {
                self.items_left -= 1;
                Some(self.file.read_u8().unwrap())
            }
        }
    }
}

/// Image Iterator
struct ImageIterator {
    file: File,
    items_left: u32,
}

impl ImageIterator {
    fn new(path: &Path) -> ImageIterator {
        match File::open(&path) {
            Err(why) => {
                panic!("unable to open file {}. Reason: '{}' Please download minst data set from \
                        http://yann.lecun.com/exdb/mnist/ and unpack it into 'examples/data'.",
                       path.display(),
                       why.description())
            }
            Ok(mut file) => {
                let magic = file.read_u32::<BigEndian>().unwrap();
                assert!(magic == 2051);
                let number_of_items = file.read_u32::<BigEndian>().unwrap();
                let rows = file.read_u32::<BigEndian>().unwrap();
                assert!(rows == 28);
                let columns = file.read_u32::<BigEndian>().unwrap();
                assert!(columns == 28);
                ImageIterator {
                    file: file,
                    items_left: number_of_items,
                }
            }
        }
    }
}

impl Iterator for ImageIterator {
    type Item = [u8; 28 * 28];

    fn next(&mut self) -> Option<Self::Item> {
        match self.items_left {
            0 => None,
            _ => {
                self.items_left -= 1;
                let mut result = [0; 28 * 28];
                self.file.read_exact(&mut result).unwrap();
                Some(result)
            }
        }
    }
}