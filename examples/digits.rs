//! Train a classifier to distinguish between hand written digits
//! This example uses the MINST datas et for training. Since it is kind of large users are
//! encouraged to download it themselves from http://yann.lecun.com/exdb/mnist/. The data needs to
//! be unpacked into the `examples/data` for this example to work. Since the minst dataset comes in
//! its own special file format we have to write the parsing logic ourselves. It's implemented in
//! the label and image iterator types.
extern crate byteorder;

use std::fs::File;
use std::path::Path;
use std::error::Error;
use std::io::Read;
use byteorder::{BigEndian, ReadBytesExt};

fn main() {

    let labels = LabelIterator::new(&Path::new("examples/data/train-labels.idx1-ubyte"));
    let images = ImageIterator::new(&Path::new("examples/data/train-images.idx3-ubyte"));

    let events = labels.zip(images);

    for (truth, features) in events {
        print!("\n{}\n", truth);
        for (index, pixel) in features.iter().enumerate() {
            if index % 28 == 0 {
                println!("");
            }
            print!("{}", pixel);
        }
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