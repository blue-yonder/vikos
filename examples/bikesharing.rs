extern crate vikos;
extern crate csv;
extern crate rustc_serialize;

use vikos::{cost, teacher, Model};
use std::marker::PhantomData;

#[derive(Debug, RustcDecodable)]
struct Record {
    instant: i32,
    date: String,
    season: f64,
    year: f64,
    month: f64,
    hour: f64,
    holiday: f64,
    weekday: f64,
    workingday: f64,
    weather_sit: f64,
    temp: f64,
    atemp: f64,
    huminity: f64,
    windspeed: f64,
    casual: f64,
    registered: f64,
    count: f64,
}

fn main() {

    let mut rdr = csv::Reader::from_file("examples/data/bikesharing_hour.csv").expect("Open files without problem");

    let history: Vec<_> = rdr.decode()
        .map(|r| {
            let record: Record = r.unwrap();
            record
        })
        .map(|r| {
            ((),
             r.count)
        })
        .collect();

    let mut model = BikesharingModel::default();
    let cost = cost::LeastAbsoluteDeviation {};
    let teacher = teacher::Nesterov {
        l0: 0.001,
        t: history.len() as f64 * 10.0,
        inertia: 0.999,
    };
    vikos::learn_history(&teacher,
                         &cost,
                         &mut model,
                         history.iter().cycle().take(10000 * history.len()).map(|x| *x));

    let mad = history.iter().fold(0.0, |sum, &(x, y)| sum + (model.predict(&x) - y).abs()) /
              history.len() as f64;
    println!("MAD: {}", mad);
    println!("model: {:?}", model);

}

#[derive(Debug)]
struct BikesharingModel<F>{
    count : f64,
    _phantom : PhantomData<F>
}

impl<F> Default for BikesharingModel<F>{
    fn default() -> BikesharingModel<F>{
        BikesharingModel{
            count : 0.0,
            _phantom : PhantomData::default()
        }
    }
}

impl<F> Model for BikesharingModel<F>{

    type Input = F;

    fn predict(&self, _features : &F) -> f64{
        self.count
    }

    fn num_coefficents(&self) -> usize{
        1
    }

    fn coefficent(&mut self, c : usize) -> &mut f64{
        match c {
            0 => &mut self.count,
            _ => panic!("Out of bounds")
        }
    }

    fn gradient(&self, c : usize, _input : &F) -> f64{
        match c {
            0 => 1.0,
            _ => panic!("Out of bounds")
        }
    }
}
