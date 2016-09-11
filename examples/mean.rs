extern crate vikos;
use vikos::{model, cost, teacher, learn_history, Model};

fn main(){

    //mean is 9, but we don't know that yet of course
    let history = [
    (2.0, 1.0), (3.0, 3.0), (3.5, 4.0),
    (5.0, 7.0), (5.5, 8.0), (7.0, 11.0),
    (16.0, 29.0)
    ];
    // The mean is just a simple number ...
    let mut model = model::Constant::new(0.0);
    // ... which minimizes the square error
    let cost = cost::LeastSquares{};
    // Use Stochasic Gradient Descent with an annealed learning rate
    let teacher = teacher::GradientDescentAl{ l0 : 0.3, t : 4.0 };
    // Train 100 (admitettly repetitive) events
    learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(100).cloned());
    // We need an input vector for predictions, the 42 won't influence the mean
    println!("{}", model.predict(&42.0));
    // Since we know models type is `Constant` we could just access the members
    println!("{}", model.c);
}