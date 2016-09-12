extern crate vikos;
use vikos::{model, cost, teacher, learn_history};

fn main(){

    //mean is 9, but of course we do not know that yet
    let history = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0];
    // The mean is just a simple number ...
    let mut model = model::Constant::new(0.0);
    // ... which minimizes the square error
    let cost = cost::LeastSquares{};
    // Use stochasic gradient descent with an annealed learning rate
    let teacher = teacher::GradientDescentAl{ l0 : 0.3, t : 4.0 };
    // Train 100 (admitettly repetitive) events
    // We use the `map` iterator adaptor to extend an empty feature vector to each data point
    learn_history(&teacher, &cost, &mut model, history.iter().cycle().take(100).map(|&y|((),y)));
    // Since we know the model's type is `Constant`, we could just access the members
    println!("{}", model.c);
}