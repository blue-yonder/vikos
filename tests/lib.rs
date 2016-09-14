extern crate vikos;

use vikos::{cost, model, training, teacher};

#[test]
fn estimate_median() {

    use vikos::Training;

    let features = ();
    let history = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //median is seven

    let cost = cost::LeastAbsoluteDeviation {};
    let mut model = model::Constant::new(0.0);

    let mut training = training::GradientDescentAl {
        l0: 0.9,
        t: 9.0,
        learned_events: 0.0,
    };

    for &truth in history.iter().cycle().take(150) {

        training.teach_event(&cost, &mut model, &features, truth);
        println!("model: {:?}, learning_rate: {:?}",
                 model,
                 training.learning_rate());
    }

    assert!(model.c < 7.1);
    assert!(model.c > 6.9);
}

#[test]
fn estimate_mean() {

    use vikos::Training;

    let features = ();
    let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9

    let cost = cost::LeastSquares {};
    let mut model = model::Constant::new(0.0);

    let mut training = training::GradientDescentAl {
        l0: 0.3,
        t: 4.0,
        learned_events: 0.0,
    };

    for &truth in history.iter().cycle().take(100) {

        training.teach_event(&cost, &mut model, &features, truth);
        println!("model: {:?}, learning_rate: {:?}",
                 model,
                 training.learning_rate());
    }

    assert!(model.c < 9.1);
    assert!(model.c > 8.9);
}

#[test]
fn linear_stochastic_gradient_descent() {

    use vikos::learn_history;

    let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

    let mut model = model::Linear { m: 0.0, c: 0.0 };

    let teacher = teacher::GradientDescent { learning_rate: 0.2 };

    let cost = cost::LeastSquares {};
    learn_history(&teacher,
                  &cost,
                  &mut model,
                  history.iter().cycle().take(20).cloned());

    assert!(model.m < 1.1);
    assert!(model.m > 0.9);
    assert!(model.c < 3.1);
    assert!(model.c > 2.9);
}

#[test]
fn linear_stochastic_gradient_descent_iter() {

    use vikos::Teacher;
    use vikos::Training;

    let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

    let cost = cost::LeastSquares {};
    let mut model = model::Linear { m: 0.0, c: 0.0 };

    let teacher = teacher::GradientDescent { learning_rate: 0.2 };
    let mut training = teacher.new_training(&model);

    for &(features, truth) in history.iter().cycle().take(20) {

        training.teach_event(&cost, &mut model, &features, truth);
        println!("model: {:?}", model);
    }

    assert!(model.m < 1.1);
    assert!(model.m > 0.9);
    assert!(model.c < 3.1);
    assert!(model.c > 2.9);
}

#[test]
fn linear_sgd_2d() {

    use vikos::learn_history;

    let history = [([0.0, 7.0], 17.0), ([1.0, 2.0], 8.0), ([2.0, -2.0], 1.0)];
    let mut model = model::Linear {
        m: [0.0, 0.0],
        c: 0.0,
    };
    let cost = cost::LeastSquares {};
    let teacher = teacher::Momentum {
        l0: 0.009,
        t: 1000.0,
        inertia: 0.995,
    };

    learn_history(&teacher,
                  &cost,
                  &mut model,
                  history.iter().cycle().take(1500).cloned());

    println!("{:?}", model);

    assert!(model.m[0] < 1.1);
    assert!(model.m[0] > 0.9);
    assert!(model.m[1] < 2.1);
    assert!(model.m[1] > 1.9);
    assert!(model.c < 3.1);
    assert!(model.c > 2.9);
}

#[test]
fn linear_nesterov_2d() {

    use vikos::learn_history;

    let history = [([0.0, 7.0], 17.0), ([1.0, 2.0], 8.0), ([2.0, -2.0], 1.0)];
    let mut model = model::Linear {
        m: [0.0, 0.0],
        c: 0.0,
    };
    let cost = cost::LeastSquares {};
    let teacher = teacher::Nesterov {
        l0: 0.009,
        t: 1000.0,
        inertia: 0.995,
    };

    learn_history(&teacher,
                  &cost,
                  &mut model,
                  history.iter().cycle().take(1500).cloned());

    println!("{:?}", model);

    assert!(model.m[0] < 1.1);
    assert!(model.m[0] > 0.9);
    assert!(model.m[1] < 2.1);
    assert!(model.m[1] > 1.9);
    assert!(model.c < 3.1);
    assert!(model.c > 2.9);
}

#[test]
fn logistic_sgd_2d_least_squares() {

    use vikos::{learn_history, Model};

    let history = [([2.7, 2.5], 0.0),
                   ([1.4, 2.3], 0.0),
                   ([3.3, 4.4], 0.0),
                   ([1.3, 1.8], 0.0),
                   ([3.0, 3.0], 0.0),
                   ([7.6, 2.7], 1.0),
                   ([5.3, 2.0], 1.0),
                   ([6.9, 1.7], 1.0),
                   ([8.6, -0.2], 1.0),
                   ([7.6, 3.5], 1.0)];

    let mut model = model::Logistic {
        linear: model::Linear {
            m: [0.0, 0.0],
            c: 0.0,
        },
    };
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::LeastSquares {};

    learn_history(&teacher,
                  &cost,
                  &mut model,
                  history.iter().cycle().take(40).cloned());

    println!("{:?}", model.linear);

    let classification_errors = history.iter()
        .map(|&(input, truth)| model.predict(&input).round() == truth)
        .fold(0,
              |errors, correct| if correct { errors } else { errors + 1 });

    assert_eq!(0, classification_errors);
}

#[test]
fn logistic_sgd_2d_max_likelihood() {
    use vikos::{learn_history, Model};
    use vikos::model::{Logistic, Linear};

    let history = [([2.7, 2.5], 0.0),
                   ([1.4, 2.3], 0.0),
                   ([3.3, 4.4], 0.0),
                   ([1.3, 1.8], 0.0),
                   ([3.0, 3.0], 0.0),
                   ([7.6, 2.7], 1.0),
                   ([5.3, 2.0], 1.0),
                   ([6.9, 1.7], 1.0),
                   ([8.6, -0.2], 1.0),
                   ([7.6, 3.5], 1.0)];

    let mut model = Logistic {
        linear: Linear {
            m: [0.0, 0.0],
            c: 0.0,
        },
    };
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::MaxLikelihood {};

    learn_history(&teacher,
                  &cost,
                  &mut model,
                  history.iter().cycle().take(20).cloned());

    println!("{:?}", model.linear);

    let classification_errors = history.iter()
        .map(|&(input, truth)| model.predict(&input).round() == truth)
        .fold(0,
              |errors, correct| if correct { errors } else { errors + 1 });

    assert_eq!(0, classification_errors);
}

#[test]
fn logistic_sgd_2d_max_likelihood_bool() {
    use vikos::{learn_history, Model};
    use vikos::model::{Logistic, Linear};

    let history = [([2.7, 2.5], false),
                   ([1.4, 2.3], false),
                   ([3.3, 4.4], false),
                   ([1.3, 1.8], false),
                   ([3.0, 3.0], false),
                   ([7.6, 2.7], true),
                   ([5.3, 2.0], true),
                   ([6.9, 1.7], true),
                   ([8.6, -0.2], true),
                   ([7.6, 3.5], true)];

    let mut model = Logistic {
        linear: Linear {
            m: [0.0, 0.0],
            c: 0.0,
        },
    };
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::MaxLikelihood {};

    learn_history(&teacher,
                  &cost,
                  &mut model,
                  history.iter().cycle().take(20).cloned());

    println!("{:?}", model.linear);

    let classification_errors = history.iter()
        .map(|&(input, truth)| model.predict(&input).round() == if truth {1.0} else {0.0})
        .fold(0,
              |errors, correct| if correct { errors } else { errors + 1 });

    assert_eq!(0, classification_errors);
}