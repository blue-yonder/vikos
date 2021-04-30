use csv;
use std::default::Default;
use vikos::{cost, model, teacher};

#[test]
fn estimate_median() {
    use vikos::Teacher;

    let features = ();
    let history = [1.0, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //median is seven

    let cost = cost::LeastAbsoluteDeviation {};
    let mut model = 0.0;

    let teacher = teacher::GradientDescentAl { l0: 0.9, t: 9.0 };
    let mut training = teacher.new_training(&model);

    for &truth in history.iter().cycle().take(150) {
        teacher.teach_event(&mut training, &mut model, &cost, &features, truth);
        println!("model: {:?}", model);
    }

    assert!(model < 7.1);
    assert!(model > 6.9);
}

#[test]
fn estimate_mean() {
    use vikos::Teacher;

    let features = ();
    let history = [1f64, 3.0, 4.0, 7.0, 8.0, 11.0, 29.0]; //mean is 9

    let cost = cost::LeastSquares {};
    let mut model = 0.0;

    let teacher = teacher::GradientDescentAl { l0: 0.3, t: 4.0 };
    let mut training = teacher.new_training(&model);

    for &truth in history.iter().cycle().take(100) {
        teacher.teach_event(&mut training, &mut model, &cost, &features, truth);
        println!("model: {:?}", model);
    }

    assert!(model < 9.1);
    assert!(model > 8.9);
}

#[test]
fn linear_stochastic_gradient_descent() {
    use vikos::learn_history;

    let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

    let mut model = model::Linear { m: 0.0, c: 0.0 };

    let teacher = teacher::GradientDescent { learning_rate: 0.2 };

    let cost = cost::LeastSquares {};
    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(20).cloned(),
    );

    assert!(model.m < 1.1);
    assert!(model.m > 0.9);
    assert!(model.c < 3.1);
    assert!(model.c > 2.9);
}

#[test]
fn linear_stochastic_gradient_descent_iter() {
    use vikos::Teacher;

    let history = [(0f64, 3f64), (1.0, 4.0), (2.0, 5.0)];

    let cost = cost::LeastSquares {};
    let mut model = model::Linear { m: 0.0, c: 0.0 };

    let teacher = teacher::GradientDescent { learning_rate: 0.2 };
    let mut training = teacher.new_training(&model);

    for &(features, truth) in history.iter().cycle().take(20) {
        teacher.teach_event(&mut training, &mut model, &cost, &features, truth);
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

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(1500).cloned(),
    );

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

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(1500).cloned(),
    );

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

    let history = [
        ([2.7, 2.5], 0.0),
        ([1.4, 2.3], 0.0),
        ([3.3, 4.4], 0.0),
        ([1.3, 1.8], 0.0),
        ([3.0, 3.0], 0.0),
        ([7.6, 2.7], 1.0),
        ([5.3, 2.0], 1.0),
        ([6.9, 1.7], 1.0),
        ([8.6, -0.2], 1.0),
        ([7.6, 3.5], 1.0),
    ];

    let mut model = model::Logistic::default();
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::LeastSquares {};

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(40).cloned(),
    );

    println!("{:?}", model);

    let classification_errors = history
        .iter()
        .map(|&(input, truth)| model.predict(&input).round() == truth)
        .fold(
            0,
            |errors, correct| if correct { errors } else { errors + 1 },
        );

    assert_eq!(0, classification_errors);
}

#[test]
fn logistic_sgd_2d_max_likelihood() {
    use vikos::{learn_history, Model};

    let history = [
        ([2.7, 2.5], 0.0),
        ([1.4, 2.3], 0.0),
        ([3.3, 4.4], 0.0),
        ([1.3, 1.8], 0.0),
        ([3.0, 3.0], 0.0),
        ([7.6, 2.7], 1.0),
        ([5.3, 2.0], 1.0),
        ([6.9, 1.7], 1.0),
        ([8.6, -0.2], 1.0),
        ([7.6, 3.5], 1.0),
    ];

    let mut model = model::Logistic::default();
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::MaxLikelihood {};

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(20).cloned(),
    );

    println!("{:?}", model);

    let classification_errors = history
        .iter()
        .map(|&(input, truth)| model.predict(&input).round() == truth)
        .map(|correct| if correct { 0 } else { 1 })
        .sum();

    assert_eq!(0, classification_errors);
}

#[test]
fn logistic_sgd_2d_max_likelihood_bool() {
    use vikos::{learn_history, Crisp, Model};

    let history = [
        ([2.7, 2.5], false),
        ([1.4, 2.3], false),
        ([3.3, 4.4], false),
        ([1.3, 1.8], false),
        ([3.0, 3.0], false),
        ([7.6, 2.7], true),
        ([5.3, 2.0], true),
        ([6.9, 1.7], true),
        ([8.6, -0.2], true),
        ([7.6, 3.5], true),
    ];

    let mut model = model::Logistic::default();
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::MaxLikelihood {};

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(20).cloned(),
    );

    println!("{:?}", model);

    let classification_errors = history
        .iter()
        .map(|&(input, truth)| model.predict(&input).crisp() == truth)
        .map(|correct| if correct { 0 } else { 1 })
        .sum();

    assert_eq!(0, classification_errors);
}

#[test]
fn logistic_adagard_2d_max_likelihood_bool() {
    use vikos::{learn_history, Crisp, Model};

    let history = [
        ([2.7, 2.5], false),
        ([1.4, 2.3], false),
        ([3.3, 4.4], false),
        ([1.3, 1.8], false),
        ([3.0, 3.0], false),
        ([7.6, 2.7], true),
        ([5.3, 2.0], true),
        ([6.9, 1.7], true),
        ([8.6, -0.2], true),
        ([7.6, 3.5], true),
    ];

    let mut model = model::Logistic::default();
    let teacher = teacher::Adagard {
        learning_rate: 3.0,
        epsilon: 10000.0,
    };
    let cost = cost::MaxLikelihood {};

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(60).cloned(),
    );

    println!("{:?}", model);

    let classification_errors = history
        .iter()
        .map(|&(input, truth)| model.predict(&input).crisp() == truth)
        .map(|correct| if correct { 0 } else { 1 })
        .sum();

    assert_eq!(0, classification_errors);
}

#[test]
fn generalized_linear_model_as_logistic_regression() {
    use vikos::{learn_history, Crisp, Model};

    let history = [
        ([2.7, 2.5], false),
        ([1.4, 2.3], false),
        ([3.3, 4.4], false),
        ([1.3, 1.8], false),
        ([3.0, 3.0], false),
        ([7.6, 2.7], true),
        ([5.3, 2.0], true),
        ([6.9, 1.7], true),
        ([8.6, -0.2], true),
        ([7.6, 3.5], true),
    ];

    let mut model = model::GeneralizedLinearModel::new(
        |x| 1.0 / (1.0 + x.exp()),
        |x| -x.exp() / (1.0 + x.exp()).powi(2),
    );
    let teacher = teacher::GradientDescent { learning_rate: 0.3 };
    let cost = cost::MaxLikelihood {};

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(20).cloned(),
    );

    let classification_errors = history
        .iter()
        .map(|&(input, truth)| model.predict(&input).crisp() == truth)
        .map(|correct| if correct { 0 } else { 1 })
        .sum();

    assert_eq!(0, classification_errors);
}

#[test]
fn iris() {
    use csv;
    use vikos::{learn_history, Crisp, Model};

    let mut model = model::OneVsRest::<[model::Logistic<[f64; 4]>; 3]>::default();
    let teacher = vikos::teacher::Nesterov {
        l0: 0.0001,
        t: 1000.0,
        inertia: 0.99,
    };
    let cost = cost::MaxLikelihood {};

    let history: Vec<_> = csv::Reader::from_path("examples/data/iris.csv")
        .expect("File is ok")
        .deserialize()
        .map(|row| {
            let (t, f): (String, _) = row.unwrap();
            (t, f)
        })
        .map(|(truth, features)| {
            (
                features,
                match truth.as_ref() {
                    "setosa" => 0,
                    "versicolor" => 1,
                    "virginica" => 2,
                    _ => panic!("unknow class"),
                },
            )
        })
        .collect();

    learn_history(
        &teacher,
        &cost,
        &mut model,
        history.iter().cycle().take(3000).cloned(),
    );

    println!("{:?}", model);

    let classification_errors = history
        .iter()
        .map(|&(input, truth)| model.predict(&input).crisp() == truth)
        .map(|correct| if correct { 0 } else { 1 })
        .sum();

    assert_eq!(3, classification_errors);
}
