Changelog
=========

0.3.1
-----

* `Teacher`s now support `serde::{Serialize, Deserialize}`

0.3.0
-----

* `FixedDimension` introduced to distinguish Vector types which dimension is known at compile time.
* `Linear` and `Logistic` models have gained new constructors `with_feature_dimension`.
* `Linear` and `Logistic` models are no longer default constructible with Vector types whose
  dimension is unknown at compile time. This has been introduced to catch an error a where dot product
  of vectors with different dimensionality has been calculated at compile time.
* A new example `iris_vec` has been added. Using a dynamically allocated vector for the features.

0.2.1
-----

* Implement `linear_algebra::Vector` for `Vec<f64>`.