A library for supervised trainining of parameterized, regression and classification models

Design Goals
------------

* Changing model representation, cost function and optimization algorithm independently from each other
* Generic: Not commiting to a particular datastructure for i.e. Input Vectors
* If the design goals above can only be achieved by sacrificing performance, so be it 

Current State
-------------

Just starting to get the traits right, by continously trying new use cases
and implementing the learning algorithms.

If you are looking for more mature rust libraries in the domain of ML you might want to check out:
* [rustlearn]
* [leaf]

Getting started
---------------

1. Use [rustup] to setup rust.
3. Add the dependency in your cargo toml:
```
[dependencies]
vikos = "0.1"
```

Documentation
-------------

You still need to build the documentation yourself.

1. Use [rustup] to setup rust.
2. Clone this crate
3. run `cargo doc --open`

Contributing
------------

Want to help out? Just create an issue, pull request or contact markus.klein@blue-yonder.com

[rustup]:  http://www.rustup.rs
[rustlearn]: https://github.com/maciejkula/rustlearn
[leaf]: https://github.com/autumnai/leaf