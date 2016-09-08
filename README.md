[![Build Status](https://travis-ci.org/blue-yonder/vikos.svg?branch=master)](https://travis-ci.org/blue-yonder/vikos)
[![Docs](https://docs.rs/vikos/badge.svg)](https://docs.rs/vikos/)

A library for supervised trainining of parameterized, regression and classification models

Design Goals
------------

* Changing model representation, cost function and optimization algorithm independently from each other
* Generic: Not commiting to a particular datastructure for inputs, targets etc.
* If the design goals above can only be achieved by sacrificing performance, so be it 

Current State
-------------

Just starting to get the traits right, by continously trying new use cases
and implementing the learning algorithms. If you are looking for more mature
rust libraries in the domain of ML you might want to check out:
* [rustlearn]
* [leaf]

Getting started
---------------

1. Use [rustup] to setup rust.
3. Add the dependency in your cargo toml:
```
[dependencies]
vikos = "0.1.2"
```

Documentation
-------------

Thanks to the folks of [docs.rs] for building and hosting the [documentation]! 

Contributing
------------

Want to help out? Just create an issue, pull request or contact markus.klein@blue-yonder.com

[docs.rs]: https://docs.rs
[documentation]: https://docs.rs/vikos/
[rustup]:  http://www.rustup.rs
[rustlearn]: https://github.com/maciejkula/rustlearn
[leaf]: https://github.com/autumnai/leaf