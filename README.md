[![Build Status](https://travis-ci.org/blue-yonder/vikos.svg?branch=master)](https://travis-ci.org/blue-yonder/vikos)
[![Docs](https://docs.rs/vikos/badge.svg)](https://docs.rs/vikos/)
[![MIT licensed](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/blue-yonder/vikos/blob/master/LICENSE)
[![Published](http://meritbadge.herokuapp.com/vikos)](https://crates.io/crates/vikos)

A library for supervised trainining of parameterized, regression and classification models

Design Goals
------------

* Changing model representation, cost function and optimization algorithm independently from each other
* Generics: Not commiting to a particular datastructure for inputs, targets etc.
* If the design goals above can only be achieved by sacrificing performance, so be it

Current State
-------------

Just starting to get the traits right, by continously trying new use cases
and implementing the learning algorithms. If you are looking for more mature
rust libraries in the domain of ML you might want to check out:
* [rustlearn]
* [leaf]

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