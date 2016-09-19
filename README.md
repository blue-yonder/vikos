[![Build Status](https://travis-ci.org/blue-yonder/vikos.svg?branch=master)](https://travis-ci.org/blue-yonder/vikos)
[![Docs](https://docs.rs/vikos/badge.svg)](https://docs.rs/vikos/)
[![MIT licensed](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/blue-yonder/vikos/blob/master/LICENSE)
[![Published](http://meritbadge.herokuapp.com/vikos)](https://crates.io/crates/vikos)
[![Join the chat at https://gitter.im/vikos-optimization/Lobby](https://badges.gitter.im/vikos-optimization/Lobby.svg)](https://gitter.im/vikos-optimization/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Vikos is a library for supervised training of parameterized, regression, and classification models.

Design Goals
------------

* Model representations, cost functions, and optimization algorithms can be changed independently of each other.
* Generics: Not committed to a particular data structure for inputs, targets, etc.
* If the design goals above can only be achieved by sacrificing performance, so be it.

Current State
-------------

Just starting to get the traits right, by continuously trying new use cases
and implementing the learning algorithms. If you are looking for more mature
rust libraries in the domain of ML, you might want to check out:
* [rusty-machine]
* [leaf].

Getting Started
---------------

1. Install the rust package manager `cargo`. Goto [rustup] and follow the instructions on
   the page (in my experience this works fine for Windows, Ubuntu and OS X).
2. Run `cargo new --bin hello_vikos`.
3. switch to the `hello_vikos` directory.
4. Run `cargo run` to execute the hello world program.
5. Edit the `Cargo.toml` file. Add `vikos = "0.1.6"` to your dependencies. The file should
   now look somewhat like this:
   ```
   [package]
   name = "hello_vikos"
   version = "0.1.0"
   authors = ["..."]

   [dependencies]
   vikos = "0.1.6"
   ```
6. Insert `extern crate vikos;` at the first line in `src/main.rs`
7. You can now start replacing code in `main` with code from the [tutorial].

   ```
   fn main(){
       /* tutorial code goes here */
   }
   ```

Documentation
-------------

Thanks to the folks of [docs.rs] for building and hosting the [documentation]!

Contributing
------------

Want to help out? Just create an issue, pull request or contact markus.klein@blue-yonder.com.

[docs.rs]: https://docs.rs
[documentation]: https://docs.rs/vikos/
[tutorial]: https://docs.rs/vikos/0.1.6/vikos/tutorial/index.html
[rustup]:  http://www.rustup.rs
[rusty-machine]: https://github.com/AtheMathmo/rusty-machine
[leaf]: https://github.com/autumnai/leaf
