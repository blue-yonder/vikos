Overview of regular commands for working and developing vikos.

Building
--------

* `cargo build` makes a simple build.
* When using the nightly build `cargo build --features clippy ` will build 
  vikos with added checks and hints from the 
  [clippy](https://github.com/Manishearth/rust-clippy) package.


Execute Tests
-------------

Tests are run using `cargo test`.

Executing Examples
------------------

The examples in the `examples/` directory can be compiled and run using `cargo
run --example <example-name>`. So `cargo run --example mean` will compile and
run the file `examples/mean.rs`.

Source Code Formatting
----------------------

[Rustfmt](https://github.com/rust-lang-nursery/rustfmt) is the appropriate tool
to lay out code according to the community standards. You can invoke it with
`cargo fmt` (you can install it via `cargo install rustfmt`).


