use num::{Num, Zero, One};
use rustc_serialize::{Encodable, Decodable};
use std::fmt::Debug;

/// Vector whose dimension is known at runtime
///
/// Assumes the `Vector` is represented as a
/// tuple of numbers representing its projection
/// along orthogonal base vectors
pub trait Vector: Clone {
    /// Underlying scalar type of `Vector` type
    type Scalar: Num + Zero + One + Copy + Encodable + Decodable + Debug;
    /// Maximum allowed index for `at` and `mut_at`
    fn dimension(&self) -> usize;
    /// Length of projection along `i`-th base
    fn at(&self, i: usize) -> Self::Scalar;
    /// Mutable access to length of projection along `i`-th base
    fn mut_at(&mut self, i: usize) -> &mut Self::Scalar;
    /// Scalar product
    ///
    /// Default implementation using `at` and `dimension` is provided
    fn dot(&self, other: &Self) -> Self::Scalar {
        let mut result = Self::Scalar::zero();
        for i in 0..self.dimension() {
            result = result + self.at(i) * other.at(i)
        }
        result
    }
}

impl Vector for f64 {
    type Scalar = f64;

    fn dimension(&self) -> usize {
        1
    }

    fn at(&self, _: usize) -> f64 {
        *self
    }

    fn mut_at(&mut self, _: usize) -> &mut f64 {
        self
    }

    fn dot(&self, other: &f64) -> f64 {
        self * other
    }
}

impl Vector for [f64; 2] {
    type Scalar = f64;

    fn dimension(&self) -> usize {
        2
    }

    fn at(&self, index: usize) -> f64 {
        self[index]
    }

    fn mut_at(&mut self, index: usize) -> &mut f64 {
        &mut self[index]
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn dot() {

        use linear_algebra::Vector;

        let a = [1.0, 2.0];
        let b = [3.0, 4.0];

        assert_eq!(11.0, a.dot(&b))
    }
}
