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
    type Scalar: Num + Zero + One + Copy + Encodable + Decodable + Default + Debug;
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
    /// Scalar Multiplication
    ///
    /// Default implementation using `at` and `dimension` is provided
    fn mul_scalar(&self, scalar: Self::Scalar) -> Self{
        let mut copy = self.clone();
        for i in 0..self.dimension() {
            *copy.mut_at(i) = copy.at(i) * scalar
        }
        copy
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

macro_rules! vec_impl_for_array {
    ($v:expr) => {
        impl Vector for [f64; $v] {

            type Scalar = f64;

            fn dimension(&self) -> usize{ $v }

            fn at(&self, index: usize) -> f64 {
                self[index]
            }

            fn mut_at(&mut self, index: usize) -> &mut f64 {
                &mut self[index]
            }
        }
    }
}

vec_impl_for_array! { 1 }
vec_impl_for_array! { 2 }
vec_impl_for_array! { 3 }
vec_impl_for_array! { 4 }
vec_impl_for_array! { 5 }
vec_impl_for_array! { 6 }
vec_impl_for_array! { 7 }
vec_impl_for_array! { 8 }
vec_impl_for_array! { 9 }
vec_impl_for_array! { 10 }
vec_impl_for_array! { 11 }
vec_impl_for_array! { 12 }
vec_impl_for_array! { 13 }
vec_impl_for_array! { 14 }
vec_impl_for_array! { 15 }
vec_impl_for_array! { 16 }
vec_impl_for_array! { 17 }
vec_impl_for_array! { 18 }
vec_impl_for_array! { 19 }
vec_impl_for_array! { 20 }
vec_impl_for_array! { 21 }
vec_impl_for_array! { 22 }
vec_impl_for_array! { 23 }
vec_impl_for_array! { 24 }
vec_impl_for_array! { 25 }
vec_impl_for_array! { 26 }
vec_impl_for_array! { 27 }
vec_impl_for_array! { 28 }
vec_impl_for_array! { 29 }
vec_impl_for_array! { 30 }
vec_impl_for_array! { 31 }
vec_impl_for_array! { 32 }

impl<V> Vector for (V::Scalar, V) where V: Vector{

    type Scalar = V::Scalar;

    fn dimension(&self) -> usize {
        self.1.dimension() + 1
    }

    fn at(&self, index: usize) -> V::Scalar {
        match index {
            0 => self.0,
            _ => self.1.at(index - 1)
        }
    }

    fn mut_at(&mut self, index: usize) -> &mut V::Scalar {
        match index {
            0 => &mut self.0,
            _ => self.1.mut_at(index - 1)
        }
    }

    fn dot(&self, other: &Self) -> V::Scalar {
        self.0 * other.0 + self.1.dot(&other.1)
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
