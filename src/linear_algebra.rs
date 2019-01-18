//! Defines linear algebra traits used for some model parameters

/// Vector whose dimension is known at runtime
///
/// Assumes the `Vector` is represented as a tuple of numbers representing its projection along
/// orthogonal base vectors
pub trait Vector: Clone {
    /// Retuns a new instance of Vector with all elements set to zero
    ///
    /// Not every possible implementation knows its dimension at compiletime, therefore a size hint
    /// is necessary to allocate the correct number of elements
    fn zero(dimension: usize) -> Self;
    /// Maximum allowed index for `at` and `at_mut`
    fn dimension(&self) -> usize;
    /// Length of projection along `i`-th base
    fn at(&self, i: usize) -> f64;
    /// Mutable access to length of projection along `i`-th base
    fn at_mut(&mut self, i: usize) -> &mut f64;
    /// Scalar product
    ///
    /// Default implementation using `at` and `dimension` is provided
    fn dot(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.dimension(), other.dimension());
        let mut result = 0.0;
        for i in 0..self.dimension() {
            result += self.at(i) * other.at(i)
        }
        result
    }
}

impl Vector for f64 {
    fn zero(dimension: usize) -> f64 {
        assert!(dimension == 1);
        0.0
    }

    fn dimension(&self) -> usize {
        1
    }
    fn at(&self, i: usize) -> f64 {
        assert!(i == 0);
        *self
    }
    fn at_mut(&mut self, i: usize) -> &mut f64 {
        assert!(i == 0);
        self
    }

    fn dot(&self, other: &Self) -> f64 {
        self * other
    }
}

impl Vector for Vec<f64> {

    fn zero(dimension: usize) -> Vec<f64> {
        vec![0.; dimension]
    }

    fn dimension(&self) -> usize {
        self.len()
    }

    fn at(&self, index: usize) -> f64 {
        self[index]
    }

    fn at_mut(&mut self, index: usize) -> &mut f64 {
        &mut self[index]
    }
}

macro_rules! vec_impl_for_array {
    ($v:expr) => {
        impl Vector for [f64; $v] {
            fn zero(_: usize) -> [f64; $v] {
                [0.0; $v]
            }

            fn dimension(&self) -> usize {
                $v
            }

            fn at(&self, index: usize) -> f64 {
                self[index]
            }

            fn at_mut(&mut self, index: usize) -> &mut f64 {
                &mut self[index]
            }
        }
    };
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

#[cfg(test)]
mod tests {

    #[test]
    fn dot() {
        use crate::linear_algebra::Vector;

        let a = [1.0, 2.0];
        let b = [3.0, 4.0];

        assert_eq!(11.0, a.dot(&b))
    }
}
