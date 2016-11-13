//! Helper Module to treat Vec and fixed sized arrays as generic in some contexts
use linear_algebra::Vector;

/// This trait is used to make up for the lack of generics over array lengths
pub trait Array {
    /// Element type of the array
    type Element;

    /// Corresponding Vector type with same dimension
    type Vector: Vector;

    /// Number of elements within the array
    fn length(&self) -> usize;
    /// Access element by immutable reference
    fn at_ref(&self, index: usize) -> &Self::Element;
    /// Access element by mutable reference
    fn at_mut(&mut self, index: usize) -> &mut Self::Element;
}

// impl<T> Array for Vec<T> {
//     type Element = T;
//     type Vector = Vec<f64>;

//     fn length(&self) -> usize {
//         self.length()
//     }

//     fn at_ref(&self, index: usize) -> &T {
//         &self[index]
//     }

//     fn at_mut(&mut self, index: usize) -> &mut T {
//         &mut self[index]
//     }
// }

macro_rules! array_impl_for {
    ($v:expr) => {
        impl<T> Array for [T; $v] {
            type Element = T;
            type Vector = [f64; $v];

            fn length(&self) -> usize{ $v }

            fn at_ref(&self, index: usize) -> &T {
                &self[index]
            }

            fn at_mut(&mut self, index: usize) -> &mut T {
                &mut self[index]
            }
        }
    }
}

array_impl_for! { 1 }
array_impl_for! { 2 }
array_impl_for! { 3 }
array_impl_for! { 4 }
array_impl_for! { 5 }
array_impl_for! { 6 }
array_impl_for! { 7 }
array_impl_for! { 8 }
array_impl_for! { 9 }
array_impl_for! { 10 }
array_impl_for! { 11 }
array_impl_for! { 12 }
array_impl_for! { 13 }
array_impl_for! { 14 }
array_impl_for! { 15 }
array_impl_for! { 16 }
array_impl_for! { 17 }
array_impl_for! { 18 }
array_impl_for! { 19 }
array_impl_for! { 20 }
array_impl_for! { 21 }
array_impl_for! { 22 }
array_impl_for! { 23 }
array_impl_for! { 24 }
array_impl_for! { 25 }
array_impl_for! { 26 }
array_impl_for! { 27 }
array_impl_for! { 28 }
array_impl_for! { 29 }
array_impl_for! { 30 }
array_impl_for! { 31 }
array_impl_for! { 32 }