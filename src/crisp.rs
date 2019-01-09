//! Contains implementations for crisp trait
use crate::{array::Array, Crisp};

impl Crisp for f64 {
    type Truth = bool;

    fn crisp(&self) -> bool {
        *self > 0.5
    }
}

impl<A: Array<Element = f64>> Crisp for A {
    type Truth = usize;

    fn crisp(&self) -> usize {
        // return index of the maximum
        (0..self.length())
            .map(|index| self.at_ref(index))
            .enumerate()
            .fold((0, 0.0), |m, (i, &v)| if v > m.1 { (i, v) } else { m })
            .0
    }
}
