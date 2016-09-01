use num::Num;
use num::Zero;
use num::One;

pub trait Vector : Clone{

    type Scalar : Num + Zero + One + Copy;
    fn dimension(&self) -> u32;
    fn at(&self, i : u32) -> Self::Scalar;
    fn mut_at<'a>(&'a mut self, i : u32) -> &'a mut Self::Scalar;
    // fn dot(&self, other : &Self) -> Self::Scalar;
    fn dot(&self, other : &Self) -> Self::Scalar{
        let mut result = Self::Scalar::zero();
        for i in 0..self.dimension(){
            result = result + self.at(i) * other.at(i)
        }
        result
    }
}

impl Vector for f64{
    type Scalar = f64;

    fn dimension(&self) -> u32{
        1
    }

    fn at(&self, _ : u32) -> f64{
        *self
    }

    fn mut_at<'a>(&'a mut self, _ : u32) -> &'a mut f64{
        self
    }

    fn dot(&self, other : &f64) -> f64{
        self * other
    }
}

impl Vector for [f64;2]{
    type Scalar = f64;

    fn dimension(&self) -> u32{
        2
    }

    fn at(&self, index : u32) -> f64{
        self[index as usize]
    }

    fn mut_at<'a>(&'a mut self, index : u32) -> &'a mut f64{
        & mut self[index as usize]
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn dot() {

        use linear_algebra::Vector;

        let a  = [1.0, 2.0];
        let b  = [3.0, 4.0];

        assert_eq!(11.0, a.dot(&b))
    }
}