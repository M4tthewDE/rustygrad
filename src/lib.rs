use std::{iter::zip, ops};

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<i64>,
    pub shape: Vec<usize>,
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let result: Vec<i64> = zip(self.data, rhs.data).map(|(x1, x2)| x1 + x2).collect();

        Tensor {
            data: result,
            shape: self.shape,
        }
    }
}

impl Tensor {
    pub fn from_scalar(data: i64) -> Self {
        Self {
            data: vec![data],
            shape: vec![1],
        }
    }

    pub fn new(data: Vec<i64>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn addition_1d() {
        let a = Tensor::from_scalar(2);
        let b = Tensor::from_scalar(3);
        let result = a + b;

        assert_eq!(result.data, vec![5]);
    }

    #[test]
    fn addition_2d() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![2, 2]);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![2, 2]);
        let result = a + b;

        assert_eq!(result.data, vec![6, 8, 10, 12]);
    }
}
