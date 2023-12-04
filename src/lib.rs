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

impl ops::Mul<Tensor> for Tensor {
    type Output = Self;

    fn mul(self, rhs: Tensor) -> Self::Output {
        assert!(
            self.shape.len() == 2 && rhs.shape.len() == 2,
            "only supporting 2d tensors for now"
        );

        let rows: Vec<_> = self.data.chunks(self.shape[1]).collect();

        let mut cols: Vec<Vec<i64>> = Vec::new();

        for (i, val) in rhs.data.iter().enumerate() {
            let remainder = i % rhs.shape[1];
            if let Some(col) = cols.get_mut(remainder) {
                col.push(*val);
            } else {
                cols.push(vec![*val]);
            }
        }

        let mut result = Vec::new();

        for row in rows {
            for col in &cols {
                result.push(zip(row, col).map(|(x1, x2)| x1 * x2).sum());
            }
        }

        Tensor {
            data: result,
            shape: vec![self.shape[0], rhs.shape[1]],
        }
    }
}

impl Tensor {
    pub fn from_scalar(data: i64) -> Self {
        Self {
            data: vec![data],
            shape: vec![],
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

    #[test]
    fn multiplication_2d() {
        let a = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
        let b = Tensor::new(vec![1, 2, 3, 1, 2, 3], vec![3, 2]);
        let result = a * b;

        assert_eq!(result.data, vec![13, 13, 31, 31]);
        assert_eq!(result.shape, vec![2, 2]);
    }
}
