use std::ops;

use crate::op::UnrealizedOp;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub unrealized_op: UnrealizedOp,
    pub data: Option<Vec<f64>>,
    pub shape: Option<Vec<usize>>,
}

impl Tensor {
    pub fn new(
        unrealized_op: UnrealizedOp,
        data: Option<Vec<f64>>,
        shape: Option<Vec<usize>>,
    ) -> Tensor {
        Tensor {
            unrealized_op,
            data,
            shape,
        }
    }
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::new(UnrealizedOp::Load(data, shape), None, None)
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor::new(UnrealizedOp::Load(vec![data], vec![1]), None, None)
    }

    pub fn realize(&self) -> Tensor {
        self.unrealized_op.realize()
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        Tensor::new(
            UnrealizedOp::Add(Box::new(self.clone()), Box::new(Tensor::from_scalar(rhs))),
            None,
            None,
        )
    }
}

impl ops::Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::new(
            UnrealizedOp::Add(Box::new(Tensor::from_scalar(self)), Box::new(rhs)),
            None,
            None,
        )
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::new(
            UnrealizedOp::Add(Box::new(self.clone()), Box::new(rhs)),
            None,
            None,
        )
    }
}

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        Tensor::new(
            UnrealizedOp::Sub(Box::new(self.clone()), Box::new(Tensor::from_scalar(rhs))),
            None,
            None,
        )
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::new(
            UnrealizedOp::Sub(Box::new(self.clone()), Box::new(rhs)),
            None,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;
    #[test]
    fn addition_scalar() {
        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let result = (a + b).realize();

        assert_eq!(result.data.unwrap(), vec![5.0]);
    }

    #[test]
    fn addition_vector() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let b = Tensor::from_vec(vec![8.0, 7.0], vec![2]);
        let result = (a + b).realize();

        assert_eq!(result.data.unwrap(), vec![10.0, 10.0]);
    }

    #[test]
    fn addition_f64() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let result = (a + 5.0).realize();

        assert_eq!(result.data.unwrap(), vec![7.0, 8.0]);
    }

    #[test]
    fn addition_f64_left_side() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let result = (5.0 + a).realize();

        assert_eq!(result.data.unwrap(), vec![7.0, 8.0]);
    }
}
