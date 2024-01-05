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
    fn unrealized_op_hard() {
        let a = Tensor::from_vec(vec![2.0], vec![1]);
        let a = a + 1.0;
        let b = Tensor::from_vec(vec![5.0], vec![1]);
        let b = b - 2.0;
        let c = a - b;

        let result = c.realize();
        assert_eq!(result.shape.unwrap(), vec![1]);
        assert_eq!(result.data.unwrap(), vec![0.0]);
    }
}
