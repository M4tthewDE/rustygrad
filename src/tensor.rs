use std::ops;

use crate::op::{Op, UnrealizedOp};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub unrealized_op: UnrealizedOp,
}

impl Tensor {
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor {
            unrealized_op: UnrealizedOp::new(Op::Load(data, shape)),
        }
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor {
            unrealized_op: UnrealizedOp::new(Op::Load(vec![data], vec![1])),
        }
    }

    pub fn realize(&self) -> Tensor {
        unimplemented!();
    }

    fn add_op(&mut self, op: Op) {
        self.unrealized_op.next = Box::new(Some(UnrealizedOp::new(op)));
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        let mut t = self.clone();
        t.add_op(Op::Add(Box::new(Tensor::from_scalar(rhs))));
        t
    }
}

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        let mut t = self.clone();
        t.add_op(Op::Sub(Box::new(Tensor::from_scalar(rhs))));
        t
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let mut t = self.clone();
        t.add_op(Op::Sub(Box::new(rhs)));
        t
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn unrealized_op_hard() {
        let a = Tensor::from_vec(vec![1.0], vec![1]);
        let a = a + 1.0;
        let b = Tensor::from_vec(vec![5.0], vec![1]);
        let b = b - 2.0;

        // TODO: this overrides the add from line 69!
        let c = a - b;
        dbg!(&c);
        let result = c.realize();
        dbg!(result);
        panic!();
    }
}
