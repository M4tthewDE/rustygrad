use std::iter::zip;

use crate::{op::UnrealizedOp, tensor::Tensor};

impl UnrealizedOp {
    pub fn realize(&self) -> Tensor {
        match self {
            UnrealizedOp::Add(lhs, rhs) => {
                let lhs = lhs.realize();
                let rhs = rhs.realize();
                let data: Vec<f64> = zip(lhs.data.unwrap(), rhs.data.unwrap())
                    .map(|(l, r)| l + r)
                    .collect();

                Tensor::new(self.clone(), Some(data), Some(lhs.shape.unwrap().clone()))
            }
            UnrealizedOp::Sub(lhs, rhs) => {
                let lhs = lhs.realize();
                let rhs = rhs.realize();
                let data: Vec<f64> = zip(lhs.data.unwrap(), rhs.data.unwrap())
                    .map(|(l, r)| l - r)
                    .collect();

                Tensor::new(self.clone(), Some(data), Some(lhs.shape.unwrap().clone()))
            }
            UnrealizedOp::Load(data, shape) => {
                Tensor::new(self.clone(), Some(data.clone()), Some(shape.clone()))
            }
        }
    }
}
