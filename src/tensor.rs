use std::iter::zip;
use std::{cmp, ops};

use itertools::{EitherOrBoth, Itertools};
use rand::{distributions::Uniform, prelude::Distribution};

use crate::op::UnrealizedOp;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub unrealized_op: UnrealizedOp,
    pub data: Option<Vec<f64>>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(unrealized_op: UnrealizedOp, data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "invalid shape for data length"
        );
        Tensor {
            unrealized_op,
            data: Some(data),
            shape,
        }
    }

    fn from_op(unrealized_op: UnrealizedOp, shape: Vec<usize>) -> Tensor {
        Tensor {
            unrealized_op,
            data: None,
            shape,
        }
    }

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(data, shape.clone()), shape)
    }

    pub fn from_vec_single_dim(data: Vec<f64>) -> Tensor {
        let data_len = data.len();
        Tensor::from_op(UnrealizedOp::Load(data, vec![data_len]), vec![data_len])
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![data], vec![1]), vec![1])
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        // feels like this should happen in an op
        let data = Uniform::new(-1.0, 1.0)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(UnrealizedOp::Load(data, shape.clone()), shape)
    }

    pub fn to_tch(&self) -> tch::Tensor {
        // TODO: this should realize, avoids a lot of typing that way
        tch::Tensor::from_slice(&self.data.clone().unwrap())
            .reshape(self.shape.iter().map(|&d| d as i64).collect::<Vec<i64>>())
    }

    pub fn realize(&self) -> Tensor {
        self.unrealized_op.realize()
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Add(Box::new(self.clone()), Box::new(Tensor::from_scalar(rhs))),
            self.shape,
        )
    }
}

impl ops::Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Add(Box::new(Tensor::from_scalar(self)), Box::new(rhs.clone())),
            rhs.shape,
        )
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Add(Box::new(self.clone()), Box::new(rhs.clone())),
            broadcast_shape(self.shape, rhs.shape),
        )
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Mul(Box::new(self.clone()), Box::new(Tensor::from_scalar(rhs))),
            self.shape,
        )
    }
}

impl ops::Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Mul(Box::new(Tensor::from_scalar(self)), Box::new(rhs.clone())),
            rhs.shape,
        )
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Mul(Box::new(self.clone()), Box::new(rhs.clone())),
            broadcast_shape(self.shape, rhs.shape),
        )
    }
}

fn broadcast_shape(mut lhs_shape: Vec<usize>, mut rhs_shape: Vec<usize>) -> Vec<usize> {
    let broadcastable = lhs_shape
        .iter()
        .rev()
        .zip_longest(rhs_shape.iter().rev())
        .all(|dim_pair| match dim_pair {
            EitherOrBoth::Both(&left, &right) => left == right || left == 1 || right == 1,
            _ => true,
        });
    assert!(
        broadcastable,
        "{:?} and {:?} aren't broadcastable",
        lhs_shape, rhs_shape
    );

    let max_len = lhs_shape.len().max(rhs_shape.len());
    while rhs_shape.len() < max_len {
        rhs_shape.insert(0, 1);
    }

    while lhs_shape.len() < max_len {
        lhs_shape.insert(0, 1);
    }

    let output_shape: Vec<usize> = zip(&lhs_shape, &rhs_shape)
        .map(|(d1, d2)| cmp::max(*d1, *d2))
        .collect();

    output_shape
}

#[cfg(test)]
mod tests {
    use crate::{tensor::Tensor, util};

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

    #[test]
    fn mul() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();
        let tch_input2 = input.realize().to_tch();

        let output = (input.clone() * input).realize();
        let tch_result = tch_input * tch_input2;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn mul_scalar() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = (input * 2.0).realize();
        let tch_result = tch_input * 2.0;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn basic() {
        let a = Tensor::from_scalar(1.0);
        let b = Tensor::from_scalar(2.0);

        let c = (a + b).realize();

        assert_eq!(c.data.unwrap(), vec![3.0]);
        assert_eq!(c.shape, vec![1]);
    }
}
