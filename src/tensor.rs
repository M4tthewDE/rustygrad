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

    fn from_op(unrealized_op: UnrealizedOp) -> Tensor {
        let shape = match unrealized_op {
            UnrealizedOp::Add(ref lhs, ref rhs) => broadcast_shape(&lhs.shape, &rhs.shape),
            UnrealizedOp::Sub(ref lhs, ref rhs) => broadcast_shape(&lhs.shape, &rhs.shape),
            UnrealizedOp::Mul(ref lhs, ref rhs) => broadcast_shape(&lhs.shape, &rhs.shape),
            UnrealizedOp::Div(ref lhs, ref rhs) => broadcast_shape(&lhs.shape, &rhs.shape),
            UnrealizedOp::Max(_) => vec![],
            UnrealizedOp::Min(_) => vec![],
            UnrealizedOp::Load(_, ref shape) => shape.to_vec(),
        };
        Tensor {
            unrealized_op,
            data: None,
            shape,
        }
    }

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(data, shape))
    }

    pub fn from_vec_single_dim(data: Vec<f64>) -> Tensor {
        let data_len = data.len();
        Tensor::from_op(UnrealizedOp::Load(data, vec![data_len]))
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![data], vec![1]))
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        // feels like this should happen in an op
        let data = Uniform::new(-1.0, 1.0)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(UnrealizedOp::Load(data, shape))
    }

    pub fn to_tch(&self) -> tch::Tensor {
        let mut x = self.clone();
        x.realize();
        tch::Tensor::from_slice(&x.data.unwrap())
            .reshape(x.shape.iter().map(|&d| d as i64).collect::<Vec<i64>>())
    }

    pub fn max(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Max(Box::new(self)))
    }

    pub fn min(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Min(Box::new(self)))
    }

    pub fn realize(&mut self) {
        let (data, shape) = self.unrealized_op.realize();
        self.data = Some(data);
        self.shape = shape;
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Add(
            Box::new(self),
            Box::new(Tensor::from_scalar(rhs)),
        ))
    }
}

impl ops::Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Add(
            Box::new(Tensor::from_scalar(self)),
            Box::new(rhs),
        ))
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Add(Box::new(self), Box::new(rhs)))
    }
}

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(
            Box::new(self),
            Box::new(Tensor::from_scalar(rhs)),
        ))
    }
}

impl ops::Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(
            Box::new(Tensor::from_scalar(self)),
            Box::new(rhs),
        ))
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(Box::new(self), Box::new(rhs)))
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(
            Box::new(self),
            Box::new(Tensor::from_scalar(rhs)),
        ))
    }
}

impl ops::Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(
            Box::new(Tensor::from_scalar(self)),
            Box::new(rhs),
        ))
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(Box::new(self), Box::new(rhs)))
    }
}

impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(
            Box::new(self),
            Box::new(Tensor::from_scalar(rhs)),
        ))
    }
}

impl ops::Div<Tensor> for f64 {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(
            Box::new(Tensor::from_scalar(self)),
            Box::new(rhs),
        ))
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(Box::new(self), Box::new(rhs)))
    }
}

fn broadcast_shape(lhs_shape: &[usize], rhs_shape: &[usize]) -> Vec<usize> {
    let mut lhs_shape = lhs_shape.to_owned();
    let mut rhs_shape = rhs_shape.to_owned();
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

    let output_shape: Vec<usize> = zip(lhs_shape, rhs_shape)
        .map(|(d1, d2)| cmp::max(d1, d2))
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
        let mut result = a + b;
        result.realize();

        assert_eq!(result.data.unwrap(), vec![5.0]);
    }

    #[test]
    fn addition_vector() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let b = Tensor::from_vec(vec![8.0, 7.0], vec![2]);
        let mut result = a + b;
        result.realize();

        assert_eq!(result.data.unwrap(), vec![10.0, 10.0]);
    }

    #[test]
    fn addition_f64() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let mut result = a + 5.0;
        result.realize();

        assert_eq!(result.data.unwrap(), vec![7.0, 8.0]);
    }

    #[test]
    fn addition_f64_left_side() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let mut result = 5.0 + a;
        result.realize();

        assert_eq!(result.data.unwrap(), vec![7.0, 8.0]);
    }

    #[test]
    fn div() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();
        let tch_input2 = input.to_tch();

        let mut output = input.clone() / input;
        output.realize();
        let tch_result = tch_input / tch_input2;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn div_scalar() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();

        let mut output = input / 2.0;
        output.realize();

        let tch_result = tch_input / 2.0;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }
    #[test]
    fn max() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.max();
        output.realize();
        let tch_result = tch_input.max();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn min() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.min();
        output.realize();
        let tch_result = tch_input.min();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }
}
