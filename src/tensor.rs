use std::ops;

use rand::{distributions::Uniform, prelude::Distribution};

use crate::op::UnrealizedOp;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub unrealized_op: UnrealizedOp,
    pub data: Option<Vec<f64>>,
    pub shape: Option<Vec<usize>>,
}

impl Tensor {
    pub fn new(unrealized_op: UnrealizedOp, data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor {
            unrealized_op,
            data: Some(data),
            shape: Some(shape),
        }
    }

    fn from_op(unrealized_op: UnrealizedOp) -> Tensor {
        Tensor {
            unrealized_op,
            data: None,
            shape: None,
        }
    }

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(data, shape))
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![data], vec![1]))
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        let data = Uniform::new(-1.0, 1.0)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(UnrealizedOp::Load(data, shape))
    }

    pub fn to_tch(self) -> tch::Tensor {
        tch::Tensor::from_slice(&self.data.unwrap()).reshape(
            self.shape
                .unwrap()
                .iter()
                .map(|&d| d as i64)
                .collect::<Vec<i64>>(),
        )
    }

    pub fn max(&self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Max(Box::new(self.clone())))
    }

    pub fn min(&self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Min(Box::new(self.clone())))
    }

    pub fn realize(&self) -> Tensor {
        self.unrealized_op.realize()
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Add(
            Box::new(self.clone()),
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
        Tensor::from_op(UnrealizedOp::Add(Box::new(self.clone()), Box::new(rhs)))
    }
}

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(
            Box::new(self.clone()),
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
        Tensor::from_op(UnrealizedOp::Sub(Box::new(self.clone()), Box::new(rhs)))
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(
            Box::new(self.clone()),
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
        Tensor::from_op(UnrealizedOp::Mul(Box::new(self.clone()), Box::new(rhs)))
    }
}

impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(
            Box::new(self.clone()),
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
        Tensor::from_op(UnrealizedOp::Div(Box::new(self.clone()), Box::new(rhs)))
    }
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
    fn sub() {
        let input1 = Tensor::from_vec(vec![0., 1., 2., 3.], vec![4]);
        let input2 = Tensor::from_vec(vec![0., 1., 2., 3.], vec![4]);

        let result = (input1 - input2).realize();

        assert_eq!(result.data.unwrap(), vec![0., 0., 0., 0.]);
        assert_eq!(result.shape.unwrap(), vec![4]);
    }

    #[test]
    #[should_panic]
    fn sub_broadcast_incompatible_dims_error() {
        let input1 = Tensor::from_vec(vec![0.; 40], vec![5, 2, 4, 1]);
        let input2 = Tensor::from_vec(vec![0.; 3], vec![3, 1, 1]);

        let _ = (input1 - input2).realize();
    }

    #[test]
    fn sub_broadcast_simple() {
        let input1 = Tensor::from_vec(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![2, 3, 2],
        );
        let input2 = Tensor::from_vec(vec![4., 3., 7., 8., 10., 2.], vec![3, 2]);

        let result = (input1 - input2).realize();
        assert_eq!(
            result.data.unwrap(),
            vec![-4., -2., -5., -5., -6., 3., 2., 4., 1., 1., 0., 9.]
        );
        assert_eq!(result.shape.unwrap(), vec![2, 3, 2]);
    }

    #[test]
    fn sub_broadcast_both_sides() {
        let input1 = Tensor::from_vec(vec![0., 1., 2., 3., 4., 5.], vec![2, 3, 1]);
        let input2 = Tensor::from_vec(vec![4., 3., 7., 8., 10., 2.], vec![3, 2]);

        let result = (input1 - input2).realize();
        assert_eq!(
            result.data.unwrap(),
            vec![-4., -3., -6., -7., -8., 0., -1., 0., -3., -4., -5., 3.]
        );
        assert_eq!(result.shape.unwrap(), vec![2, 3, 2]);
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
        assert_eq!(output.shape.unwrap(), tch_shape);
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
        assert_eq!(output.shape.unwrap(), tch_shape);
    }

    #[test]
    fn div() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();
        let tch_input2 = input.realize().to_tch();

        let output = (input.clone() / input).realize();
        let tch_result = tch_input / tch_input2;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape.unwrap(), tch_shape);
    }

    #[test]
    fn div_scalar() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = (input / 2.0).realize();
        let tch_result = tch_input / 2.0;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape.unwrap(), tch_shape);
    }

    #[test]
    fn max() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.max().realize();
        let tch_result = tch_input.max();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        dbg!(&tch_shape, &tch_output);
        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape.unwrap(), tch_shape);
    }

    #[test]
    fn min() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.min().realize();
        let tch_result = tch_input.min();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        dbg!(&tch_shape, &tch_output);
        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape.unwrap(), tch_shape);
    }
}
