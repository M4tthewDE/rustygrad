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
        tch::Tensor::from_slice(&self.data.clone().unwrap())
            .reshape(self.shape.iter().map(|&d| d as i64).collect::<Vec<i64>>())
    }

    pub fn max(&self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Max(Box::new(self.clone())), vec![1])
    }

    pub fn min(&self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Min(Box::new(self.clone())), vec![1])
    }

    pub fn avg_pool_2d(&self, kernel: (usize, usize), stride: Option<usize>) -> Tensor {
        let stride = stride.unwrap_or(1);
        Tensor::from_op(
            UnrealizedOp::Pool2D(Box::new(self.clone()), kernel, stride, 0.0, |a, b| a + b),
            vec![
                self.shape[0],
                self.shape[1],
                ((self.shape[2] - kernel.0) / stride) + 1,
                ((self.shape[3] - kernel.1) / stride) + 1,
            ],
        ) / (kernel.0 * kernel.1) as f64
    }

    pub fn max_pool_2d(&self, kernel: (usize, usize), stride: Option<usize>) -> Tensor {
        let stride = stride.unwrap_or(1);
        Tensor::from_op(
            UnrealizedOp::Pool2D(Box::new(self.clone()), kernel, stride, f64::MIN, |a, b| {
                a.max(b)
            }),
            vec![
                self.shape[0],
                self.shape[1],
                ((self.shape[2] - kernel.0) / stride) + 1,
                ((self.shape[3] - kernel.1) / stride) + 1,
            ],
        )
    }

    pub fn reduce_sum(&self, dims: Option<&Vec<usize>>, keepdim: bool) -> Tensor {
        let new_shape = if let Some(dims) = dims {
            if keepdim {
                self.shape
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                    .collect()
            } else {
                let mut reduced_shape = self.shape.clone();
                for (i, dim) in dims.iter().enumerate() {
                    reduced_shape.remove(*dim - i);
                }

                reduced_shape
            }
        } else {
            self.shape.clone()
        };
        Tensor::from_op(
            UnrealizedOp::Sum(Box::new(self.clone()), dims.cloned(), keepdim),
            new_shape,
        )
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Reshape(Box::new(self.clone()), shape.clone()),
            shape,
        )
    }

    pub fn permute(&self, dims: Vec<usize>) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Permute(Box::new(self.clone()), dims.clone()),
            dims.iter().map(|&d| self.shape[d]).collect(),
        )
    }

    pub fn reduce_mean(
        &self,
        dims: Option<&Vec<usize>>,
        keepdim: bool,
        correction: Option<f64>,
    ) -> Tensor {
        let divisor = dims
            .map(|dims| {
                dims.iter()
                    .map(|&dim| self.shape[dim] as f64)
                    .product::<f64>()
            })
            .unwrap_or(self.shape.iter().product::<usize>() as f64);

        self.reduce_sum(dims, keepdim) / (divisor - correction.unwrap_or(0.0)).max(1.0)
    }

    pub fn variance(&self, dims: Option<&Vec<usize>>, correction: Option<f64>) -> Tensor {
        let mean = self.reduce_mean(dims, true, None);
        let diff = self.clone() - mean;
        (diff.clone() * diff).reduce_mean(dims, false, correction.or(Some(1.0)))
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Sqrt(Box::new(self.clone())),
            self.shape.clone(),
        )
    }

    pub fn rsqrt(&self) -> Tensor {
        1.0 / self.sqrt()
    }

    pub fn log(&self) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Log(Box::new(self.clone())),
            self.shape.clone(),
        )
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Sigmoid(Box::new(self.clone())),
            self.shape.clone(),
        )
    }

    pub fn swish(&self) -> Tensor {
        self.clone() * self.sigmoid()
    }

    pub fn relu(&self) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Relu(Box::new(self.clone())),
            self.shape.clone(),
        )
    }

    // does not require tensor to be realized!
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    pub fn realize(&self) -> Tensor {
        self.unrealized_op.realize()
    }
}

impl ops::Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        self * -1.0
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

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Sub(Box::new(self.clone()), Box::new(Tensor::from_scalar(rhs))),
            self.shape,
        )
    }
}

impl ops::Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Sub(Box::new(Tensor::from_scalar(self)), Box::new(rhs.clone())),
            rhs.shape,
        )
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Sub(Box::new(self.clone()), Box::new(rhs.clone())),
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

impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Div(Box::new(self.clone()), Box::new(Tensor::from_scalar(rhs))),
            self.shape,
        )
    }
}

impl ops::Div<Tensor> for f64 {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Div(Box::new(Tensor::from_scalar(self)), Box::new(rhs.clone())),
            rhs.shape,
        )
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(
            UnrealizedOp::Div(Box::new(self.clone()), Box::new(rhs.clone())),
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
    fn sub() {
        let input1 = Tensor::from_vec(vec![0., 1., 2., 3.], vec![4]);
        let input2 = Tensor::from_vec(vec![0., 1., 2., 3.], vec![4]);

        let result = (input1 - input2).realize();

        assert_eq!(result.data.unwrap(), vec![0., 0., 0., 0.]);
        assert_eq!(result.shape, vec![4]);
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
        assert_eq!(result.shape, vec![2, 3, 2]);
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
        assert_eq!(result.shape, vec![2, 3, 2]);
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
    fn div() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();
        let tch_input2 = input.realize().to_tch();

        let output = (input.clone() / input).realize();
        let tch_result = tch_input / tch_input2;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
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
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn max() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.max().realize();
        let tch_result = tch_input.max();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn min() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.min().realize();
        let tch_result = tch_input.min();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn reduce_sum() {
        let input = Tensor::rand(vec![2, 4, 3, 3]);
        let tch_input = input.realize().to_tch();
        let sum = input.reduce_sum(Some(&vec![0, 2, 3]), false).realize();
        let tch_sum = tch_input.sum_dim_intlist(vec![0, 2, 3], false, None);
        let tch_shape = util::tch_shape(&tch_sum);
        let tch_output = util::tch_data(&tch_sum);
        assert_eq!(sum.shape, tch_shape);
        util::assert_aprox_eq_vec(sum.data.unwrap(), tch_output, 1e-6);
    }

    #[test]
    fn avg_pool_2d() {
        let input = Tensor::rand(vec![1, 1, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.avg_pool_2d((2, 2), None).realize();
        let tch_result = tch_input.avg_pool2d(vec![2, 2], 1, 0, false, true, None);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn max_pool_2d() {
        let input = Tensor::rand(vec![1, 1, 10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.max_pool_2d((2, 2), None).realize();
        let tch_result = tch_input.max_pool2d(vec![2, 2], 1, 0, 1, false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn reshape() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.reshape(vec![25, 2, 2]).realize();
        let tch_result = tch_input.reshape(vec![25, 2, 2]);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn permute() {
        let input = Tensor::rand(vec![5, 15]);
        let tch_input = input.realize().to_tch();

        let output = input.permute(vec![1, 0]).realize();
        let tch_result = tch_input.permute(vec![1, 0]);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn mean_4d_over_3_axis() {
        let input = Tensor::rand(vec![2, 4, 3, 3]);
        let tch_input = input.realize().to_tch();

        let mean = input
            .reduce_mean(Some(&vec![0, 2, 3]), false, None)
            .realize();
        let tch_mean = tch_input.mean_dim(vec![0, 2, 3], false, None);
        let tch_shape = util::tch_shape(&tch_mean);
        let tch_output = util::tch_data(&tch_mean);
        assert_eq!(mean.shape, tch_shape);
        util::assert_aprox_eq_vec(mean.data.unwrap(), tch_output, 1e-6);
    }

    #[test]
    fn variance_4d_over_3_axis() {
        let input = Tensor::rand(vec![2, 4, 3, 3]);
        let tch_input = input.realize().to_tch();

        let var = input.variance(Some(&vec![0, 2, 3]), None).realize();
        let tch_var = tch_input.var_dim(vec![0, 2, 3], true, false);
        let tch_shape = util::tch_shape(&tch_var);
        let tch_output = util::tch_data(&tch_var);
        assert_eq!(var.shape, tch_shape);
        util::assert_aprox_eq_vec(var.data.unwrap(), tch_output, 1e-6);
    }

    #[test]
    fn sqrt() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.sqrt().realize();
        let tch_result = tch_input.sqrt();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn rsqrt() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.rsqrt().realize();
        let tch_result = tch_input.rsqrt();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn log() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.log().realize();
        let tch_result = tch_input.log2();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn swish() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.swish().realize();
        let tch_result = tch_input.silu();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn relu() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let output = input.relu().realize();
        let tch_result = tch_input.relu();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn numel() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.realize().to_tch();

        let result = input.numel();
        let tch_result = tch_input.numel();
        assert_eq!(result, tch_result);
    }
}
