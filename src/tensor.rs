use std::iter::zip;
use std::{cmp, ops};

use image::DynamicImage;
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

    pub fn glorot_uniform(shape: Vec<usize>) -> Tensor {
        let limit = (6.0 / (shape[0] + shape[1..].iter().product::<usize>()) as f64).sqrt();
        let data = Uniform::new(0.0, limit)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(UnrealizedOp::Load(data, shape.clone()), shape)
    }

    pub fn zeros(size: usize) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![0.0; size], vec![size]), vec![size])
    }

    pub fn ones(size: usize) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![1.0; size], vec![size]), vec![size])
    }

    pub fn from_image(img: DynamicImage) -> Tensor {
        let shape = vec![img.width() as usize, img.height() as usize, 3];
        let data: Vec<f64> = img
            .to_rgb8()
            .pixels()
            .flat_map(|p| p.0.map(|x| x as f64))
            .collect_vec();

        Tensor::from_op(UnrealizedOp::Load(data, shape.clone()), shape)
    }

    pub fn to_tch(&self) -> tch::Tensor {
        // TODO: this should realize, avoids a lot of typing that way
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

    pub fn matmul(&self, rhs: Tensor) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::MatMul(Box::new(self.clone()), Box::new(rhs.clone())),
            vec![self.shape[0], rhs.shape[1]],
        )
    }

    pub fn expand(&self, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(
            UnrealizedOp::Expand(Box::new(self.clone()), shape.clone()),
            shape,
        )
    }

    pub fn realize(&self) -> Tensor {
        self.unrealized_op.realize()
    }

    pub fn batchnorm(
        &self,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        mean: &Tensor,
        invstd: &Tensor,
    ) -> Tensor {
        let x = self.clone() - mean.reshape(vec![1, mean.shape[0], 1, 1]);
        let x = if let Some(weight) = weight {
            x * weight.reshape(vec![1, weight.shape[0], 1, 1])
        } else {
            x
        };

        let ret = if invstd.shape.len() == 1 {
            x * (invstd.reshape(vec![1, invstd.shape[1], 1, 1]))
        } else {
            x * invstd.clone()
        };

        if let Some(bias) = bias {
            ret + bias.reshape(vec![1, bias.shape[0], 1, 1])
        } else {
            ret
        }
    }

    pub fn index_4d_to_1d(self, n: usize, c: usize, h: usize, w: usize) -> usize {
        let (height, width) = (self.shape[2], self.shape[3]);
        let channels = self.shape[1];
        n * (channels * height * width) + c * (height * width) + h * width + w
    }

    pub fn pad_2d(self, value: f64, padding: [usize; 4]) -> Tensor {
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape[self.shape.len() - 2] += padding[2] + padding[3];
        new_shape[self.shape.len() - 1] += padding[0] + padding[1];
        Tensor::from_op(
            UnrealizedOp::Pad2D(Box::new(self.clone()), value, padding),
            new_shape,
        )
    }

    pub fn conv2d(
        self,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        padding: Option<[usize; 4]>,
        strides: Option<(usize, usize)>,
        groups: Option<usize>,
    ) -> Tensor {
        let x = if let Some(padding) = padding {
            self.clone().pad_2d(0.0, padding)
        } else {
            self.clone()
        };
        let res = Tensor::from_op(
            UnrealizedOp::Conv2D(
                Box::new(x.clone()),
                Box::new(kernel.clone()),
                strides,
                groups,
            ),
            vec![
                x.shape[0],
                x.shape[1],
                ((x.shape[2] - kernel.shape[1]) / strides.unwrap_or((1, 1)).0) + 1,
                ((x.shape[3] - kernel.shape[2]) / strides.unwrap_or((1, 1)).1) + 1,
            ],
        );

        if let Some(bias) = bias {
            res + bias.clone()
        } else {
            res
        }
    }

    pub fn linear(&self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        match bias {
            Some(bias) => self.matmul(weight.clone()) + bias.clone(),
            None => self.matmul(weight.clone()),
        }
    }

    pub fn sequential(&self, callables: &Vec<Box<dyn Callable>>) -> Tensor {
        let mut x = self.clone();
        for callable in callables {
            x = callable.call(x);
        }

        x
    }
}

pub trait Callable {
    fn call(&self, x: Tensor) -> Tensor;
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
    fn matmul() {
        let input1 = Tensor::rand(vec![8, 10]);
        let input2 = Tensor::rand(vec![10, 12]);
        let tch_input1 = input1.realize().to_tch();
        let tch_input2 = input2.realize().to_tch();

        let output = input1.matmul(input2).realize();
        let tch_result = tch_input1.matmul(&tch_input2);
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

    #[test]
    fn expand() {
        let input = Tensor::rand(vec![10, 1]);
        let tch_input = input.realize().to_tch();

        let output = input.expand(vec![10, 10]).realize();
        let tch_result = tch_input.expand(vec![10, 10], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn conv2d_4d() {
        let input = Tensor::rand(vec![1, 3, 224, 224]);
        let tch_input = input.realize().to_tch();
        let kernel = Tensor::rand(vec![32, 3, 3, 3]);
        let tch_kernel = kernel.realize().to_tch();
        let output = input
            .conv2d(&kernel, None, Some([1, 1, 1, 1]), Some((2, 2)), None)
            .realize();
        let tch_output = tch_input.conv2d(
            &tch_kernel,
            None::<tch::Tensor>,
            vec![2, 2],
            vec![1, 1],
            1,
            1,
        );

        let tch_shape = util::tch_shape(&tch_output);
        let tch_output = util::tch_data(&tch_output);

        assert_eq!(output.shape, tch_shape);
        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
    }

    #[test]
    fn pad_2d_weird_padding() {
        let input = Tensor::rand(vec![1, 3, 16, 16]);
        let tch_input = input.realize().to_tch();
        let output = input.pad_2d(0., [1, 2, 3, 4]).realize();
        let tch_output = tch_input.zero_pad2d(1, 2, 3, 4);
        let tch_shape = util::tch_shape(&tch_output);
        let tch_output = util::tch_data(&tch_output);

        assert_eq!(output.shape, tch_shape);
        assert_eq!(output.data.unwrap(), tch_output,);
    }
}
