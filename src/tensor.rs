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

    fn from_op(unrealized_op: UnrealizedOp) -> Tensor {
        Tensor {
            unrealized_op,
            data: None,
            shape: Vec::new(),
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

    pub fn from_image(img: DynamicImage) -> Tensor {
        let shape = vec![img.width() as usize, img.height() as usize, 3];
        let data: Vec<f64> = img
            .to_rgb8()
            .pixels()
            .flat_map(|p| p.0.map(|x| x as f64))
            .collect_vec();

        Tensor::from_op(UnrealizedOp::Load(data, shape))
    }

    pub fn zeros(size: usize) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![0.0; size], vec![size]))
    }

    pub fn ones(size: usize) -> Tensor {
        Tensor::from_op(UnrealizedOp::Load(vec![1.0; size], vec![size]))
    }

    pub fn glorot_uniform(shape: Vec<usize>) -> Tensor {
        let limit = (6.0 / (shape[0] + shape[1..].iter().product::<usize>()) as f64).sqrt();
        let data = Uniform::new(0.0, limit)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(UnrealizedOp::Load(data, shape))
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
        Tensor::from_op(UnrealizedOp::Max(Box::new(self.unrealized_op)))
    }

    pub fn min(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Min(Box::new(self.unrealized_op)))
    }

    pub fn sqrt(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Sqrt(Box::new(self.unrealized_op)))
    }

    pub fn rsqrt(self) -> Tensor {
        1.0 / self.sqrt()
    }

    pub fn log(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Log(Box::new(self.unrealized_op)))
    }

    pub fn sigmoid(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Sigmoid(Box::new(self.unrealized_op)))
    }

    pub fn swish(self) -> Tensor {
        self.clone() * self.sigmoid()
    }

    pub fn relu(self) -> Tensor {
        Tensor::from_op(UnrealizedOp::Relu(Box::new(self.unrealized_op)))
    }

    pub fn reduce_sum(self, dims: Option<&Vec<usize>>, keepdim: bool) -> Tensor {
        Tensor::from_op(UnrealizedOp::Sum(
            Box::new(self.unrealized_op),
            dims.cloned(),
            keepdim,
        ))
    }

    pub fn avg_pool_2d(self, kernel: (usize, usize), stride: Option<usize>) -> Tensor {
        let stride = stride.unwrap_or(1);
        Tensor::from_op(UnrealizedOp::Pool2D(
            Box::new(self.unrealized_op),
            kernel,
            stride,
            0.0,
            |a, b| a + b,
        )) / (kernel.0 * kernel.1) as f64
    }

    pub fn max_pool_2d(self, kernel: (usize, usize), stride: Option<usize>) -> Tensor {
        let stride = stride.unwrap_or(1);
        Tensor::from_op(UnrealizedOp::Pool2D(
            Box::new(self.unrealized_op),
            kernel,
            stride,
            f64::MIN,
            |a, b| a.max(b),
        ))
    }

    pub fn pad_2d(self, value: f64, padding: [usize; 4]) -> Tensor {
        Tensor::from_op(UnrealizedOp::Pad2D(
            Box::new(self.unrealized_op),
            value,
            padding,
        ))
    }

    pub fn conv2d(
        self,
        kernel: Tensor,
        bias: Option<Tensor>,
        padding: Option<[usize; 4]>,
        strides: Option<(usize, usize)>,
        groups: Option<usize>,
    ) -> Tensor {
        let x = if let Some(padding) = padding {
            self.pad_2d(0.0, padding)
        } else {
            self
        };
        let res = Tensor::from_op(UnrealizedOp::Conv2D(
            Box::new(x.unrealized_op),
            Box::new(kernel.unrealized_op),
            strides,
            groups,
        ));

        if let Some(bias) = bias {
            res + bias
        } else {
            res
        }
    }

    pub fn reshape(self, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(UnrealizedOp::Reshape(Box::new(self.unrealized_op), shape))
    }

    pub fn permute(self, dims: Vec<usize>) -> Tensor {
        Tensor::from_op(UnrealizedOp::Permute(
            Box::new(self.unrealized_op),
            dims.clone(),
        ))
    }

    pub fn expand(self, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(UnrealizedOp::Expand(
            Box::new(self.unrealized_op),
            shape.clone(),
        ))
    }

    pub fn matmul(self, rhs: Tensor) -> Tensor {
        Tensor::from_op(UnrealizedOp::MatMul(
            Box::new(self.unrealized_op),
            Box::new(rhs.unrealized_op),
        ))
    }

    pub fn batchnorm(
        self,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        mean: Tensor,
        invstd: Tensor,
    ) -> Tensor {
        let mean_shape = mean.shape.clone();
        let x = self.clone() - mean.reshape(vec![1, mean_shape[0], 1, 1]);
        let x = if let Some(weight) = weight {
            let shape = weight.shape.clone();
            x * weight.reshape(vec![1, shape[0], 1, 1])
        } else {
            x
        };

        let ret = if invstd.shape.len() == 1 {
            let shape = invstd.shape.clone();
            x * (invstd.reshape(vec![1, shape[1], 1, 1]))
        } else {
            x * invstd.clone()
        };

        if let Some(bias) = bias {
            let shape = bias.shape.clone();
            ret + bias.reshape(vec![1, shape[0], 1, 1])
        } else {
            ret
        }
    }

    pub fn linear(self, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
        match bias {
            Some(bias) => self.matmul(weight.clone()) + bias.clone(),
            None => self.matmul(weight.clone()),
        }
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
            Box::new(self.unrealized_op),
            Box::new(UnrealizedOp::Load(vec![rhs], vec![1])),
        ))
    }
}

impl ops::Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Add(
            Box::new(UnrealizedOp::Load(vec![self], vec![1])),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Add(
            Box::new(self.unrealized_op),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(
            Box::new(self.unrealized_op),
            Box::new(UnrealizedOp::Load(vec![rhs], vec![1])),
        ))
    }
}

impl ops::Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(
            Box::new(UnrealizedOp::Load(vec![self], vec![1])),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Sub(
            Box::new(self.unrealized_op),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(
            Box::new(self.unrealized_op),
            Box::new(UnrealizedOp::Load(vec![rhs], vec![1])),
        ))
    }
}

impl ops::Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(
            Box::new(UnrealizedOp::Load(vec![self], vec![1])),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Mul(
            Box::new(self.unrealized_op),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(
            Box::new(self.unrealized_op),
            Box::new(UnrealizedOp::Load(vec![rhs], vec![1])),
        ))
    }
}

impl ops::Div<Tensor> for f64 {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(
            Box::new(UnrealizedOp::Load(vec![self], vec![1])),
            Box::new(rhs.unrealized_op),
        ))
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        Tensor::from_op(UnrealizedOp::Div(
            Box::new(self.unrealized_op),
            Box::new(rhs.unrealized_op),
        ))
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

    #[test]
    fn sqrt() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.sqrt();
        output.realize();
        let tch_result = tch_input.sqrt();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn log() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.log();
        output.realize();
        let tch_result = tch_input.log2();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn swish() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.swish();
        output.realize();
        let tch_result = tch_input.silu();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn relu() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.relu();
        output.realize();
        let tch_result = tch_input.relu();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn reduce_sum() {
        let input = Tensor::rand(vec![2, 4, 3, 3]);
        let tch_input = input.to_tch();
        let mut sum = input.reduce_sum(Some(&vec![0, 2, 3]), false);
        sum.realize();
        let tch_sum = tch_input.sum_dim_intlist(vec![0, 2, 3], false, None);
        let tch_shape = util::tch_shape(&tch_sum);
        let tch_output = util::tch_data(&tch_sum);
        assert_eq!(sum.shape, tch_shape);
        util::assert_aprox_eq_vec(sum.data.unwrap(), tch_output, 1e-6);
    }

    #[test]
    fn avg_pool_2d() {
        let input = Tensor::rand(vec![1, 1, 10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.avg_pool_2d((2, 2), None);
        output.realize();
        let tch_result = tch_input.avg_pool2d(vec![2, 2], 1, 0, false, true, None);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn max_pool_2d() {
        let input = Tensor::rand(vec![1, 1, 10, 10]);
        let tch_input = input.to_tch();

        let mut output = input.max_pool_2d((2, 2), None);
        output.realize();
        let tch_result = tch_input.max_pool2d(vec![2, 2], 1, 0, 1, false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn conv2d_4d() {
        let input = Tensor::rand(vec![1, 3, 224, 224]);
        let tch_input = input.to_tch();
        let kernel = Tensor::rand(vec![32, 3, 3, 3]);
        let tch_kernel = kernel.to_tch();

        let mut output = input.conv2d(kernel, None, Some([1, 1, 1, 1]), Some((2, 2)), None);
        output.realize();
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
        let tch_input = input.to_tch();
        let mut output = input.pad_2d(0., [1, 2, 3, 4]);
        output.realize();
        let tch_output = tch_input.zero_pad2d(1, 2, 3, 4);
        let tch_shape = util::tch_shape(&tch_output);
        let tch_output = util::tch_data(&tch_output);

        assert_eq!(output.shape, tch_shape);
        assert_eq!(output.data.unwrap(), tch_output,);
    }

    #[test]
    fn permute() {
        let input = Tensor::rand(vec![5, 15]);
        let tch_input = input.to_tch();

        let mut output = input.permute(vec![1, 0]);
        output.realize();
        let tch_result = tch_input.permute(vec![1, 0]);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(output.data.unwrap(), tch_output);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn expand() {
        let input = Tensor::rand(vec![10, 1]);
        let tch_input = input.to_tch();

        let mut output = input.expand(vec![10, 10]);
        output.realize();
        let tch_result = tch_input.expand(vec![10, 10], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }

    #[test]
    fn matmul() {
        let input1 = Tensor::rand(vec![8, 10]);
        let input2 = Tensor::rand(vec![10, 12]);
        let tch_input1 = input1.to_tch();
        let tch_input2 = input2.to_tch();

        let mut output = input1.matmul(input2);
        output.realize();
        let tch_result = tch_input1.matmul(&tch_input2);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output.data.unwrap(), tch_output, 1e-6);
        assert_eq!(output.shape, tch_shape);
    }
}
