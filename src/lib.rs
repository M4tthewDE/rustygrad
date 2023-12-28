use std::{cmp, f64::consts::E, iter::zip, ops};

use image::DynamicImage;
use itertools::{EitherOrBoth, Itertools};
use rand::distributions::{Distribution, Uniform};

pub mod batch_norm;
pub mod efficientnet;
pub mod util;

pub fn broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    shape1
        .iter()
        .rev()
        .zip_longest(shape2.iter().rev())
        .all(|dim_pair| match dim_pair {
            EitherOrBoth::Both(&left, &right) => left == right || left == 1 || right == 1,
            _ => true,
        })
}

type BroadcastOp = fn(lhs: f64, rhs: f64) -> f64;

// https://pytorch.org/docs/stable/notes/broadcasting.html
fn broadcast_op(mut lhs: Tensor, mut rhs: Tensor, op: BroadcastOp) -> Tensor {
    assert!(broadcastable(&lhs.shape, &rhs.shape));

    let max_len = lhs.shape.len().max(rhs.shape.len());
    while rhs.shape.len() < max_len {
        rhs.shape.insert(0, 1);
    }

    while lhs.shape.len() < max_len {
        lhs.shape.insert(0, 1);
    }

    let output_shape: Vec<usize> = zip(&lhs.shape, &rhs.shape)
        .map(|(d1, d2)| cmp::max(*d1, *d2))
        .collect();

    let result_len = output_shape.iter().product();
    let mut result = Tensor::new(vec![0.; result_len], output_shape);

    let mut shape_pos = Vec::with_capacity(result.shape.len());
    for (i, elem) in result.data.iter_mut().enumerate() {
        shape_pos.clear();
        let mut offset = 0;
        for (j, _) in result.shape.iter().enumerate() {
            let count = result.shape[..=j].iter().product::<usize>();
            let index = (i - offset) / (result_len / count);
            shape_pos.push(index);
            offset += (result_len / count) * index;
        }

        *elem = (op)(
            lhs.point_from_shape_pos(&shape_pos, true),
            rhs.point_from_shape_pos(&shape_pos, true),
        );
    }

    result
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        broadcast_op(self, rhs, |x1, x2| x1 + x2)
    }
}

impl ops::Add<Tensor> for f64 {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        let result: Vec<f64> = rhs.data.iter().map(|x| x + self).collect();
        Tensor::new(result, rhs.shape)
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        let result: Vec<f64> = self.data.iter().map(|x| x + rhs).collect();
        Tensor::new(result, self.shape)
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        broadcast_op(self, rhs, |x1, x2| x1 - x2)
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        broadcast_op(self, rhs, |x1, x2| x1 * x2)
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let result: Vec<f64> = self.data.iter().map(|x| x * rhs).collect();
        Tensor::new(result, self.shape)
    }
}

impl ops::Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let result: Vec<f64> = rhs.data.iter().map(|x| self * x).collect();
        Tensor::new(result, rhs.shape)
    }
}

impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        let result: Vec<f64> = self.data.iter().map(|x| x / rhs).collect();
        Tensor::new(result, self.shape)
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        broadcast_op(self, rhs, |x1, x2| x1 / x2)
    }
}

impl Tensor {
    pub fn empty() -> Tensor {
        Tensor {
            data: vec![],
            shape: vec![],
        }
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        let data = Uniform::new(-1.0, 1.0)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::new(data, shape)
    }

    pub fn rand_with_range(shape: Vec<usize>, range: (f64, f64)) -> Tensor {
        let data = Uniform::new(range.0, range.1)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::new(data, shape)
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor::new(vec![data], vec![1])
    }

    pub fn from_vec(data: Vec<f64>) -> Tensor {
        let len = data.len();
        Tensor::new(data, vec![len])
    }

    pub fn zeros(size: usize) -> Tensor {
        Tensor::from_vec(vec![0.0; size])
    }

    pub fn ones(size: usize) -> Tensor {
        Tensor::from_vec(vec![1.0; size])
    }

    pub fn from_image(img: DynamicImage) -> Tensor {
        let shape = vec![img.width() as usize, img.height() as usize, 3];
        let data: Vec<f64> = img
            .to_rgb8()
            .pixels()
            .flat_map(|p| p.0.map(|x| x as f64))
            .collect_vec();

        Tensor::new(data, shape)
    }

    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        // empty tensors are an exception
        if !(data.is_empty() && shape.is_empty()) {
            assert_eq!(
                data.len(),
                shape.iter().product::<usize>(),
                "invalid shape for data length"
            );
        }

        Tensor { data, shape }
    }

    pub fn glorot_uniform(shape: Vec<usize>) -> Tensor {
        let limit = (6.0 / (shape[0] + shape[1..].iter().product::<usize>()) as f64).sqrt();
        let data = Uniform::new(0.0, limit)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::new(data, shape)
    }

    pub fn matmul(&self, rhs: Tensor) -> Tensor {
        assert!(
            self.shape.len() == 2 && rhs.shape.len() == 2,
            "only supporting 2d tensors for now"
        );

        let rows: Vec<_> = self.data.chunks(self.shape[1]).collect();

        let mut cols: Vec<Vec<f64>> = Vec::new();

        for (i, val) in rhs.data.iter().enumerate() {
            let remainder = i % rhs.shape[1];
            if let Some(col) = cols.get_mut(remainder) {
                col.push(*val);
            } else {
                cols.push(vec![*val]);
            }
        }

        let mut result = Vec::new();

        for row in rows {
            for col in &cols {
                result.push(zip(row, col).map(|(x1, x2)| x1 * x2).sum());
            }
        }

        Tensor::new(result, vec![self.shape[0], rhs.shape[1]])
    }

    fn point_from_shape_pos(&self, shape_pos: &[usize], broadcasting: bool) -> f64 {
        let mut index = 0;
        let mut divisor = 1;
        for (i, dim) in self.shape.iter().enumerate() {
            if broadcasting && *dim == 1 {
                continue;
            }
            assert!(shape_pos[i] < *dim);
            divisor *= dim;
            index += (self.data.len() / divisor) * shape_pos[i];
        }

        self.data[index]
    }

    pub fn transpose(self) -> Tensor {
        let mut cols: Vec<Vec<f64>> = vec![Vec::new(); self.shape[1]];

        for (i, val) in self.data.iter().enumerate() {
            let col_index = i % self.shape[1];
            cols[col_index].push(*val);
        }

        Tensor::new(cols.concat(), self.shape.iter().rev().copied().collect())
    }

    pub fn max(self) -> Tensor {
        match self
            .data
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
        {
            Some(val) => Tensor::new(vec![val], vec![1]),
            None => Tensor::empty(),
        }
    }

    fn index_4d_to_1d(&self, n: usize, c: usize, h: usize, w: usize) -> usize {
        let (height, width) = (self.shape[2], self.shape[3]);
        let channels = self.shape[1];
        n * (channels * height * width) + c * (height * width) + h * width + w
    }

    pub fn conv2d(
        mut self,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        padding: Option<&[usize; 4]>,
        strides: Option<(usize, usize)>,
        groups: Option<usize>,
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "only supporting 4d tensors");
        assert_eq!(kernel.shape.len(), 4, "only supporting 4d kernels");

        let groups = groups.unwrap_or(1);
        assert_eq!(
            self.shape[1] % groups,
            0,
            "input channels must be divisible by groups"
        );
        assert_eq!(
            kernel.shape[0] % groups,
            0,
            "output channels must be divisible by groups"
        );

        let n = self.shape[0];
        if let Some(padding) = padding {
            self = self.pad2d(0.0, padding);
        }

        let strides = strides.unwrap_or((1, 1));

        let (c_in, height, width) = (self.shape[1], self.shape[2], self.shape[3]);
        let (c_out, kernel_height, kernel_width) =
            (kernel.shape[0], kernel.shape[2], kernel.shape[3]);

        let output_height = ((height - kernel_height) / strides.0) + 1;
        let output_width = ((width - kernel_width) / strides.1) + 1;

        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;

        let mut output_data = Vec::new();
        for n_index in 0..n {
            for g in 0..groups {
                for c_out_index in (g * c_out_per_group)..((g + 1) * c_out_per_group) {
                    for i in 0..output_height {
                        for j in 0..output_width {
                            let mut value = 0.0;
                            for c_in_index in (g * c_in_per_group)..((g + 1) * c_in_per_group) {
                                for k_row in 0..kernel_height {
                                    for k_col in 0..kernel_width {
                                        let row = i * strides.0 + k_row;
                                        let col = j * strides.1 + k_col;
                                        value += self.data
                                            [self.index_4d_to_1d(n_index, c_in_index, row, col)]
                                            * kernel.data[kernel.index_4d_to_1d(
                                                c_out_index - (g * c_out_per_group), // adjust for group offset
                                                c_in_index - (g * c_in_per_group), // adjust for group offset
                                                k_row,
                                                k_col,
                                            )];
                                    }
                                }
                            }
                            output_data.push(value);
                        }
                    }
                }
            }
        }

        if let Some(bias) = bias {
            Tensor::new(output_data, vec![n, c_out, output_height, output_width]) + bias.to_owned()
        } else {
            Tensor::new(output_data, vec![n, c_out, output_height, output_width])
        }
    }

    pub fn max_pool2d(&self, kernel_size: usize, stride: Option<usize>) -> Tensor {
        assert_eq!(self.shape.len(), 4, "only supporting 4d tensors");
        let stride = stride.unwrap_or(1);

        let (batch, channels, height, width) =
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (kernel_height, kernel_width) = (kernel_size, kernel_size);

        let output_height = ((height - kernel_height) / stride) + 1;
        let output_width = ((width - kernel_width) / stride) + 1;

        let mut output_data = Vec::with_capacity(batch * channels * output_height * output_width);
        for n in 0..batch {
            for c in 0..channels {
                for i in 0..output_height {
                    for j in 0..output_width {
                        let mut max_val: f64 = f64::MIN;
                        for ki in 0..kernel_height {
                            for kj in 0..kernel_width {
                                let row = i * stride + ki;
                                let col = j * stride + kj;
                                let idx = n * (channels * height * width)
                                    + c * (height * width)
                                    + row * width
                                    + col;
                                max_val = max_val.max(self.data[idx]);
                            }
                        }
                        output_data.push(max_val);
                    }
                }
            }
        }

        Tensor::new(
            output_data,
            vec![batch, channels, output_height, output_width],
        )
    }

    pub fn avg_pool2d(&self, kernel_size: (usize, usize), stride: Option<usize>) -> Tensor {
        assert_eq!(self.shape.len(), 4, "only supporting 4d tensors");
        let stride = stride.unwrap_or(1);

        let (batch, channels, height, width) =
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (kernel_height, kernel_width) = (kernel_size.0, kernel_size.1);

        let output_height = ((height - kernel_height) / stride) + 1;
        let output_width = ((width - kernel_width) / stride) + 1;

        let mut output_data = Vec::with_capacity(batch * channels * output_height * output_width);
        for n in 0..batch {
            for c in 0..channels {
                for i in 0..output_height {
                    for j in 0..output_width {
                        let mut sum_val: f64 = 0.0;
                        let mut count: usize = 0;
                        for ki in 0..kernel_height {
                            for kj in 0..kernel_width {
                                let row = i * stride + ki;
                                let col = j * stride + kj;
                                let idx = n * (channels * height * width)
                                    + c * (height * width)
                                    + row * width
                                    + col;
                                sum_val += self.data[idx];
                                count += 1;
                            }
                        }
                        output_data.push(sum_val / count as f64);
                    }
                }
            }
        }

        Tensor::new(
            output_data,
            vec![batch, channels, output_height, output_width],
        )
    }

    pub fn pad2d(self, value: f64, padding: &[usize; 4]) -> Tensor {
        if self.shape.len() < 2 {
            panic!("Tensor must have at least 2 dimensions for 2D padding.");
        }

        let last_two_dims = self.shape.len() - 2;
        let mut new_shape: Vec<usize> = self.shape.clone();

        new_shape[last_two_dims + 1] += padding[2] + padding[3];
        new_shape[last_two_dims] += padding[0] + padding[1];

        let mut new_data = vec![value; new_shape.iter().product()];

        for i in 0..self.data.len() {
            let mut temp_index = i;
            let mut multi_dim_index = Vec::new();

            for &size in self.shape.iter().rev() {
                multi_dim_index.push(temp_index % size);
                temp_index /= size;
            }
            multi_dim_index.reverse();

            if multi_dim_index.len() >= 2 {
                multi_dim_index[last_two_dims] += padding[0];
                multi_dim_index[last_two_dims + 1] += padding[2];
            }

            let mut new_index = 0;
            let mut stride = 1;
            for (&size, &index) in new_shape.iter().rev().zip(multi_dim_index.iter().rev()) {
                new_index += index * stride;
                stride *= size;
            }

            new_data[new_index] = self.data[i];
        }

        Tensor::new(new_data, new_shape)
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
            .unwrap_or(self.data.len() as f64);

        self.reduce_sum(dims, keepdim) / (divisor - correction.unwrap_or(0.0)).max(1.0)
    }

    pub fn reduce_sum(&self, dims: Option<&Vec<usize>>, keepdim: bool) -> Tensor {
        let dims = match dims {
            Some(dims) => dims,
            None => return Tensor::from_scalar(self.data.iter().sum()),
        };

        let mut reduced_shape = self.shape.clone();
        for (i, dim) in dims.iter().enumerate() {
            reduced_shape.remove(*dim - i);
        }

        let mut result: Vec<f64> = vec![0.; reduced_shape.iter().product()];

        let mut shape_pos = Vec::with_capacity(self.shape.len() - dims.len());
        for (i, elem) in self.data.iter().enumerate() {
            shape_pos.clear();
            let mut offset = 0;
            for (j, _shape) in self.shape.iter().enumerate() {
                let count = self.shape[..=j].iter().product::<usize>();
                let index = (i - offset) / (self.data.len() / count);
                if !dims.contains(&j) {
                    shape_pos.push(index);
                }
                offset += (self.data.len() / count) * index;
            }

            let mut index = 0;
            for (j, dim) in reduced_shape.iter().rev().enumerate() {
                if j == reduced_shape.len() - 1 {
                    index += shape_pos[j];
                } else {
                    index += shape_pos[j] * dim;
                }
            }

            *result.get_mut(index).unwrap() += elem;
        }

        let new_shape = if keepdim {
            self.shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                .collect()
        } else {
            reduced_shape
        };

        Tensor::new(result, new_shape)
    }

    pub fn variance(&self, dims: Option<&Vec<usize>>, correction: Option<f64>) -> Tensor {
        let mean = self.reduce_mean(dims, true, None);
        let diff = self.clone() - mean;
        (diff.clone() * diff).reduce_mean(dims, false, correction.or(Some(1.0)))
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        Tensor::new(self.data.clone(), shape)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn size(&self, dim: Option<usize>) -> Vec<usize> {
        if let Some(dim) = dim {
            vec![self.shape[dim]]
        } else {
            self.shape.clone()
        }
    }

    pub fn sqrt(&self) -> Tensor {
        let result: Vec<f64> = self.data.iter().map(|x| x.sqrt()).collect();
        Tensor::new(result, self.shape.clone())
    }

    pub fn relu(&self) -> Tensor {
        let result: Vec<f64> = self
            .data
            .iter()
            .map(|x| if x < &0.0 { 0.0 } else { *x })
            .collect();
        Tensor::new(result, self.shape.clone())
    }

    pub fn linear(
        &self,
        in_features: &Tensor,
        out_features: &Tensor,
        bias: Option<bool>,
    ) -> Tensor {
        let in_size = in_features.shape[0];
        let out_size = out_features.shape[0];
        // Kaiming uniform
        let weight = Tensor::rand(vec![in_size, out_size]) * (6.0 / in_size as f64).sqrt();

        if bias.unwrap_or(true) {
            let bias_range = (1.0 / in_size as f64).sqrt();
            let bias = Tensor::rand_with_range(vec![out_size], (-bias_range, bias_range));
            self.matmul(weight) + bias
        } else {
            self.matmul(weight)
        }
    }

    pub fn flatten(&self, start_dim: Option<usize>) -> Tensor {
        let start_dim = start_dim.unwrap_or(0);
        let last_dim = self.shape[start_dim..].iter().product();
        let new_shape = [&self.shape[..start_dim], &[last_dim]].concat();

        self.reshape(new_shape)
    }

    pub fn permute(&self, dims: Vec<usize>) -> Tensor {
        let new_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();
        let mut new_data = vec![0.0; self.data.len()];

        // Permute the data
        for i in 0..self.data.len() {
            let mut temp_index = i;
            let mut multi_dim_index = Vec::new();
            for &size in self.shape.iter().rev() {
                multi_dim_index.push(temp_index % size);
                temp_index /= size;
            }
            multi_dim_index.reverse();

            let mut new_multi_dim_index: Vec<usize> = vec![0; dims.len()];
            for (new_i, &old_i) in dims.iter().enumerate() {
                new_multi_dim_index[new_i] = multi_dim_index[old_i];
            }

            let mut new_index = 0;
            let mut stride = 1;
            for (&size, &index) in new_shape.iter().rev().zip(new_multi_dim_index.iter().rev()) {
                new_index += index * stride;
                stride *= size;
            }

            // Place the original data into its new position
            new_data[new_index] = self.data[i];
        }

        Tensor::new(new_data, new_shape)
    }

    pub fn swish(&self) -> Tensor {
        Tensor::new(
            self.data
                .iter()
                .map(|x| x * (1.0 / (1.0 + E.powf(-x))))
                .collect(),
            self.shape.clone(),
        )
    }

    pub fn sequential(&self, callables: &Vec<Box<dyn Callable>>) -> Tensor {
        let mut x = self.clone();
        for callable in callables {
            x = callable.call(x);
        }

        x
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor::new(
            self.data.iter().map(|x| 1.0 / (1.0 + E.powf(-x))).collect(),
            self.shape.clone(),
        )
    }
}

pub trait Callable {
    fn call(&self, x: Tensor) -> Tensor;
}

#[cfg(test)]
mod tests {
    use std::f64::NAN;

    use crate::{batch_norm::INPUT, util::assert_aprox_eq_vec, Tensor};

    #[test]
    fn addition_scalar() {
        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let result = a + b;

        assert_eq!(result.data, vec![5.0]);
    }

    #[test]
    fn addition_vector() {
        let a = Tensor::from_vec(vec![2.0, 3.0]);
        let b = Tensor::from_vec(vec![8.0, 7.0]);
        let result = a + b;

        assert_eq!(result.data, vec![10.0, 10.0]);
    }

    #[test]
    fn addition_f64() {
        let a = Tensor::from_vec(vec![2.0, 3.0]);
        let result = a + 5.0;

        assert_eq!(result.data, vec![7.0, 8.0]);
    }

    #[test]
    fn addition_f64_left_side() {
        let a = Tensor::from_vec(vec![2.0, 3.0]);
        let result = 5.0 + a;

        assert_eq!(result.data, vec![7.0, 8.0]);
    }

    #[test]
    fn addition_matrix_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a + b;

        assert_eq!(result.data, vec![6.0, 8.0, 10., 12.0]);
    }

    #[test]
    fn matmul_matrix_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![3, 2]);
        let result = a.matmul(b);

        assert_eq!(result.data, vec![13.0, 13.0, 31.0, 31.0]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn element_wise_multiplication() {
        let a = Tensor::new(
            vec![
                -0.1258, 0.9666, 1.4808, -0.7937, 1.1734, -0.6563, 0.2612, 0.3245, -1.9038, 1.0037,
                -0.8889, -0.9841, -0.2029, -0.8373, 1.3127, -0.1301,
            ],
            vec![4, 4],
        );

        let result = a.clone() * a;
        assert_eq!(
            result.data,
            vec![
                0.01582564,
                0.93431556,
                2.1927686399999997,
                0.62995969,
                1.37686756,
                0.43072969,
                0.06822544,
                0.10530025000000001,
                3.6244544399999996,
                1.0074136900000001,
                0.7901432100000001,
                0.9684528099999999,
                0.041168409999999996,
                0.7010712900000001,
                1.7231812899999999,
                0.01692601
            ]
        );
        assert_eq!(result.shape, vec![4, 4]);
    }

    #[test]
    fn element_wise_mul_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = a * 2.0;

        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result.shape, vec![2, 3]);
    }

    #[test]
    fn element_wise_mul_scalar_left_side() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = 2.0 * a;

        assert_eq!(result.data, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(result.shape, vec![2, 3]);
    }

    #[test]
    fn transpose() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0], vec![2, 3]);
        let result = a.transpose();

        assert_eq!(result.data, vec![1.0, 3.0, 2.0, 4.0, 3.0, 5.0]);
        assert_eq!(result.shape, vec![3, 2]);
    }

    #[test]
    fn max() {
        let a = Tensor::empty();
        let result = a.max();

        assert!(result.data.is_empty());
        assert!(result.shape.is_empty());
    }

    #[test]
    fn conv2d() {
        // https://medium.com/apache-mxnet/convolutions-explained-with-ms-excel-465d6649831c
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![1, 1, 4, 4],
        );
        let kernel = Tensor::new(
            vec![
                1., 2., 3., //
                0., 1., 0., //
                2., 1., 2., //
            ],
            vec![1, 1, 3, 3],
        );
        let output = input.conv2d(&kernel, None, None, None, None);

        assert_eq!(output.data, vec![23., 22., 31., 26.]);
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn conv2d_weird() {
        let input = Tensor::rand(vec![1, 96, 112, 112]);
        let kernel = Tensor::rand(vec![96, 1, 3, 3]);
        let output = input.conv2d(&kernel, None, Some(&[0, 1, 0, 1]), Some((2, 2)), Some(96));

        assert_eq!(output.shape, vec![1, 96, 56, 56]);
    }

    #[test]
    fn conv2d_with_bias() {
        // https://medium.com/apache-mxnet/convolutions-explained-with-ms-excel-465d6649831c
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![1, 1, 4, 4],
        );
        let kernel = Tensor::new(
            vec![
                1., 2., 3., //
                0., 1., 0., //
                2., 1., 2., //
            ],
            vec![1, 1, 3, 3],
        );
        let output = input.conv2d(&kernel, Some(&Tensor::from_scalar(1.0)), None, None, None);

        assert_eq!(output.data, vec![24., 23., 32., 27.]);
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn conv2d_4d() {
        let input = Tensor::rand(vec![1, 3, 224, 224]);
        let kernel = Tensor::rand(vec![32, 3, 3, 3]);
        let output = input.conv2d(&kernel, None, Some(&[1, 1, 1, 1]), Some((2, 2)), None);

        assert_eq!(output.shape, vec![1, 32, 112, 112]);
    }

    #[test]
    fn conv2d_with_padding() {
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![1, 1, 4, 4],
        );
        let kernel = Tensor::new(
            vec![
                1., 2., 3., //
                0., 1., 0., //
                2., 1., 2., //
            ],
            vec![1, 1, 3, 3],
        );
        let output = input.conv2d(&kernel, None, Some(&[1, 1, 1, 1]), None, None);

        assert_eq!(
            output.data,
            vec![
                8., 14., 13., 8., //
                16., 23., 22., 10., //
                20., 31., 26., 17., //
                10., 9., 15., 10., //
            ]
        );
        assert_eq!(output.shape, vec![1, 1, 4, 4]);
    }

    #[test]
    fn conv2d_with_stride() {
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., 2., //
                1., 3., 3., 1., 2., //
                2., 1., 1., 3., 1., //
                3., 2., 3., 3., 2., //
                2., 3., 1., 2., 2., //
            ],
            vec![1, 1, 5, 5],
        );
        let kernel = Tensor::new(
            vec![
                1., 2., 3., //
                0., 1., 0., //
                2., 1., 2., //
            ],
            vec![1, 1, 3, 3],
        );
        let output = input.conv2d(&kernel, None, None, Some((2, 2)), None);

        assert_eq!(
            output.data,
            vec![
                23., 18., //
                18., 21., //
            ]
        );
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
    }

    #[test]
    fn pad_2d() {
        let input = Tensor::new(
            vec![
                1., 3., //
                2., 5., //
            ],
            vec![2, 2],
        );
        let output = input.pad2d(0., &[1, 1, 1, 1]);

        assert_eq!(
            output.data,
            vec![
                0., 0., 0., 0., //
                0., 1., 3., 0., //
                0., 2., 5., 0., //
                0., 0., 0., 0., //
            ]
        );
        assert_eq!(output.shape, vec![4, 4]);

        let input = Tensor::new(
            vec![
                2., 3., //
                2., 5., //
            ],
            vec![2, 2],
        );
        let output = input.pad2d(1., &[2, 2, 2, 2]);

        assert_eq!(
            output.data,
            vec![
                1., 1., 1., 1., 1., 1., //
                1., 1., 1., 1., 1., 1., //
                1., 1., 2., 3., 1., 1., //
                1., 1., 2., 5., 1., 1., //
                1., 1., 1., 1., 1., 1., //
                1., 1., 1., 1., 1., 1., //
            ]
        );
        assert_eq!(output.shape, vec![6, 6]);
    }

    #[test]
    fn mean() {
        let input = Tensor::from_vec(vec![0.2294, -0.5481, 1.3288]);

        let mean = input.reduce_mean(None, false, None);

        assert_eq!(mean.data, vec![0.3367]);
        assert_eq!(mean.shape, vec![1]);
    }

    #[test]
    fn mean_2d() {
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![4, 4],
        );

        let mean = input.reduce_mean(None, false, None);

        assert_eq!(mean.data, vec![2.0625]);
        assert_eq!(mean.shape, vec![1]);
    }

    #[test]
    fn mean_2d_keepdim() {
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![4, 4],
        );

        let mean = input.reduce_mean(Some(&vec![0]), true, None);

        assert_eq!(mean.data, vec![1.75, 2.25, 2.25, 2.0]);
        assert_eq!(mean.shape, vec![1, 4]);
    }

    #[test]
    fn mean_with_axis_first() {
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![4, 4],
        );

        let mean = input.reduce_mean(Some(&vec![0]), false, None);

        assert_eq!(mean.data, vec![1.75, 2.25, 2.25, 2.0]);
        assert_eq!(mean.shape, vec![4]);
    }

    #[test]
    fn mean_with_axis_last() {
        let input = Tensor::new(
            vec![
                1., 3., 2., 1., //
                1., 3., 3., 1., //
                2., 1., 1., 3., //
                3., 2., 3., 3., //
            ],
            vec![4, 4],
        );

        let mean = input.reduce_mean(Some(&vec![1]), false, None);

        assert_eq!(mean.data, vec![1.75, 2.0, 1.75, 2.75]);
        assert_eq!(mean.shape, vec![4]);
    }

    #[test]
    fn mean_4d_over_3_axis() {
        let input = Tensor::new(INPUT.to_vec(), vec![2, 4, 3, 3]);

        let mean = input.reduce_mean(Some(&vec![0, 2, 3]), false, None);
        assert_aprox_eq_vec(
            mean.data,
            vec![0.43395922, 0.45119032, 0.5364723, 0.49982092],
            1e-6,
        );
        assert_eq!(mean.shape, vec![4]);
    }

    #[test]
    fn reduce_sum() {
        let input = Tensor::new(
            vec![
                1., 1., 1., //
                1., 1., 1., //
            ],
            vec![2, 3],
        );

        let sum = input.reduce_sum(None, false);

        assert_eq!(sum.data, vec![6.0]);
        assert_eq!(sum.shape, vec![1]);
    }

    #[test]
    fn reduce_sum_axis_first() {
        let input = Tensor::new(
            vec![
                1., 2., 3., //
                4., 5., 6., //
            ],
            vec![2, 3],
        );

        let sum = input.reduce_sum(Some(&vec![0]), false);

        assert_eq!(sum.data, vec![5.0, 7.0, 9.0]);
        assert_eq!(sum.shape, vec![3]);
    }

    #[test]
    fn reduce_sum_axis_last() {
        let input = Tensor::new(
            vec![
                1., 2., 3., //
                4., 5., 6., //
            ],
            vec![2, 3],
        );

        let sum = input.reduce_sum(Some(&vec![1]), false);

        assert_eq!(sum.data, vec![6.0, 15.0]);
        assert_eq!(sum.shape, vec![2]);
    }

    #[test]
    fn reduce_sum_axis_first_3d() {
        let input = Tensor::new(
            vec![
                1., 2., //
                3., 4., //
                5., 6., //
                //
                7., 8., //
                9., 10., //
                11., 12., //
            ],
            vec![2, 3, 2],
        );

        let sum = input.reduce_sum(Some(&vec![0]), false);

        assert_eq!(sum.data, vec![8., 10., 12., 14., 16., 18.]);
        assert_eq!(sum.shape, vec![3, 2]);
    }

    #[test]
    fn reduce_sum_axis_middle_3d() {
        let input = Tensor::new(
            vec![
                1., 2., //
                3., 4., //
                5., 6., //
                //
                7., 8., //
                9., 10., //
                11., 12., //
            ],
            vec![2, 3, 2],
        );

        let sum = input.reduce_sum(Some(&vec![1]), false);

        assert_eq!(sum.data, vec![9., 12., 27., 30.0]);
        assert_eq!(sum.shape, vec![2, 2]);
    }

    #[test]
    fn reduce_sum_multiple_axis() {
        let input = Tensor::new(INPUT.to_vec(), vec![2, 4, 3, 3]);
        let sum = input.reduce_sum(Some(&vec![0, 2, 3]), false);

        assert_eq!(sum.shape, vec![4]);
        assert_aprox_eq_vec(sum.data, vec![7.811266, 8.121426, 9.656502, 8.996777], 1e-6);
    }

    #[test]
    fn reduce_sum_multiple_axis_keepdim() {
        let input = Tensor::new(INPUT.to_vec(), vec![2, 4, 3, 3]);
        let sum = input.reduce_sum(Some(&vec![0, 2, 3]), true);

        assert_eq!(sum.shape, vec![1, 4, 1, 1]);
        assert_aprox_eq_vec(sum.data, vec![7.811266, 8.121426, 9.656502, 8.996777], 1e-6);
    }

    #[test]
    fn reduce_sum_axis_last_3d() {
        let input = Tensor::new(
            vec![
                1., 2., //
                3., 4., //
                5., 6., //
                //
                7., 8., //
                9., 10., //
                11., 12., //
            ],
            vec![2, 3, 2],
        );

        let sum = input.reduce_sum(Some(&vec![2]), false);

        assert_eq!(sum.data, vec![3., 7., 11., 15., 19., 23.]);
        assert_eq!(sum.shape, vec![2, 3]);
    }

    #[test]
    fn reduce_sum_square_first() {
        let input = Tensor::new(
            vec![
                1., 3., 2., //
                1., 3., 3., //
                2., 1., 1., //
            ],
            vec![3, 3],
        );

        let mean = input.reduce_sum(Some(&vec![0]), false);

        assert_eq!(mean.data, vec![4., 7., 6.]);
        assert_eq!(mean.shape, vec![3]);
    }

    #[test]
    fn reduce_sum_square_last() {
        let input = Tensor::new(
            vec![
                1., 3., 2., //
                1., 3., 3., //
                2., 1., 1., //
            ],
            vec![3, 3],
        );

        let sum = input.reduce_sum(Some(&vec![1]), false);

        assert_eq!(sum.data, vec![6., 7., 4.]);
        assert_eq!(sum.shape, vec![3]);
    }

    #[test]
    fn div() {
        let input = Tensor::new(vec![4., 8., 10., 12.], vec![2, 2]);

        let result = input / 2.0;

        assert_eq!(result.data, vec![2., 4., 5., 6.]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn div_tensor() {
        let t1 = Tensor::new(vec![4., 8., 10., 12.], vec![2, 2]);
        let t2 = Tensor::new(vec![2., 2., 5., 4.], vec![2, 2]);

        let result = t1 / t2;

        assert_eq!(result.data, vec![2., 4., 2., 3.]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn reduce_sum_multiple_axis_3d() {
        let input = Tensor::new(
            vec![
                1., 2., //
                3., 4., //
                5., 6., //
                //
                7., 8., //
                9., 10., //
                11., 12., //
            ],
            vec![2, 3, 2],
        );

        let sum = input.reduce_sum(Some(&vec![0, 1]), false);

        assert_eq!(sum.data, vec![36., 42.]);
        assert_eq!(sum.shape, vec![2]);
    }

    #[test]
    fn reshape() {
        let input = Tensor::new(vec![0., 1., 2., 3.], vec![4]);

        let result = input.reshape(vec![2, 2]);

        assert_eq!(result.data, vec![0., 1., 2., 3.]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn variance() {
        let input = Tensor::new(
            vec![
                0.2035, 1.2959, 1.8101, -0.4644, //
                1.5027, -0.3270, 0.5905, 0.6538, //
                -1.5745, 1.3330, -0.5596, -0.6548, //
                0.1264, -0.5080, 1.6420, 0.1992, //
            ],
            vec![4, 4],
        );

        let result = input.variance(Some(&vec![1]), None);

        assert_aprox_eq_vec(
            result.data,
            vec![1.0630833092, 0.5590269933, 1.4893144158, 0.8257591867],
            1e-9,
        );
        assert_eq!(result.shape, vec![4]);
    }

    #[test]
    fn variance_4d_over_3_axis() {
        let input = Tensor::new(INPUT.to_vec(), vec![2, 4, 3, 3]);
        let var = input.variance(Some(&vec![0, 2, 3]), None);

        assert_aprox_eq_vec(
            var.data,
            vec![0.06047015, 0.1051994, 0.05764891, 0.08270448],
            1e-6,
        );
        assert_eq!(var.shape, vec![4]);
    }

    #[test]
    fn sub() {
        let input1 = Tensor::new(vec![0., 1., 2., 3.], vec![4]);
        let input2 = Tensor::new(vec![0., 1., 2., 3.], vec![4]);

        let result = input1 - input2;

        assert_eq!(result.data, vec![0., 0., 0., 0.]);
        assert_eq!(result.shape, vec![4]);
    }

    #[test]
    fn sub_no_broadcast() {
        let input1 = Tensor::new(vec![0.; 105], vec![5, 7, 3]);
        let input2 = Tensor::new(vec![0.; 105], vec![5, 7, 3]);

        let result = input1 - input2;
        assert_eq!(result.data, vec![0.; 105]);
        assert_eq!(result.shape, vec![5, 7, 3]);
    }

    #[test]
    #[should_panic]
    fn sub_broadcast_no_dim_error() {
        let input1 = Tensor::empty();
        let input2 = Tensor::new(vec![0.; 4], vec![2, 2]);

        let _ = input1 - input2;
    }

    #[test]
    #[should_panic]
    fn sub_broadcast_incompatible_dims_error() {
        let input1 = Tensor::new(vec![0.; 40], vec![5, 2, 4, 1]);
        let input2 = Tensor::new(vec![0.; 3], vec![3, 1, 1]);

        let _ = input1 - input2;
    }

    #[test]
    fn sub_broadcast_simple() {
        let input1 = Tensor::new(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![2, 3, 2],
        );
        let input2 = Tensor::new(vec![4., 3., 7., 8., 10., 2.], vec![3, 2]);

        let result = input1 - input2;
        assert_eq!(
            result.data,
            vec![-4., -2., -5., -5., -6., 3., 2., 4., 1., 1., 0., 9.]
        );
        assert_eq!(result.shape, vec![2, 3, 2]);
    }

    #[test]
    fn sub_broadcast_both_sides() {
        let input1 = Tensor::new(vec![0., 1., 2., 3., 4., 5.], vec![2, 3, 1]);
        let input2 = Tensor::new(vec![4., 3., 7., 8., 10., 2.], vec![3, 2]);

        let result = input1 - input2;
        assert_eq!(
            result.data,
            vec![-4., -3., -6., -7., -8., 0., -1., 0., -3., -4., -5., 3.]
        );
        assert_eq!(result.shape, vec![2, 3, 2]);
    }

    #[test]
    fn point_from_shape_pos() {
        let t = Tensor::new(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![2, 3, 2],
        );

        assert_eq!(t.point_from_shape_pos(&[1, 2, 0], false), 10.);
        assert_eq!(t.point_from_shape_pos(&[1, 2, 1], false), 11.);

        assert_eq!(t.point_from_shape_pos(&[0, 1, 1], false), 3.);
        assert_eq!(t.point_from_shape_pos(&[0, 0, 1], false), 1.);
    }

    #[test]
    fn point_from_shape_pos_broadcasting() {
        let t = Tensor::new(vec![0., 1., 2., 3., 4., 5.], vec![1, 3, 2]);

        assert_eq!(t.point_from_shape_pos(&[1, 2, 0], true), 4.);
    }

    #[test]
    #[should_panic]
    fn point_from_shape_pos_invalid_shape_pos() {
        let t = Tensor::new(
            vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            vec![2, 3, 2],
        );

        t.point_from_shape_pos(&[0, 0, 2], false);
    }

    #[test]
    fn test_sqrt() {
        let t = Tensor::from_vec(vec![
            -2.0755000114,
            1.0226000547,
            0.0830999985,
            0.4805999994,
        ]);
        let result = t.sqrt();

        assert_aprox_eq_vec(
            result.data,
            vec![NAN, 1.0112369061, 0.2882707119, 0.6932532191],
            1e-6,
        )
    }

    #[test]
    fn test_relu() {
        let t = Tensor::from_vec(vec![0.1421, -0.2579, 1.5635, -0.4066]);

        let result = t.relu();

        assert_eq!(result.data, vec![0.1421, 0.0000, 1.5635, 0.0000]);
        assert_eq!(result.shape, vec![4]);
    }

    #[test]
    fn test_linear() {
        let t = Tensor::rand(vec![128, 20]);
        let result = t.linear(&Tensor::zeros(20), &Tensor::zeros(30), None);
        assert_eq!(result.shape, vec![128, 30]);
    }

    #[test]
    fn test_flatten() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let result = t.flatten(None);
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(result.shape, vec![8]);
    }

    #[test]
    fn test_flatten_with_start_dim() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let result = t.flatten(Some(1));
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(result.shape, vec![2, 4]);
    }

    #[test]
    fn test_flatten_with_start_dim_zero() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
        let result = t.flatten(Some(0));
        assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(result.shape, vec![8]);
    }

    #[test]
    fn test_max_pool2d() {
        let input = Tensor::new(
            vec![
                8., 6., 6., 8., //
                9., 7., 7., 9., //
                8., 6., 6., 8., //
                9., 7., 7., 9., //
            ],
            vec![1, 1, 4, 4],
        );

        let output = input.max_pool2d(2, None);
        assert_eq!(
            output.data,
            vec![9.0, 7.0, 9.0, 9.0, 7.0, 9.0, 9.0, 7.0, 9.0]
        );
        assert_eq!(output.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_avg_pool2d() {
        let input = Tensor::new(
            vec![
                1., 2., 3., 4., //
                5., 6., 7., 8., //
                9., 10., 11., 12., //
                13., 14., 15., 16., //
            ],
            vec![1, 1, 4, 4],
        );

        let output = input.avg_pool2d((2, 2), None);
        assert_eq!(
            output.data,
            vec![3.5, 4.5, 5.5, 7.5, 8.5, 9.5, 11.5, 12.5, 13.5]
        );
        assert_eq!(output.shape, vec![1, 1, 3, 3]);
    }

    #[test]
    fn test_max_pool2d_stride() {
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, //
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
                9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, //
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, //
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
                9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, //
            ],
            vec![1, 1, 8, 8],
        );

        let output = input.max_pool2d(2, Some(2));
        assert_eq!(
            output.data,
            vec![8., 6., 6., 8., 9., 7., 7., 9., 8., 6., 6., 8., 9., 7., 7., 9.]
        );
        assert_eq!(output.shape, vec![1, 1, 4, 4]);
    }
    #[test]
    fn test_permute() {
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            vec![2, 3, 4],
        );

        let result = input.permute(vec![1, 2, 0]);
        assert_eq!(
            result.data,
            vec![
                1.0, 13.0, 2.0, 14.0, 3.0, 15.0, 4.0, 16.0, 5.0, 17.0, 6.0, 18.0, 7.0, 19.0, 8.0,
                20.0, 9.0, 21.0, 10.0, 22.0, 11.0, 23.0, 12.0, 24.0
            ]
        );
        assert_eq!(result.shape, vec![3, 4, 2]);
    }

    #[test]
    fn test_swish() {
        let input = Tensor::from_vec(vec![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0]);
        let result = input.swish();

        assert_eq!(
            result.data,
            vec![
                -0.014835738939808652,
                -0.07194483984836625,
                -0.23840584404423515,
                0.0,
                1.7615941559557646,
                3.928055160151634,
            ]
        );
        assert_eq!(result.shape, vec![6]);
    }

    #[test]
    fn test_sigmoid() {
        let input = Tensor::from_vec(vec![
            0.3841006458,
            0.5181258917,
            -0.5540073514,
            -0.4315102994,
        ]);
        let result = input.sigmoid();

        assert_aprox_eq_vec(
            result.data,
            vec![0.5948617458, 0.6267094612, 0.3649351895, 0.3937657773],
            1e-6,
        );
        assert_eq!(input.shape, vec![4]);
    }
}
