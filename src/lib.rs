use std::{cmp, iter::zip, ops};

use image::DynamicImage;
use itertools::{EitherOrBoth, Itertools};
use rand::distributions::{Distribution, Uniform};

pub mod batch_norm;
pub mod efficientnet;
pub mod util;

pub fn broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    for dim_pair in shape1.iter().rev().zip_longest(shape2.iter().rev()) {
        match dim_pair {
            EitherOrBoth::Both(left, right) => {
                if !(left == right || *right == 1 || *left == 1) {
                    return false;
                }
            }
            EitherOrBoth::Left(_) => (),
            EitherOrBoth::Right(_) => (),
        }
    }

    true
}

type BroadcastOp = fn(lhs: f64, rhs: f64) -> f64;

fn broadcast_op(mut lhs: Tensor, mut rhs: Tensor, op: BroadcastOp) -> Tensor {
    assert!(broadcastable(&lhs.shape, &rhs.shape));

    match lhs.shape.len().cmp(&rhs.shape.len()) {
        cmp::Ordering::Greater => {
            while rhs.shape.len() < lhs.shape.len() {
                rhs.shape.insert(0, 1);
            }
        }
        cmp::Ordering::Less => {
            while lhs.shape.len() < rhs.shape.len() {
                lhs.shape.insert(0, 1);
            }
        }
        cmp::Ordering::Equal => (),
    }

    let output_shape: Vec<usize> = zip(&lhs.shape, &rhs.shape)
        .map(|(d1, d2)| cmp::max(*d1, *d2))
        .collect();

    let result_len = output_shape.iter().product();
    let mut result = Tensor::new(vec![0.; result_len], output_shape);

    for (i, elem) in result.data.iter_mut().enumerate() {
        let mut shape_pos: Vec<usize> = Vec::new();
        let mut offset = 0;
        for (j, _shape) in result.shape.iter().enumerate() {
            let mut count: usize = 1;
            result.shape[..=j].iter().for_each(|x| count *= *x);
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
        let uniform = Uniform::new(-1.0, 1.0);
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..shape.iter().product())
            .map(|_| uniform.sample(&mut rng))
            .collect();

        Tensor::new(data, shape)
    }

    pub fn rand_with_range(shape: Vec<usize>, range: (f64, f64)) -> Tensor {
        let uniform = Uniform::new(range.0, range.1);
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..shape.iter().product())
            .map(|_| uniform.sample(&mut rng))
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

    pub fn glorot_uniform(fan_in: usize, fan_out: usize, shape: Vec<usize>) -> Tensor {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let uniform = Uniform::from(0.0..limit);
        let mut rng = rand::thread_rng();

        let data: Vec<f64> = (0..shape.iter().product())
            .map(|_| uniform.sample(&mut rng))
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
        let mut cols: Vec<Vec<f64>> = Vec::new();

        for (i, val) in self.data.iter().enumerate() {
            let remainder = i % self.shape[1];
            if let Some(col) = cols.get_mut(remainder) {
                col.push(*val);
            } else {
                cols.push(vec![*val]);
            }
        }

        Tensor::new(cols.concat(), self.shape.iter().rev().map(|d| *d).collect())
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

    pub fn conv2d(
        mut self,
        kernel: Tensor,
        padding: Option<Vec<usize>>,
        stride: Option<usize>,
    ) -> Tensor {
        assert_eq!(self.shape.len(), 4, "only supporting 4d tensors");
        if let Some(padding) = padding {
            self = self.pad(0.0, padding);
        }

        let stride = stride.unwrap_or(1);

        let (height, width) = (self.shape[2], self.shape[3]);
        let (kernel_height, kernel_width) = (kernel.shape[0], kernel.shape[1]);

        let output_height = ((height - kernel_height) / stride) + 1;
        let output_width = ((width - kernel_width) / stride) + 1;

        let mut output_data = Vec::new();
        for i in 0..output_height {
            for j in 0..output_width {
                let patch: Vec<Vec<f64>> = (0..kernel_height)
                    .map(|k| {
                        let row_start = (i * stride + k) * width + j * stride;
                        let row_end = row_start + kernel_width;
                        self.data[row_start..row_end].to_vec()
                    })
                    .collect();

                let mut value = 0.0;
                for (y, row) in patch.iter().enumerate() {
                    for (x, cell) in row.iter().enumerate() {
                        value += cell * kernel.data[y * kernel_width + x];
                    }
                }
                output_data.push(value);
            }
        }

        Tensor::new(output_data, vec![output_height, output_width])
    }

    pub fn max_pool2d(&self, kernel_size: usize, stride: Option<usize>) -> Tensor {
        assert_eq!(self.shape.len(), 2, "only supporting 2d tensors");
        let stride = stride.unwrap_or(1);

        let (height, width) = (self.shape[0], self.shape[1]);
        let (kernel_height, kernel_width) = (kernel_size, kernel_size);

        let output_height = ((height - kernel_height) / stride) + 1;
        let output_width = ((width - kernel_width) / stride) + 1;

        let mut output_data = Vec::new();
        for i in 0..output_height {
            for j in 0..output_width {
                let patch: Vec<Vec<f64>> = (0..kernel_height)
                    .map(|k| {
                        let row_start = (i * stride + k) * width + j * stride;
                        let row_end = row_start + kernel_width;
                        self.data[row_start..row_end].to_vec()
                    })
                    .collect();

                let mut value: f64 = 0.0;
                for (_, row) in patch.iter().enumerate() {
                    for (_, cell) in row.iter().enumerate() {
                        value = value.max(*cell);
                    }
                }
                output_data.push(value);
            }
        }

        Tensor::new(output_data, vec![output_height, output_width])
    }

    pub fn pad(self, value: f64, dims: Vec<usize>) -> Tensor {
        if dims.len() != self.shape.len() {
            panic!("Padding dimensions must match the tensor dimensions.");
        }

        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .zip(dims.iter())
            .map(|(&dim, &pad)| dim + pad * 2)
            .collect();

        let mut new_data = vec![value; new_shape.iter().product()];

        for i in 0..self.data.len() {
            let mut temp_index = i;
            let mut multi_dim_index = Vec::new();

            for &size in self.shape.iter().rev() {
                multi_dim_index.push(temp_index % size);
                temp_index /= size;
            }
            multi_dim_index.reverse();

            let padded_multi_dim_index: Vec<usize> = multi_dim_index
                .iter()
                .zip(dims.iter())
                .map(|(&index, &pad)| index + pad)
                .collect();

            let mut new_index = 0;
            let mut stride = 1;
            for (&size, &index) in new_shape
                .iter()
                .rev()
                .zip(padded_multi_dim_index.iter().rev())
            {
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
        let divisor = match dims {
            Some(dims) => {
                let mut divisor = 1.0;
                for dim in dims {
                    divisor *= self.shape[*dim] as f64;
                }

                divisor
            }
            None => self.data.len() as f64,
        };

        self.reduce_sum(dims, keepdim) / (divisor - correction.unwrap_or(0.0)).max(1.0)
    }

    pub fn reduce_sum(&self, dims: Option<&Vec<usize>>, keepdim: bool) -> Tensor {
        if let Some(dims) = dims {
            let mut reduced_shape = self.shape.clone();
            for (i, dim) in dims.iter().enumerate() {
                reduced_shape.remove(*dim - i);
            }

            let mut result: Vec<f64> = vec![0.; reduced_shape.iter().product()];

            for (i, elem) in self.data.iter().enumerate() {
                let mut shape_pos: Vec<usize> = Vec::new();
                let mut offset = 0;
                for (j, _shape) in self.shape.iter().enumerate() {
                    let mut count: usize = 1;
                    self.shape[..=j].iter().for_each(|x| count *= *x);
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
                let mut new_shape = self.shape.clone();
                for dim in dims {
                    new_shape[*dim] = 1;
                }

                new_shape
            } else {
                reduced_shape
            };

            Tensor::new(result, new_shape)
        } else {
            Tensor::from_scalar(self.data.iter().sum())
        }
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

    pub fn linear(&self, in_features: usize, out_features: usize, bias: Option<bool>) -> Tensor {
        // Kaiming uniform
        let a = (6.0 / in_features as f64).sqrt();
        let weight = Tensor::rand(vec![in_features, out_features]) * a;

        if bias.unwrap_or(true) {
            let k: f64 = 1.0 / in_features as f64;
            let bias = Tensor::rand_with_range(vec![out_features], (-k.sqrt(), k.sqrt()));
            self.matmul(weight) + bias
        } else {
            self.matmul(weight)
        }
    }

    pub fn flatten(&self, start_dim: Option<usize>) -> Tensor {
        let start_dim = start_dim.unwrap_or(0);

        let mut last_dim = 1;
        for (i, dim) in self.shape.iter().rev().enumerate() {
            if self.shape.len() - i == start_dim {
                break;
            }

            last_dim *= dim;
        }

        let mut new_shape = self.shape.clone()[..start_dim].to_owned();
        new_shape.push(last_dim);

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
            vec![3, 3],
        );
        let output = input.conv2d(kernel, None, None);

        assert_eq!(output.data, vec![23., 22., 31., 26.]);
        assert_eq!(output.shape, vec![2, 2]);
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
            vec![3, 3],
        );
        let output = input.conv2d(kernel, Some(vec![0, 0, 1, 1]), None);

        assert_eq!(
            output.data,
            vec![
                8., 14., 13., 8., //
                16., 23., 22., 10., //
                20., 31., 26., 17., //
                10., 9., 15., 10., //
            ]
        );
        assert_eq!(output.shape, vec![4, 4]);
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
            vec![3, 3],
        );
        let output = input.conv2d(kernel, None, Some(2));

        assert_eq!(
            output.data,
            vec![
                23., 18., //
                18., 21., //
            ]
        );
        assert_eq!(output.shape, vec![2, 2]);
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
        let output = input.pad(0., vec![1, 1]);

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
        let output = input.pad(1., vec![2, 2]);

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
        let result = t.linear(20, 30, None);
        dbg!(result.shape, vec![128, 30]);
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
            vec![4, 4],
        );

        let output = input.max_pool2d(2, None);
        assert_eq!(
            output.data,
            vec![9.0, 7.0, 9.0, 9.0, 7.0, 9.0, 9.0, 7.0, 9.0]
        );
        assert_eq!(output.shape, vec![3, 3]);
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
            vec![8, 8],
        );

        let output = input.max_pool2d(2, Some(2));
        assert_eq!(
            output.data,
            vec![8., 6., 6., 8., 9., 7., 7., 9., 8., 6., 6., 8., 9., 7., 7., 9.]
        );
        assert_eq!(output.shape, vec![4, 4]);
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
}
