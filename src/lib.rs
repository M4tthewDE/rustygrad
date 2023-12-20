use std::{cmp, iter::zip, ops};

use itertools::{EitherOrBoth, Itertools};
use rand::distributions::{Distribution, Uniform};

pub mod batch_norm;
pub mod efficientnet;
pub mod util;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        assert_eq!(
            self.shape, rhs.shape,
            "broadcasting not supported for addition"
        );

        let result: Vec<f64> = zip(self.data, rhs.data).map(|(x1, x2)| x1 + x2).collect();

        Self::new(result, self.shape)
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    // https://pytorch.org/docs/stable/notes/broadcasting.html
    fn sub(self, rhs: Tensor) -> Self::Output {
        let mut lhs = self;
        let mut rhs = rhs;
        assert!(!lhs.shape.is_empty());
        assert!(!rhs.shape.is_empty());

        // no broadcasting needed
        if lhs.shape == rhs.shape {
            let result: Vec<f64> = zip(lhs.data, rhs.data).map(|(x1, x2)| x1 - x2).collect();
            return Tensor::new(result, lhs.shape.clone());
        }

        // check if broadcasting is possible
        for dim_pair in rhs.shape.iter().rev().zip_longest(lhs.shape.iter().rev()) {
            match dim_pair {
                EitherOrBoth::Both(left, right) => {
                    assert!(
                        left == right || *right == 1 || *left == 1,
                        "tensors are not broadcastable"
                    )
                }
                EitherOrBoth::Left(_) => (),
                EitherOrBoth::Right(_) => (),
            }
        }

        // 1. calculate new shapes
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

        let mut output_size: usize = 1;
        output_shape.iter().for_each(|x| output_size *= *x);

        let mut result = Tensor::new(vec![0.; output_size], output_shape);
        let result_len = result.data.len();

        // 2. calculate output tensor
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

            *elem = lhs.point_from_shape_pos(&shape_pos, true)
                - rhs.point_from_shape_pos(&shape_pos, true);
        }

        result
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        assert_eq!(
            self.shape, rhs.shape,
            "broadcasting not supported for multiplication"
        );

        let result: Vec<f64> = zip(self.data, rhs.data).map(|(x1, x2)| x1 * x2).collect();

        Tensor::new(result, self.shape)
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let result: Vec<f64> = self.data.iter().map(|x| x * rhs).collect();

        Tensor::new(result, self.shape)
    }
}
impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        let mut result = Vec::new();
        for elem in self.data {
            result.push(elem / rhs);
        }

        Tensor::new(result, self.shape)
    }
}

impl Tensor {
    pub fn empty() -> Self {
        Self {
            data: vec![],
            shape: vec![],
        }
    }
    pub fn from_scalar(data: f64) -> Self {
        Self::new(vec![data], vec![1])
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let len = data.len();
        Self::new(data, vec![len])
    }

    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        // empty tensors are an exception
        if !(data.is_empty() && shape.is_empty()) {
            let mut count: usize = 1;
            shape.iter().for_each(|x| count *= *x);
            assert_eq!(data.len(), count, "invalid shape for data length");
        }

        Self { data, shape }
    }

    pub fn glorot_uniform(fan_in: usize, fan_out: usize, shape: Vec<usize>) -> Self {
        let limit = (6. / (fan_in + fan_out) as f64).sqrt();
        let uniform = Uniform::from(0.0..limit);

        let mut count: usize = 1;
        shape.iter().for_each(|x| count *= *x);

        let mut rng = rand::thread_rng();
        let mut data = Vec::new();
        for _ in 0..count {
            data.push(uniform.sample(&mut rng))
        }

        Tensor::new(data, shape)
    }

    pub fn matmul(lhs: Tensor, rhs: Tensor) -> Tensor {
        assert!(
            lhs.shape.len() == 2 && rhs.shape.len() == 2,
            "only supporting 2d tensors for now"
        );

        let rows: Vec<_> = lhs.data.chunks(lhs.shape[1]).collect();

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

        Tensor::new(result, vec![lhs.shape[0], rhs.shape[1]])
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

    pub fn transpose(self) -> Self {
        let mut cols: Vec<Vec<f64>> = Vec::new();

        // TODO: do this in one pass!
        for (i, val) in self.data.iter().enumerate() {
            let remainder = i % self.shape[1];
            if let Some(col) = cols.get_mut(remainder) {
                col.push(*val);
            } else {
                cols.push(vec![*val]);
            }
        }

        let mut shape = self.shape.clone();
        shape.reverse();

        Self::new(cols.concat(), shape)
    }

    pub fn max(self) -> Self {
        match self
            .data
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
        {
            Some(val) => Self::new(vec![val], vec![1]),
            None => Self::empty(),
        }
    }

    pub fn conv2d(
        self,
        kernel: Self,
        padding: Option<(f64, usize)>,
        stride: Option<usize>,
    ) -> Self {
        if let Some((padding_value, padding_size)) = padding {
            return self
                .pad2d(padding_value, padding_size)
                .conv2d(kernel, None, stride);
        }

        let stride = stride.unwrap_or(1);

        let (height, width) = (self.shape[0], self.shape[1]);
        let (kernel_height, kernel_width) = (kernel.shape[0], kernel.shape[1]);

        let output_height = (height - kernel_height) / stride + 1;
        let output_width = (width - kernel_width) / stride + 1;

        let mut output_data = Vec::new();
        for i in 0..output_height {
            for j in 0..output_width {
                let patch: Vec<Vec<f64>> = (0..kernel_height)
                    .map(|k| {
                        (i * height + j + k * height)..(i * height + kernel_width + j + k * height)
                    })
                    .map(|range| self.data[range.clone()].to_vec())
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

        Self::new(output_data, vec![output_height, output_width])
    }

    pub fn pad2d(self, value: f64, size: usize) -> Self {
        let mut result = Vec::new();
        for _ in 0..(self.shape[0] + 2 * size) * size {
            result.push(value);
        }

        for row in self.data.chunks(self.shape[1]) {
            for _ in 0..size {
                result.push(value);
            }
            for element in row {
                result.push(*element);
            }
            for _ in 0..size {
                result.push(value);
            }
        }

        for _ in 0..(self.shape[0] + 2 * size) * size {
            result.push(value);
        }

        let mut new_shape = self.shape.clone();
        new_shape[0] = self.shape[0] + 2 * size;
        new_shape[1] = self.shape[1] + 2 * size;

        Self::new(result, new_shape)
    }

    pub fn reduce_mean(
        &self,
        dims: Option<Vec<usize>>,
        keepdim: bool,
        correction: Option<f64>,
    ) -> Tensor {
        let correction = correction.unwrap_or(0.0);
        if let Some(dims) = dims {
            let mut result = self.clone();
            for dim in dims.iter().rev() {
                // FIXME: we should max(0, ) here, to not go negative
                result =
                    result.reduce_sum(Some(vec![*dim])) / (self.shape[*dim] as f64 - correction)
            }

            if keepdim {
                result.shape = self.shape.clone();
                for dim in dims {
                    result.shape[dim] = 1;
                }
            }

            result
        } else {
            // FIXME: we should max(0, ) here, to not go negative
            self.clone().reduce_sum(None) / (self.data.len() as f64 - correction)
        }
    }

    pub fn reduce_sum(self, dims: Option<Vec<usize>>) -> Self {
        if let Some(dims) = dims {
            let mut result = self.clone();
            for dim in dims.iter().rev() {
                result = result.reduce_sum_with_axis(*dim);
            }

            result
        } else {
            Tensor::from_scalar(self.data.into_iter().sum())
        }
    }

    fn reduce_sum_with_axis(self, axis: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);

        let mut count: usize = 1;
        new_shape.iter().for_each(|x| count *= *x);
        let mut result: Vec<f64> = vec![0.; count];

        for (i, elem) in self.data.iter().enumerate() {
            let mut shape_pos: Vec<usize> = Vec::new();
            let mut offset = 0;
            for (j, _shape) in self.shape.iter().enumerate() {
                let mut count: usize = 1;
                self.shape[..=j].iter().for_each(|x| count *= *x);
                let index = (i - offset) / (self.data.len() / count);
                if j != axis {
                    shape_pos.push(index);
                }
                offset += (self.data.len() / count) * index;
            }

            let mut index = 0;
            for (j, dim) in new_shape.iter().rev().enumerate() {
                if j == new_shape.len() - 1 {
                    index += shape_pos[j];
                } else {
                    index += shape_pos[j] * dim;
                }
            }

            *result.get_mut(index).unwrap() += elem;
        }

        Tensor::new(result, new_shape)
    }

    // FIXME: inaccurate compared to torch!
    //
    // torch.var(a, dim=1, unbiased=False)
    // torch.mean((a - torch.mean(a, dim=1, keepdim=True)) ** 2, dim=1)
    //
    //
    pub fn variance(&self, dims: Option<Vec<usize>>) -> Self {
        let mean = self.reduce_mean(dims.clone(), true, None);
        let diff = self.clone() - mean;
        (diff.clone() * diff).reduce_mean(dims, false, Some(1.0))
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        Tensor::new(self.data, shape)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn size(self, dim: Option<usize>) -> Vec<usize> {
        if let Some(dim) = dim {
            vec![self.shape[dim]]
        } else {
            self.shape
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::zip;

    use assert_approx_eq::assert_approx_eq;

    use crate::Tensor;

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
        let result = Tensor::matmul(a, b);

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
            vec![4, 4],
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
            vec![4, 4],
        );
        let kernel = Tensor::new(
            vec![
                1., 2., 3., //
                0., 1., 0., //
                2., 1., 2., //
            ],
            vec![3, 3],
        );
        let output = input.conv2d(kernel, Some((0., 1)), None);

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
            vec![5, 5],
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
                23., 22., //
                31., 26., //
            ]
        );
        assert_eq!(output.shape, vec![2, 2]);
    }

    #[test]
    fn pad2d() {
        let input = Tensor::new(
            vec![
                1., 3., //
                2., 5., //
            ],
            vec![2, 2],
        );
        let output = input.pad2d(0., 1);

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
        let output = input.pad2d(1., 2);

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

        let mean = input.reduce_mean(Some(vec![0]), true, None);

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

        let mean = input.reduce_mean(Some(vec![0]), false, None);

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

        let mean = input.reduce_mean(Some(vec![1]), false, None);

        assert_eq!(mean.data, vec![1.75, 2.0, 1.75, 2.75]);
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

        let sum = input.reduce_sum(None);

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

        let sum = input.reduce_sum(Some(vec![0]));

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

        let sum = input.reduce_sum(Some(vec![1]));

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

        let sum = input.reduce_sum(Some(vec![0]));

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

        let sum = input.reduce_sum(Some(vec![1]));

        assert_eq!(sum.data, vec![9., 12., 27., 30.0]);
        assert_eq!(sum.shape, vec![2, 2]);
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

        let sum = input.reduce_sum(Some(vec![2]));

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

        let mean = input.reduce_sum(Some(vec![0]));

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

        let sum = input.reduce_sum(Some(vec![1]));

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

        let sum = input.reduce_sum(Some(vec![0, 1]));

        assert_eq!(sum.data, vec![36., 42.]);
        assert_eq!(sum.shape, vec![2]);
    }

    #[test]
    fn reduce_mean_multiple_axis_3d() {
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

        let sum = input.reduce_mean(Some(vec![0, 1]), false, None);

        assert_eq!(sum.data, vec![6., 7.]);
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

        let result = input.variance(Some(vec![1]));

        for (a, b) in zip(
            result.data,
            vec![1.0630833092, 0.5590269933, 1.4893144158, 0.8257591867],
        ) {
            assert_approx_eq!(a, b, 1.0e-9);
        }

        assert_eq!(result.shape, vec![4]);
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
}
