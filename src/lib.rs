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

    fn sub(self, rhs: Tensor) -> Self::Output {
        // https://pytorch.org/docs/stable/notes/broadcasting.html
        assert!(!self.shape.is_empty());
        assert!(!rhs.shape.is_empty());

        let lhs = self;

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

        match lhs.shape.len().cmp(&rhs.shape.len()) {
            cmp::Ordering::Greater => {
                let mut new_shape = rhs.shape.clone();

                while new_shape.len() < lhs.shape.len() {
                    new_shape.push(1);
                }

                let new_shape: Vec<usize> = zip(lhs.shape.clone(), new_shape)
                    .map(|(d1, d2)| cmp::max(d1, d2))
                    .collect();

                let mut new_data = rhs.data.clone();
                let mut expected_len_lhs = 1;
                let mut expected_len_rhs = 1;
                for (dim_lhs, dim_rhs) in zip(lhs.shape.clone(), new_shape.clone()) {
                    expected_len_lhs *= dim_lhs;
                    expected_len_rhs *= dim_rhs;

                    let expected_len = cmp::min(expected_len_lhs, expected_len_rhs);

                    while new_data.len() < expected_len {
                        new_data.extend(new_data.clone().iter());
                    }
                }

                let result: Vec<f64> = zip(lhs.data, new_data).map(|(x1, x2)| x1 - x2).collect();
                Self::new(result, new_shape)
            }
            cmp::Ordering::Less => {
                let mut new_shape = lhs.shape.clone();

                while new_shape.len() < rhs.shape.len() {
                    new_shape.push(1);
                }

                let new_shape: Vec<usize> = zip(rhs.shape.clone(), new_shape)
                    .map(|(d1, d2)| cmp::max(d1, d2))
                    .collect();

                let mut new_data = lhs.data.clone();

                let mut expected_len_lhs = 1;
                let mut expected_len_rhs = 1;
                for (dim_lhs, dim_rhs) in zip(rhs.shape.clone(), new_shape.clone()) {
                    expected_len_lhs *= dim_lhs;
                    expected_len_rhs *= dim_rhs;

                    let expected_len = cmp::min(expected_len_lhs, expected_len_rhs);

                    while new_data.len() < expected_len {
                        new_data.extend(new_data.clone().iter());
                    }
                }

                let result: Vec<f64> = zip(new_data, rhs.data).map(|(x1, x2)| x1 - x2).collect();
                Self::new(result, new_shape)
            }
            cmp::Ordering::Equal => {
                // no broadcasting required
                let result: Vec<f64> = zip(lhs.data, rhs.data).map(|(x1, x2)| x1 - x2).collect();

                Self::new(result, lhs.shape.clone())
            }
        }
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Self;

    fn mul(self, rhs: Tensor) -> Self::Output {
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

        Self::new(result, vec![self.shape[0], rhs.shape[1]])
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

    pub fn reduce_mean(&self, axis: Option<Vec<usize>>) -> Self {
        if let Some(axis) = axis {
            let mut result = self.clone();
            for dim in axis.iter().rev() {
                result = result.reduce_sum(Some(vec![*dim])) / self.shape[*dim] as f64
            }

            result
        } else {
            self.clone().reduce_sum(None) / self.data.len() as f64
        }
    }

    pub fn reduce_sum(self, axis: Option<Vec<usize>>) -> Self {
        if let Some(axis) = axis {
            let mut result = self.clone();
            for dim in axis.iter().rev() {
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

    pub fn variance(self, axis: Option<Vec<usize>>) -> Self {
        let mean = self.reduce_mean(axis);
        let _diff = self - mean;
        todo!("variance");
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        Tensor::new(self.data, shape)
    }
}

#[cfg(test)]
mod tests {
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
    fn multiplication_matrix_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![3, 2]);
        let result = a * b;

        assert_eq!(result.data, vec![13.0, 13.0, 31.0, 31.0]);
        assert_eq!(result.shape, vec![2, 2]);
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

        let mean = input.reduce_mean(None);

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

        let mean = input.reduce_mean(None);

        assert_eq!(mean.data, vec![2.0625]);
        assert_eq!(mean.shape, vec![1]);
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

        let mean = input.reduce_mean(Some(vec![0]));

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

        let mean = input.reduce_mean(Some(vec![1]));

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

        let sum = input.reduce_mean(Some(vec![0, 1]));

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

    #[ignore]
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

        assert_eq!(result.data, vec![1.0631, 0.5590, 1.4893, 0.8258]);
        assert_eq!(result.shape, vec![4]);
    }

    #[test]
    fn sub_without_broadcasting() {
        let input1 = Tensor::new(vec![0., 1., 2., 3.], vec![4]);
        let input2 = Tensor::new(vec![0., 1., 2., 3.], vec![4]);

        let result = input1 - input2;

        assert_eq!(result.data, vec![0., 0., 0., 0.]);
        assert_eq!(result.shape, vec![4]);
    }

    #[test]
    fn sub_with_broadcasting() {
        let input1 = Tensor::new(vec![0., 1., 2., 3.], vec![2, 2]);
        let input2 = Tensor::new(vec![1., 1.], vec![2]);

        let result = input1 - input2;

        assert_eq!(result.data, vec![-1., 0., 1., 2.]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn sub_with_broadcasting_left_side() {
        let input1 = Tensor::new(vec![1., 1.], vec![2]);
        let input2 = Tensor::new(vec![0., 1., 2., 3.], vec![2, 2]);

        let result = input1 - input2;

        assert_eq!(result.data, vec![1., 0., -1., -2.]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[ignore]
    #[test]
    fn sub_with_broadcasting_both_sides() {
        let input1 = Tensor::new(vec![0.; 20], vec![5, 1, 4, 1]);
        let input2 = Tensor::new(vec![0.; 3], vec![3, 1, 1]);

        let result = input1 - input2;

        assert_eq!(result.data, vec![0.; 60]);
        assert_eq!(result.shape, vec![5, 3, 4, 1]);
    }

    #[ignore]
    #[test]
    fn sub_with_broadcasting_error() {
        let input1 = Tensor::new(vec![0.; 40], vec![5, 2, 4, 1]);
        let input2 = Tensor::new(vec![0.; 3], vec![3, 1, 1]);

        let result = input1 - input2;

        assert_eq!(result.data, vec![0.; 60]);
        assert_eq!(result.shape, vec![5, 3, 4, 1]);
    }
}
