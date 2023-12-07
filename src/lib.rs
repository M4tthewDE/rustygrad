use std::{iter::zip, ops};

pub mod efficientnet;
mod util;

#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Self::Output {
        assert_eq!(self.shape, rhs.shape);

        let result: Vec<f64> = zip(self.data, rhs.data).map(|(x1, x2)| x1 + x2).collect();

        Self::new(result, self.shape)
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

    pub fn conv2d(self, kernel: Self) -> Self {
        let (height, width) = (self.shape[0], self.shape[1]);
        let (kernel_height, kernel_width) = (kernel.shape[0], kernel.shape[1]);

        let output_height = (height - kernel_height + 1).div_ceil(1);
        let output_width = (width - kernel_width + 1).div_ceil(1);

        let mut output_data = Vec::new();
        for i in 0..output_height {
            for j in 0..output_width {
                let patch: Vec<Vec<f64>> = (0..kernel_height)
                    .map(|k| {
                        (i * height + j + (kernel_height + 1) * k)
                            ..(i * height + kernel_width + j + (kernel_height + 1) * k)
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
        let output = input.conv2d(kernel);

        assert_eq!(output.data, vec![23., 22., 31., 26.]);
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
}
