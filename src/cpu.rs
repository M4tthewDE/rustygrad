use std::{cmp, iter::zip};

use itertools::{EitherOrBoth, Itertools};

use crate::{op::UnrealizedOp, tensor::Tensor};

impl UnrealizedOp {
    pub fn realize(&self) -> Tensor {
        match self {
            UnrealizedOp::Add(lhs, rhs) => {
                let (data, shape) = broadcast_op(lhs.realize(), rhs.realize(), |x1, x2| x1 + x2);
                Tensor::new(self.clone(), data, shape)
            }
            UnrealizedOp::Sub(lhs, rhs) => {
                let (data, shape) = broadcast_op(lhs.realize(), rhs.realize(), |x1, x2| x1 - x2);
                Tensor::new(self.clone(), data, shape)
            }
            UnrealizedOp::Mul(lhs, rhs) => {
                let (data, shape) = broadcast_op(lhs.realize(), rhs.realize(), |x1, x2| x1 * x2);
                Tensor::new(self.clone(), data, shape)
            }
            UnrealizedOp::Div(lhs, rhs) => {
                let (data, shape) = broadcast_op(lhs.realize(), rhs.realize(), |x1, x2| x1 / x2);
                Tensor::new(self.clone(), data, shape)
            }
            UnrealizedOp::Max(t) => {
                let val = t
                    .realize()
                    .data
                    .expect("no data. tensor not loaded?")
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .expect("no min value found");
                Tensor::new(self.clone(), vec![val], vec![])
            }
            UnrealizedOp::Min(t) => {
                let val = t
                    .realize()
                    .data
                    .expect("no data. tensor not loaded?")
                    .into_iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .expect("no max value found");
                Tensor::new(self.clone(), vec![val], vec![])
            }
            UnrealizedOp::Sum(t, dims, keepdim) => {
                let t = t.realize();
                let data = t.data.unwrap();
                let dims = match dims {
                    Some(dims) => dims,
                    None => return Tensor::from_scalar(data.iter().sum()),
                };

                let mut reduced_shape = t.shape.clone();
                for (i, dim) in dims.iter().enumerate() {
                    reduced_shape.remove(*dim - i);
                }

                let mut result: Vec<f64> = vec![0.; reduced_shape.iter().product()];

                let mut shape_pos = Vec::with_capacity(t.shape.len() - dims.len());
                for (i, elem) in data.iter().enumerate() {
                    shape_pos.clear();
                    let mut offset = 0;
                    for (j, _shape) in t.shape.iter().enumerate() {
                        let count = t.shape[..=j].iter().product::<usize>();
                        let index = (i - offset) / (data.len() / count);
                        if !dims.contains(&j) {
                            shape_pos.push(index);
                        }
                        offset += (data.len() / count) * index;
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

                let new_shape = if *keepdim {
                    t.shape
                        .iter()
                        .enumerate()
                        .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                        .collect()
                } else {
                    reduced_shape
                };

                Tensor::new(self.clone(), result, new_shape)
            }
            UnrealizedOp::Reshape(t, shape) => {
                let t = t.realize();
                Tensor::new(self.clone(), t.data.unwrap(), shape.clone())
            }
            UnrealizedOp::Permute(t, dims) => {
                let t = t.realize();
                let data = t.data.unwrap();
                let new_shape: Vec<usize> = dims.iter().map(|&d| t.shape[d]).collect();
                let mut new_data = vec![0.0; data.len()];

                // Permute the data
                for (i, item) in data.iter().enumerate() {
                    let mut temp_index = i;
                    let mut multi_dim_index = Vec::new();
                    for &size in t.shape.iter().rev() {
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
                    for (&size, &index) in
                        new_shape.iter().rev().zip(new_multi_dim_index.iter().rev())
                    {
                        new_index += index * stride;
                        stride *= size;
                    }

                    // Place the original data into its new position
                    new_data[new_index] = *item;
                }

                Tensor::new(self.clone(), new_data, new_shape)
            }
            UnrealizedOp::Pool2D(t, kernel, stride, init_val, pool_op) => {
                let t = t.realize();
                let data = t.data.unwrap();
                // FIXME: remove this constraint, just reshape or something smarter
                assert_eq!(t.shape.len(), 4, "only supporting 4d tensors");

                let (batch, channels, height, width) =
                    (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
                let (kernel_height, kernel_width) = (kernel.0, kernel.1);

                let output_height = ((height - kernel_height) / stride) + 1;
                let output_width = ((width - kernel_width) / stride) + 1;

                let mut output_data =
                    Vec::with_capacity(batch * channels * output_height * output_width);
                for n in 0..batch {
                    for c in 0..channels {
                        for i in 0..output_height {
                            for j in 0..output_width {
                                let mut result_val = *init_val;
                                for ki in 0..kernel_height {
                                    for kj in 0..kernel_width {
                                        let row = i * stride + ki;
                                        let col = j * stride + kj;
                                        let idx = n * (channels * height * width)
                                            + c * (height * width)
                                            + row * width
                                            + col;
                                        result_val = (pool_op)(result_val, data[idx]);
                                    }
                                }
                                output_data.push(result_val);
                            }
                        }
                    }
                }

                Tensor::new(
                    self.clone(),
                    output_data,
                    vec![batch, channels, output_height, output_width],
                )
            }
            UnrealizedOp::Load(data, shape) => {
                Tensor::new(self.clone(), data.clone(), shape.clone())
            }
        }
    }
}

fn broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
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
fn broadcast_op(lhs: Tensor, rhs: Tensor, op: BroadcastOp) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, mut lhs_shape) = (lhs.data.expect("no data. tensor not loaded?"), lhs.shape);
    let (rhs_data, mut rhs_shape) = (rhs.data.expect("no data. tensor not loaded?"), rhs.shape);
    assert!(
        broadcastable(&lhs_shape, &rhs_shape),
        "{:?} and {:?} aren't broadcastable",
        lhs_shape,
        rhs_shape
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

    let result_len = output_shape.iter().product();

    let mut result_data = vec![0.0; result_len];
    let mut shape_pos = Vec::with_capacity(output_shape.len());
    for (i, elem) in result_data.iter_mut().enumerate() {
        shape_pos.clear();
        let mut offset = 0;
        for (j, _) in output_shape.iter().enumerate() {
            let count = output_shape[..=j].iter().product::<usize>();
            let index = (i - offset) / (result_len / count);
            shape_pos.push(index);
            offset += (result_len / count) * index;
        }

        *elem = (op)(
            point_from_shape_pos(&lhs_data, &lhs_shape, &shape_pos),
            point_from_shape_pos(&rhs_data, &rhs_shape, &shape_pos),
        );
    }

    (result_data, output_shape)
}

fn point_from_shape_pos(data: &Vec<f64>, shape: &[usize], shape_pos: &[usize]) -> f64 {
    let mut index = 0;
    let mut divisor = 1;
    for (i, dim) in shape.iter().enumerate() {
        if *dim == 1 {
            continue;
        }
        assert!(shape_pos[i] < *dim);
        divisor *= dim;
        index += (data.len() / divisor) * shape_pos[i];
    }

    data[index]
}
