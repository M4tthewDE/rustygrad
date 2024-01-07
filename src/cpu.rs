use std::{cmp, f64::consts::E, iter::zip};

use itertools::{EitherOrBoth, Itertools};

use crate::{op::UnrealizedOp, tensor::Tensor, util};

impl UnrealizedOp {
    pub fn realize(&mut self) -> (Vec<f64>, Vec<usize>) {
        match self {
            UnrealizedOp::Add(lhs, rhs) => {
                lhs.realize();
                rhs.realize();
                broadcast_op(lhs, rhs, |x1, x2| x1 + x2)
            }
            UnrealizedOp::Sub(lhs, rhs) => {
                lhs.realize();
                rhs.realize();
                broadcast_op(lhs, rhs, |x1, x2| x1 - x2)
            }
            UnrealizedOp::Mul(lhs, rhs) => {
                lhs.realize();
                rhs.realize();
                broadcast_op(lhs, rhs, |x1, x2| x1 * x2)
            }
            UnrealizedOp::Div(lhs, rhs) => {
                lhs.realize();
                rhs.realize();
                broadcast_op(lhs, rhs, |x1, x2| x1 / x2)
            }
            UnrealizedOp::Sqrt(t) => {
                t.realize();
                (
                    t.data.clone().unwrap().iter().map(|x| x.sqrt()).collect(),
                    t.shape.clone(),
                )
            }
            UnrealizedOp::Log(t) => {
                t.realize();
                (
                    t.data.clone().unwrap().iter().map(|x| x.log2()).collect(),
                    t.shape.clone(),
                )
            }
            UnrealizedOp::Sigmoid(t) => {
                t.realize();
                (
                    t.data
                        .clone()
                        .unwrap()
                        .iter()
                        .map(|x| (1.0 / (1.0 + E.powf(-x))))
                        .collect(),
                    t.shape.clone(),
                )
            }
            UnrealizedOp::Relu(t) => {
                t.realize();
                (
                    t.data
                        .clone()
                        .unwrap()
                        .iter()
                        .map(|&x| if x < 0.0 { 0.0 } else { x })
                        .collect(),
                    t.shape.clone(),
                )
            }
            UnrealizedOp::Max(t) => {
                t.realize();
                let val = t
                    .data
                    .clone()
                    .expect("no data. tensor not loaded?")
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .expect("no min value found");
                (vec![val], vec![])
            }
            UnrealizedOp::Min(t) => {
                t.realize();
                let val = t
                    .data
                    .clone()
                    .expect("no data. tensor not loaded?")
                    .into_iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .expect("no min value found");
                (vec![val], vec![])
            }
            UnrealizedOp::Sum(t, dims, keepdim) => {
                t.realize();
                let data = t.data.clone().expect("no data. tensor not loaded?");
                let dims = match dims {
                    Some(dims) => dims,
                    None => return (vec![data.iter().sum()], vec![]),
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

                (result, new_shape)
            }
            UnrealizedOp::Pool2D(t, kernel, stride, init_val, pool_op) => {
                t.realize();
                let data = t.data.clone().expect("no data. tensor not loaded?");
                // FIXME: remove this constraint, just reshape or something smarter
                assert_eq!(t.shape.len(), 4, "only supporting 4d tensors");

                let (batch, channels, height, width) =
                    (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
                let (kernel_height, kernel_width) = (kernel.0, kernel.1);

                let output_height = ((height - kernel_height) / *stride) + 1;
                let output_width = ((width - kernel_width) / *stride) + 1;

                let mut output_data =
                    Vec::with_capacity(batch * channels * output_height * output_width);
                for n in 0..batch {
                    for c in 0..channels {
                        for i in 0..output_height {
                            for j in 0..output_width {
                                let mut result_val = *init_val;
                                for ki in 0..kernel_height {
                                    for kj in 0..kernel_width {
                                        let row = i * *stride + ki;
                                        let col = j * *stride + kj;
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
                (
                    output_data,
                    vec![batch, channels, output_height, output_width],
                )
            }
            UnrealizedOp::Conv2D(t, kernel, strides, groups) => {
                assert_eq!(t.shape.len(), 4, "only supporting 4d tensors");
                assert_eq!(kernel.shape.len(), 4, "only supporting 4d kernels");

                let groups = groups.unwrap_or(1);
                assert_eq!(
                    t.shape[1] % groups,
                    0,
                    "input channels must be divisible by groups"
                );
                assert_eq!(
                    kernel.shape[0] % groups,
                    0,
                    "output channels must be divisible by groups"
                );

                let strides = strides.unwrap_or((1, 1));

                let (n, c_in, height, width) = (t.shape[0], t.shape[1], t.shape[2], t.shape[3]);
                let (c_out, kernel_height, kernel_width) =
                    (kernel.shape[0], kernel.shape[2], kernel.shape[3]);

                let output_height = ((height - kernel_height) / strides.0) + 1;
                let output_width = ((width - kernel_width) / strides.1) + 1;

                let c_in_per_group = c_in / groups;
                let c_out_per_group = c_out / groups;

                t.realize();
                let t_data = t.data.clone().unwrap();
                kernel.realize();
                let kernel_data = kernel.data.clone().unwrap();
                let mut output_data = Vec::new();
                for n_index in 0..n {
                    for g in 0..groups {
                        for c_out_index in (g * c_out_per_group)..((g + 1) * c_out_per_group) {
                            for i in 0..output_height {
                                for j in 0..output_width {
                                    let mut value = 0.0;
                                    for c_in_index in
                                        (g * c_in_per_group)..((g + 1) * c_in_per_group)
                                    {
                                        for k_row in 0..kernel_height {
                                            for k_col in 0..kernel_width {
                                                let row = i * strides.0 + k_row;
                                                let col = j * strides.1 + k_col;
                                                if row < height && col < width {
                                                    value += t_data[util::index_4d_to_1d(
                                                        t.shape.clone(),
                                                        n_index,
                                                        c_in_index,
                                                        row,
                                                        col,
                                                    )] * kernel_data[util::index_4d_to_1d(
                                                        kernel.shape.clone(),
                                                        c_out_index, // removed group adjustment as each kernel is only for one group
                                                        c_in_index % c_in_per_group, // local index within group
                                                        k_row,
                                                        k_col,
                                                    )];
                                                }
                                            }
                                        }
                                    }
                                    output_data.push(value);
                                }
                            }
                        }
                    }
                }

                (output_data, vec![n, c_out, output_height, output_width])
            }
            UnrealizedOp::Pad2D(t, value, padding) => {
                if t.shape.len() < 2 {
                    panic!("Tensor must have at least 2 dimensions for 2D padding.");
                }

                let last_two_dims = t.shape.len() - 2;
                let mut new_shape: Vec<usize> = t.shape.clone();

                new_shape[last_two_dims] += padding[2] + padding[3]; // top + bottom
                new_shape[last_two_dims + 1] += padding[0] + padding[1]; // left + right

                let mut new_data = vec![*value; new_shape.iter().product()];

                t.realize();
                let data = t.data.clone().unwrap();
                for i in 0..data.len() {
                    let mut temp_index = i;
                    let mut multi_dim_index = Vec::new();

                    for &size in t.shape.iter().rev() {
                        multi_dim_index.push(temp_index % size);
                        temp_index /= size;
                    }
                    multi_dim_index.reverse();

                    // bottom and right padding is added in the initialization
                    if multi_dim_index.len() >= 2 {
                        multi_dim_index[last_two_dims] += padding[2]; // top padding
                        multi_dim_index[last_two_dims + 1] += padding[0]; // left padding
                    }

                    let mut new_index = 0;
                    let mut stride = 1;
                    for (&size, &index) in new_shape.iter().rev().zip(multi_dim_index.iter().rev())
                    {
                        new_index += index * stride;
                        stride *= size;
                    }

                    new_data[new_index] = data[i];
                }

                (new_data, new_shape)
            }
            UnrealizedOp::Load(data, shape) => (data.clone(), shape.clone()),
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
fn broadcast_op(lhs: &Tensor, rhs: &Tensor, op: BroadcastOp) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, mut lhs_shape) = (
        lhs.data.clone().expect("no data. tensor not loaded?"),
        lhs.shape.clone(),
    );
    let (rhs_data, mut rhs_shape) = (
        rhs.data.clone().expect("no data. tensor not loaded?"),
        rhs.shape.clone(),
    );
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
