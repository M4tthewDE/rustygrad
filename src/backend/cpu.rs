use std::{f64::consts::E, iter::zip, rc::Rc};

use tracing::trace;

use crate::{
    op::{Op, OpCache, PoolOp, UnrealizedOp},
    util,
};

fn add(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let result = zip(lhs_data, rhs_data).map(|(l, r)| l + r).collect();

    (result, lhs_shape)
}

fn sub(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let result = zip(lhs_data, rhs_data).map(|(l, r)| l - r).collect();

    (result, lhs_shape)
}

fn mul(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let result = zip(lhs_data, rhs_data).map(|(l, r)| l * r).collect();

    (result, lhs_shape)
}

fn div(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let result = zip(lhs_data, rhs_data).map(|(l, r)| l / r).collect();

    (result, lhs_shape)
}

fn sqrt(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    (data.iter().map(|x| x.sqrt()).collect(), shape)
}

fn log(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    (data.iter().map(|x| x.log2()).collect(), shape)
}

fn sigmoid(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    (
        data.iter().map(|x| (1.0 / (1.0 + E.powf(-x)))).collect(),
        shape,
    )
}

fn relu(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    (data.iter().map(|&x| x.max(0.0)).collect(), shape)
}

fn max(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, _) = realize(t, cache);
    let val = data
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .expect("no max value found");
    (vec![val], vec![])
}

fn min(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, _) = realize(t, cache);
    let val = data
        .into_iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .expect("no min value found");
    (vec![val], vec![])
}

fn sum(
    t: &Rc<UnrealizedOp>,
    dims: &[usize],
    keepdim: &bool,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);

    let mut reduced_shape = shape.clone();
    for (i, dim) in dims.iter().enumerate() {
        reduced_shape.remove(*dim - i);
    }

    let mut result: Vec<f64> = vec![0.; reduced_shape.iter().product()];

    for (i, elem) in data.iter().enumerate() {
        let mut offset = 0;
        let mut new_index = 0;
        let mut reduced_shape_idx = 0;
        for j in 0..shape.len() {
            let count = shape[..=j].iter().product::<usize>();
            let index = (i - offset) / (data.len() / count);
            if !dims.contains(&j) {
                if reduced_shape_idx == reduced_shape.len() - 1 {
                    new_index += index;
                } else {
                    new_index += index * reduced_shape[reduced_shape.len() - reduced_shape_idx - 1];
                }
                reduced_shape_idx += 1;
            }
            offset += (data.len() / count) * index;
        }

        *result.get_mut(new_index).unwrap() += elem;
    }

    let new_shape = if *keepdim {
        shape
            .iter()
            .enumerate()
            .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
            .collect()
    } else {
        reduced_shape
    };

    (result, new_shape)
}

fn pool2d(
    t: &Rc<UnrealizedOp>,
    kernel: &(usize, usize),
    stride: &usize,
    init_val: &f64,
    pool_op: &PoolOp,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    // FIXME: remove this constraint, just reshape or something smarter
    assert_eq!(shape.len(), 4, "only supporting 4d tensors");

    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let (kernel_height, kernel_width) = (kernel.0, kernel.1);

    let output_height = ((height - kernel_height) / stride) + 1;
    let output_width = ((width - kernel_width) / stride) + 1;

    let op: fn(lhs: f64, rhs: f64) -> f64 = match pool_op {
        PoolOp::Sum => |a, b| a + b,
        PoolOp::Max => |a, b| a.max(b),
    };

    let mut output_data = Vec::with_capacity(batch * channels * output_height * output_width);
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
                            result_val = (op)(result_val, data[idx]);
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

fn conv2d(
    t: &Rc<UnrealizedOp>,
    kernel: &Rc<UnrealizedOp>,
    strides: &(usize, usize),
    groups: &usize,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    let (kernel_data, kernel_shape) = realize(kernel, cache);
    assert_eq!(shape.len(), 4, "only supporting 4d tensors");
    assert_eq!(kernel_shape.len(), 4, "only supporting 4d kernels");

    assert_eq!(
        shape[1] % groups,
        0,
        "input channels must be divisible by groups"
    );
    assert_eq!(
        kernel_shape[0] % groups,
        0,
        "output channels must be divisible by groups"
    );

    let (n, c_in, height, width) = (shape[0], shape[1], shape[2], shape[3]);
    let (c_out, kernel_height, kernel_width) = (kernel_shape[0], kernel_shape[2], kernel_shape[3]);

    let output_height = ((height - kernel_height) / strides.0) + 1;
    let output_width = ((width - kernel_width) / strides.1) + 1;

    let c_in_per_group = c_in / *groups;
    let c_out_per_group = c_out / *groups;

    let mut output_data = Vec::new();
    for n_index in 0..n {
        for g in 0..*groups {
            for c_out_index in (g * c_out_per_group)..((g + 1) * c_out_per_group) {
                for i in 0..output_height {
                    for j in 0..output_width {
                        let mut value = 0.0;
                        for c_in_index in (g * c_in_per_group)..((g + 1) * c_in_per_group) {
                            for k_row in 0..kernel_height {
                                for k_col in 0..kernel_width {
                                    let row = i * strides.0 + k_row;
                                    let col = j * strides.1 + k_col;
                                    if row < height && col < width {
                                        value += data[util::index_4d_to_1d(
                                            &shape, n_index, c_in_index, row, col,
                                        )] * kernel_data[util::index_4d_to_1d(
                                            &kernel_shape,
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

fn pad2d(
    t: &Rc<UnrealizedOp>,
    value: &f64,
    padding: &[usize; 4],
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    if shape.len() < 2 {
        panic!("Tensor must have at least 2 dimensions for 2D padding.");
    }

    let mut new_shape: Vec<usize> = shape.clone();

    new_shape[shape.len() - 2] += padding[2] + padding[3];
    new_shape[shape.len() - 1] += padding[0] + padding[1];

    let mut new_data = vec![*value; new_shape.iter().product()];

    for (i, elem) in data.iter().enumerate() {
        let mut temp_index = i;
        let mut new_index = 0;
        let mut stride = 1;
        for (j, (&size, new_size)) in zip(&shape, &new_shape).enumerate().rev() {
            let md_idx = if j == shape.len() - 2 {
                temp_index % size + padding[2]
            } else if j == shape.len() - 1 {
                temp_index % size + padding[0]
            } else {
                temp_index % size
            };
            new_index += md_idx * stride;
            stride *= new_size;
            temp_index /= size;
        }

        new_data[new_index] = *elem;
    }

    (new_data, new_shape)
}

fn reshape(
    t: &Rc<UnrealizedOp>,
    shape: &Vec<usize>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (data, _) = realize(t, cache);
    (data, shape.to_owned())
}

fn permute(t: &Rc<UnrealizedOp>, dims: &[usize], cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let (data, shape) = realize(t, cache);
    let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let mut new_data = vec![0.0; data.len()];

    for (i, item) in data.iter().enumerate() {
        let mut temp_index = i;
        let mut multi_dim_index = vec![0; shape.len()];
        for (j, &size) in shape.iter().enumerate().rev() {
            multi_dim_index[j] = temp_index % size;
            temp_index /= size;
        }

        let mut new_index = 0;
        let mut stride = 1;
        for (&size, &dim) in zip(&new_shape, dims).rev() {
            new_index += multi_dim_index[dim] * stride;
            stride *= size;
        }

        new_data[new_index] = *item;
    }

    (new_data, new_shape)
}

fn expand(
    t: &Rc<UnrealizedOp>,
    new_shape: &Vec<usize>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (input_data, shape) = realize(t, cache);
    assert_eq!(
        shape.len(),
        new_shape.len(),
        "Only supporting same size shapes"
    );
    for (old, new) in shape.iter().zip(new_shape.clone()) {
        assert!(
            *old == new || *old == 1,
            "Old dimension must be either 1 or identical to new dimension"
        );
    }

    let mut data = input_data.clone();
    let mut new_data = Vec::with_capacity(shape.iter().product());

    let mut n_parts = 1;
    for (&old_dim, &new_dim) in shape.iter().zip(new_shape) {
        n_parts *= old_dim;
        if old_dim == 1 {
            new_data.clear();

            for chunk in data.chunks(data.len() / n_parts) {
                new_data.extend_from_slice(&chunk.repeat(new_dim));
            }

            data = new_data.clone();
            n_parts *= new_dim;
        }
    }

    if data.is_empty() {
        (input_data, shape)
    } else {
        (data, new_shape.to_owned())
    }
}

fn matmul(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, rhs_shape) = realize(rhs, cache);
    assert!(
        lhs_shape.len() == 2 && rhs_shape.len() == 2,
        "only supporting 2d tensors for now"
    );
    assert_eq!(lhs_shape[1], rhs_shape[0]);

    let mut result = vec![0.0; lhs_shape[0] * rhs_shape[1]];
    for (i, elem) in lhs_data.iter().enumerate() {
        for j in 0..rhs_shape[1] {
            result[(i / rhs_shape[0]) * rhs_shape[1] + j] +=
                elem * rhs_data[i % rhs_shape[0] * rhs_shape[1] + j];
        }
    }

    (result, vec![lhs_shape[0], rhs_shape[1]])
}

fn load(data: &[f64], shape: &[usize]) -> (Vec<f64>, Vec<usize>) {
    (data.to_owned(), shape.to_owned())
}

pub fn realize(unrealized_op: &UnrealizedOp, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    if let Some(result) = cache.get(&unrealized_op.id) {
        return result.clone();
    }

    trace!("Realizing {:?}", unrealized_op);
    let result = match &unrealized_op.op {
        Op::Add(lhs, rhs) => add(lhs, rhs, cache),
        Op::Sub(lhs, rhs) => sub(lhs, rhs, cache),
        Op::Mul(lhs, rhs) => mul(lhs, rhs, cache),
        Op::Div(lhs, rhs) => div(lhs, rhs, cache),
        Op::Sqrt(t) => sqrt(t, cache),
        Op::Log(t) => log(t, cache),
        Op::Sigmoid(t) => sigmoid(t, cache),
        Op::Relu(t) => relu(t, cache),
        Op::Max(t) => max(t, cache),
        Op::Min(t) => min(t, cache),
        Op::Sum(t, dims, keepdim) => sum(t, dims, keepdim, cache),
        Op::Pool2D(t, kernel, stride, init_val, pool_op) => {
            pool2d(t, kernel, stride, init_val, pool_op, cache)
        }
        Op::Conv2D(t, kernel, strides, groups) => conv2d(t, kernel, strides, groups, cache),
        Op::Pad2D(t, value, padding) => pad2d(t, value, padding, cache),
        Op::Reshape(t, shape) => reshape(t, shape, cache),
        Op::Permute(t, dims) => permute(t, dims, cache),
        Op::Expand(t, new_shape) => expand(t, new_shape, cache),
        Op::MatMul(lhs, rhs) => matmul(lhs, rhs, cache),
        Op::Load(data, shape) => load(data, shape),
    };

    cache.insert(unrealized_op.id, result.clone());

    result
}
