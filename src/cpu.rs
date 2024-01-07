use std::{cmp, f64::consts::E, iter::zip};

use itertools::{EitherOrBoth, Itertools};

use crate::{op::UnrealizedOp, tensor::Tensor};

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
