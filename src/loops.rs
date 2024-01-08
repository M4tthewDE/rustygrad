use std::collections::HashSet;

use tracing::{debug, trace};
use uuid::Uuid;

use crate::{op::UnrealizedOp, tensor::Tensor};

pub fn is_loop(t: &Tensor) -> bool {
    trace!("Checking tensor for loops...");

    let mut seen_ops: HashSet<Uuid> = HashSet::new();

    is_cyclical(&mut seen_ops, &t.unrealized_op)
}

fn see_op(seen_ops: &mut HashSet<Uuid>, uuid: Uuid, next_ops: Vec<&UnrealizedOp>) -> bool {
    if seen_ops.contains(&uuid) {
        return true;
    }

    seen_ops.insert(uuid);

    for op in next_ops {
        if is_cyclical(seen_ops, op) {
            debug!("{:?}, {}", op, uuid);
            return true;
        }
    }

    false
}

fn is_cyclical(seen_ops: &mut HashSet<Uuid>, op: &UnrealizedOp) -> bool {
    match op {
        UnrealizedOp::Add(lhs, rhs, uuid) => see_op(seen_ops, *uuid, vec![lhs, rhs]),
        UnrealizedOp::Sub(lhs, rhs, uuid) => see_op(seen_ops, *uuid, vec![lhs, rhs]),
        UnrealizedOp::Mul(lhs, rhs, uuid) => see_op(seen_ops, *uuid, vec![lhs, rhs]),
        UnrealizedOp::Div(lhs, rhs, uuid) => see_op(seen_ops, *uuid, vec![lhs, rhs]),
        UnrealizedOp::Max(t, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Min(t, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Sqrt(t, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Log(t, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Load(_, _, uuid) => see_op(seen_ops, *uuid, vec![]),
        UnrealizedOp::Sigmoid(t, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Relu(t, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Sum(t, _, _, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Pool2D(t, _, _, _, _, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Conv2D(t, kernel, _, _, uuid) => see_op(seen_ops, *uuid, vec![t, kernel]),
        UnrealizedOp::Pad2D(t, _, _, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Reshape(t, _, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Permute(t, _, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::Expand(t, _, uuid) => see_op(seen_ops, *uuid, vec![t]),
        UnrealizedOp::MatMul(lhs, rhs, uuid) => see_op(seen_ops, *uuid, vec![lhs, rhs]),
    }
}
