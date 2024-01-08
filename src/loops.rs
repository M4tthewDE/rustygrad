use std::collections::HashSet;

use tracing::{debug, trace};

use crate::{op::UnrealizedOp, tensor::Tensor};

pub fn is_loop(t: &Tensor) -> bool {
    trace!("Checking tensor for loops...");

    let mut seen_ops: HashSet<usize> = HashSet::new();

    is_cyclical(&mut seen_ops, &t.unrealized_op)
}

fn see_op(seen_ops: &mut HashSet<usize>, id: usize, next_ops: Vec<&UnrealizedOp>) -> bool {
    if seen_ops.contains(&id) {
        return true;
    }

    seen_ops.insert(id);

    for op in next_ops {
        if is_cyclical(seen_ops, op) {
            debug!("{:?}, {}", op, id);
            return true;
        }
    }

    false
}

fn is_cyclical(seen_ops: &mut HashSet<usize>, op: &UnrealizedOp) -> bool {
    match op {
        UnrealizedOp::Add(lhs, rhs, id) => see_op(seen_ops, *id, vec![lhs, rhs]),
        UnrealizedOp::Sub(lhs, rhs, id) => see_op(seen_ops, *id, vec![lhs, rhs]),
        UnrealizedOp::Mul(lhs, rhs, id) => see_op(seen_ops, *id, vec![lhs, rhs]),
        UnrealizedOp::Div(lhs, rhs, id) => see_op(seen_ops, *id, vec![lhs, rhs]),
        UnrealizedOp::Max(t, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Min(t, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Sqrt(t, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Log(t, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Load(_, _, id) => see_op(seen_ops, *id, vec![]),
        UnrealizedOp::Sigmoid(t, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Relu(t, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Sum(t, _, _, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Pool2D(t, _, _, _, _, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Conv2D(t, kernel, _, _, id) => see_op(seen_ops, *id, vec![t, kernel]),
        UnrealizedOp::Pad2D(t, _, _, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Reshape(t, _, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Permute(t, _, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::Expand(t, _, id) => see_op(seen_ops, *id, vec![t]),
        UnrealizedOp::MatMul(lhs, rhs, id) => see_op(seen_ops, *id, vec![lhs, rhs]),
    }
}
