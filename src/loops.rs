use std::collections::HashSet;

use tracing::{debug, trace};

use crate::{op::UnrealizedOp, tensor::Tensor};

pub fn is_loop(t: &Tensor) -> bool {
    trace!("Checking tensor for loops...");

    let mut seen_ops: HashSet<usize> = HashSet::new();

    is_cyclical(&mut seen_ops, &t.unrealized_op)
}

fn is_cyclical(seen_ops: &mut HashSet<usize>, op: &UnrealizedOp) -> bool {
    let (id, children) = match op {
        UnrealizedOp::Add(lhs, rhs, id) => (*id, vec![lhs, rhs]),
        UnrealizedOp::Sub(lhs, rhs, id) => (*id, vec![lhs, rhs]),
        UnrealizedOp::Mul(lhs, rhs, id) => (*id, vec![lhs, rhs]),
        UnrealizedOp::Div(lhs, rhs, id) => (*id, vec![lhs, rhs]),
        UnrealizedOp::Max(t, id) => (*id, vec![t]),
        UnrealizedOp::Min(t, id) => (*id, vec![t]),
        UnrealizedOp::Sqrt(t, id) => (*id, vec![t]),
        UnrealizedOp::Log(t, id) => (*id, vec![t]),
        UnrealizedOp::Load(_, _, id) => (*id, vec![]),
        UnrealizedOp::Sigmoid(t, id) => (*id, vec![t]),
        UnrealizedOp::Relu(t, id) => (*id, vec![t]),
        UnrealizedOp::Sum(t, _, _, id) => (*id, vec![t]),
        UnrealizedOp::Pool2D(t, _, _, _, _, id) => (*id, vec![t]),
        UnrealizedOp::Conv2D(t, kernel, _, _, id) => (*id, vec![t, kernel]),
        UnrealizedOp::Pad2D(t, _, _, id) => (*id, vec![t]),
        UnrealizedOp::Reshape(t, _, id) => (*id, vec![t]),
        UnrealizedOp::Permute(t, _, id) => (*id, vec![t]),
        UnrealizedOp::Expand(t, _, id) => (*id, vec![t]),
        UnrealizedOp::MatMul(lhs, rhs, id) => (*id, vec![lhs, rhs]),
    };

    if seen_ops.contains(&id) {
        return true;
    }

    seen_ops.insert(id);

    for child in children {
        if is_cyclical(seen_ops, child) {
            debug!("{:?}, {}", child, id);
            return true;
        }
    }

    false
}
