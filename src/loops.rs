use tracing::{trace, warn};
use uuid::Uuid;

use crate::{op::UnrealizedOp, tensor::Tensor};

pub fn is_loop(t: &Tensor) -> bool {
    trace!("Checking tensor for loops...");

    let mut seen_ops: Vec<Uuid> = Vec::new();

    handle_op(&mut seen_ops, &t.unrealized_op);
    false
}

fn see_op(seen_ops: &mut Vec<Uuid>, uuid: Uuid) {
    if seen_ops.contains(&uuid) {
        warn!("op seen previously {}", uuid);
    }

    seen_ops.push(uuid);
}

fn handle_op(seen_ops: &mut Vec<Uuid>, op: &UnrealizedOp) {
    match op {
        UnrealizedOp::Add(lhs, rhs, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, lhs);
            handle_op(seen_ops, rhs);
        }
        UnrealizedOp::Sub(lhs, rhs, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, lhs);
            handle_op(seen_ops, rhs);
        }
        UnrealizedOp::Mul(lhs, rhs, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, lhs);
            handle_op(seen_ops, rhs);
        }
        UnrealizedOp::Div(lhs, rhs, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, lhs);
            handle_op(seen_ops, rhs);
        }
        UnrealizedOp::Max(t, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Min(t, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Sqrt(t, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Log(t, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Load(_, _, uuid) => {
            see_op(seen_ops, *uuid);
        }
        UnrealizedOp::Sigmoid(t, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Relu(t, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Sum(t, _, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Pool2D(t, _, _, _, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Conv2D(t, kernel, _, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
            handle_op(seen_ops, kernel);
        }
        UnrealizedOp::Pad2D(t, _, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Reshape(t, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Permute(t, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::Expand(t, _, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, t);
        }
        UnrealizedOp::MatMul(lhs, rhs, uuid) => {
            see_op(seen_ops, *uuid);
            handle_op(seen_ops, lhs);
            handle_op(seen_ops, rhs);
        }
    }
}
