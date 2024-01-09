use std::{
    collections::{HashMap, HashSet},
    fs,
    rc::Rc,
};

use petgraph::{dot::Dot, stable_graph::NodeIndex, Graph};
use tracing::{debug, trace};

use crate::{op::UnrealizedOp, tensor::Tensor};

pub fn is_loop(t: &Tensor) -> bool {
    trace!("Checking tensor for loops...");

    let mut seen_ops: HashSet<UnrealizedOp> = HashSet::new();
    let mut parent_ops: HashSet<UnrealizedOp> = HashSet::new();
    let mut node_indeces: HashMap<UnrealizedOp, NodeIndex> = HashMap::new();
    let mut g = Graph::<Rc<UnrealizedOp>, ()>::new();
    let node_index = g.add_node(Rc::new(t.unrealized_op.clone()));
    let looping = is_cyclical(
        &mut seen_ops,
        &mut parent_ops,
        &t.unrealized_op,
        &mut g,
        node_index,
        &mut node_indeces,
    );

    fs::write("graph.dot", format!("{:?}", Dot::new(&g))).unwrap();
    looping
}

fn is_cyclical(
    seen_ops: &mut HashSet<UnrealizedOp>,
    parent_ops: &mut HashSet<UnrealizedOp>,
    op: &UnrealizedOp,
    g: &mut Graph<Rc<UnrealizedOp>, ()>,
    node_index: NodeIndex,
    node_indeces: &mut HashMap<UnrealizedOp, NodeIndex>,
) -> bool {
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

    if seen_ops.contains(&op) && parent_ops.contains(&op) {
        return true;
    }

    if seen_ops.contains(&op) {
        return false;
    }

    seen_ops.insert(op.clone());
    parent_ops.insert(op.clone());

    for child in children {
        if let Some(i) = node_indeces.get(child) {
            g.add_edge(node_index, *i, ());
            continue;
        }

        let child_index = g.add_node(child.clone().to_owned());
        node_indeces.insert(op.clone(), child_index);
        g.add_edge(node_index, child_index, ());
        if is_cyclical(seen_ops, parent_ops, child, g, child_index, node_indeces) {
            debug!("{:?}, {}", child, id);
            return true;
        }
    }

    parent_ops.remove(&op);

    false
}
