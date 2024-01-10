use std::{
    collections::{HashMap, HashSet},
    fs,
    rc::Rc,
};

use petgraph::{dot::Dot, stable_graph::NodeIndex, Graph};
use tracing::debug;

use crate::{
    op::{Op, UnrealizedOp},
    tensor::Tensor,
};

pub fn build_graph(t: &Tensor) {
    debug!("Building graph...");

    let mut seen_ops: HashSet<UnrealizedOp> = HashSet::new();
    let mut node_indeces: HashMap<UnrealizedOp, NodeIndex> = HashMap::new();
    let mut g = Graph::<Rc<UnrealizedOp>, ()>::new();
    let node_index = g.add_node(Rc::new(t.unrealized_op.clone()));
    graph_op(
        &mut seen_ops,
        &t.unrealized_op,
        &mut g,
        node_index,
        &mut node_indeces,
    );

    debug!("writing graph...");
    fs::write("graph.dot", format!("{:?}", Dot::new(&g))).unwrap();
}

fn graph_op(
    seen_ops: &mut HashSet<UnrealizedOp>,
    unrealized_op: &UnrealizedOp,
    g: &mut Graph<Rc<UnrealizedOp>, ()>,
    node_index: NodeIndex,
    node_indeces: &mut HashMap<UnrealizedOp, NodeIndex>,
) {
    let children = match &unrealized_op.op {
        Op::Add(lhs, rhs) => vec![lhs, rhs],
        Op::Sub(lhs, rhs) => vec![lhs, rhs],
        Op::Mul(lhs, rhs) => vec![lhs, rhs],
        Op::Div(lhs, rhs) => vec![lhs, rhs],
        Op::Max(t) => vec![t],
        Op::Min(t) => vec![t],
        Op::Sqrt(t) => vec![t],
        Op::Log(t) => vec![t],
        Op::Load(_, _) => vec![],
        Op::Sigmoid(t) => vec![t],
        Op::Relu(t) => vec![t],
        Op::Sum(t, _, _) => vec![t],
        Op::Pool2D(t, _, _, _, _) => vec![t],
        Op::Conv2D(t, kernel, _, _) => vec![t, kernel],
        Op::Pad2D(t, _, _) => vec![t],
        Op::Reshape(t, _) => vec![t],
        Op::Permute(t, _) => vec![t],
        Op::Expand(t, _) => vec![t],
        Op::MatMul(lhs, rhs) => vec![lhs, rhs],
    };

    if seen_ops.contains(unrealized_op) {
        return;
    }

    seen_ops.insert(unrealized_op.clone());

    for child in children {
        if let Some(i) = node_indeces.get(child) {
            g.add_edge(node_index, *i, ());
            continue;
        }

        let child_index = g.add_node(child.clone().to_owned());
        node_indeces.insert(unrealized_op.clone(), child_index);
        g.add_edge(node_index, child_index, ());
        graph_op(seen_ops, child, g, child_index, node_indeces);
    }
}
