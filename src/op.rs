// TODO: should there be Option in here ever?
// default values could be inserted instead in tensor.rs

use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

use uuid::Uuid;

#[derive(Clone)]
pub enum UnrealizedOp {
    Add(Rc<UnrealizedOp>, Rc<UnrealizedOp>, Uuid),
    Sub(Rc<UnrealizedOp>, Rc<UnrealizedOp>, Uuid),
    Mul(Rc<UnrealizedOp>, Rc<UnrealizedOp>, Uuid),
    Div(Rc<UnrealizedOp>, Rc<UnrealizedOp>, Uuid),
    Max(Rc<UnrealizedOp>, Uuid),
    Min(Rc<UnrealizedOp>, Uuid),
    Sqrt(Rc<UnrealizedOp>, Uuid),
    Log(Rc<UnrealizedOp>, Uuid),
    Load(Vec<f64>, Vec<usize>, Uuid),
    Sigmoid(Rc<UnrealizedOp>, Uuid),
    Relu(Rc<UnrealizedOp>, Uuid),
    Sum(Rc<UnrealizedOp>, Option<Vec<usize>>, bool, Uuid),
    Pool2D(Rc<UnrealizedOp>, (usize, usize), usize, f64, PoolOp, Uuid),
    Conv2D(
        Rc<UnrealizedOp>,
        Rc<UnrealizedOp>,
        Option<(usize, usize)>,
        Option<usize>,
        Uuid,
    ),
    Pad2D(Rc<UnrealizedOp>, f64, [usize; 4], Uuid),
    Reshape(Rc<UnrealizedOp>, Vec<usize>, Uuid),
    Permute(Rc<UnrealizedOp>, Vec<usize>, Uuid),
    Expand(Rc<UnrealizedOp>, Vec<usize>, Uuid),
    MatMul(Rc<UnrealizedOp>, Rc<UnrealizedOp>, Uuid),
}

impl Debug for UnrealizedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnrealizedOp::Add(_, _, uuid) => write!(f, "Add {uuid}"),
            UnrealizedOp::Sub(_, _, uuid) => write!(f, "Sub {uuid}"),
            UnrealizedOp::Mul(_, _, uuid) => write!(f, "Mul {uuid}"),
            UnrealizedOp::Div(_, _, uuid) => write!(f, "Div {uuid}"),
            UnrealizedOp::Max(_, uuid) => write!(f, "Max {uuid}"),
            UnrealizedOp::Min(_, uuid) => write!(f, "Min {uuid}"),
            UnrealizedOp::Sqrt(_, uuid) => write!(f, "Sqrt {uuid}"),
            UnrealizedOp::Log(_, uuid) => write!(f, "Log {uuid}"),
            UnrealizedOp::Load(_, _, uuid) => write!(f, "Load {uuid}"),
            UnrealizedOp::Sigmoid(_, uuid) => write!(f, "Sigmoid {uuid}"),
            UnrealizedOp::Relu(_, uuid) => write!(f, "Relu {uuid}"),
            UnrealizedOp::Sum(_, _, _, uuid) => write!(f, "Sum {uuid}"),
            UnrealizedOp::Pool2D(_, _, _, _, _, uuid) => write!(f, "Pool2D {uuid}"),
            UnrealizedOp::Conv2D(_, _, _, _, uuid) => write!(f, "Conv2D {uuid}"),
            UnrealizedOp::Pad2D(_, _, _, uuid) => write!(f, "Pad2D {uuid}"),
            UnrealizedOp::Reshape(_, _, uuid) => write!(f, "Reshape {uuid}"),
            UnrealizedOp::Permute(_, _, uuid) => write!(f, "Permute {uuid}"),
            UnrealizedOp::Expand(_, _, uuid) => write!(f, "Expand {uuid}"),
            UnrealizedOp::MatMul(_, _, uuid) => write!(f, "MatMul {uuid}"),
        }
    }
}

impl Display for UnrealizedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnrealizedOp::Add(lhs, rhs, _) => write!(f, "Add {lhs} {rhs}"),
            UnrealizedOp::Sub(lhs, rhs, _) => write!(f, "Sub {lhs} {rhs}"),
            UnrealizedOp::Mul(lhs, rhs, _) => write!(f, "Mul {lhs} {rhs}"),
            UnrealizedOp::Div(lhs, rhs, _) => write!(f, "Div {lhs} {rhs}"),
            UnrealizedOp::Max(t, _) => write!(f, "Max {t}"),
            UnrealizedOp::Min(t, _) => write!(f, "Min {t}"),
            UnrealizedOp::Sqrt(t, _) => write!(f, "Sqrt {t}"),
            UnrealizedOp::Log(t, _) => write!(f, "Log {t}"),
            UnrealizedOp::Load(_, shape, _) => write!(f, "Load {shape:?}"),
            UnrealizedOp::Sigmoid(t, _) => write!(f, "Sigmoid {t}"),
            UnrealizedOp::Relu(t, _) => write!(f, "Relu {t}"),
            UnrealizedOp::Sum(t, _, _, _) => write!(f, "Sum {t}"),
            UnrealizedOp::Pool2D(t, _, _, _, _, _) => write!(f, "Pool2D {t}"),
            UnrealizedOp::Conv2D(t, k, _, _, _) => write!(f, "Conv2D {t} {k}"),
            UnrealizedOp::Pad2D(t, _, _, _) => write!(f, "Pad2D {t}"),
            UnrealizedOp::Reshape(t, _, _) => write!(f, "Reshape {t}"),
            UnrealizedOp::Permute(t, _, _) => write!(f, "Permute {t}"),
            UnrealizedOp::Expand(t, _, _) => write!(f, "Expand {t}"),
            UnrealizedOp::MatMul(lhs, rhs, _) => write!(f, "MatMul {lhs} {rhs}"),
        }
    }
}

pub type PoolOp = fn(lhs: f64, rhs: f64) -> f64;

/*
*
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
* 0 0 0 0 0 0 0 0 0 0
*
* Let's assume kernel (2, 2)
* What if we turn [10, 10] into [25, 2, 2]
* Then we sum over the last two axis
*
* Then map [25] into [9, 9]
*
*
*
*/
