use std::{
    fmt::{Debug, Display},
    rc::Rc,
};

// TODO: should there be Option in here ever?
// default values could be inserted in tensor.rs instead

#[derive(Clone)]
pub enum UnrealizedOp {
    Add(Rc<UnrealizedOp>, Rc<UnrealizedOp>, usize),
    Sub(Rc<UnrealizedOp>, Rc<UnrealizedOp>, usize),
    Mul(Rc<UnrealizedOp>, Rc<UnrealizedOp>, usize),
    Div(Rc<UnrealizedOp>, Rc<UnrealizedOp>, usize),
    Max(Rc<UnrealizedOp>, usize),
    Min(Rc<UnrealizedOp>, usize),
    Sqrt(Rc<UnrealizedOp>, usize),
    Log(Rc<UnrealizedOp>, usize),
    Load(Vec<f64>, Vec<usize>, usize),
    Sigmoid(Rc<UnrealizedOp>, usize),
    Relu(Rc<UnrealizedOp>, usize),
    Sum(Rc<UnrealizedOp>, Option<Vec<usize>>, bool, usize),
    Pool2D(Rc<UnrealizedOp>, (usize, usize), usize, f64, PoolOp, usize),
    Conv2D(
        Rc<UnrealizedOp>,
        Rc<UnrealizedOp>,
        Option<(usize, usize)>,
        Option<usize>,
        usize,
    ),
    Pad2D(Rc<UnrealizedOp>, f64, [usize; 4], usize),
    Reshape(Rc<UnrealizedOp>, Vec<usize>, usize),
    Permute(Rc<UnrealizedOp>, Vec<usize>, usize),
    Expand(Rc<UnrealizedOp>, Vec<usize>, usize),
    MatMul(Rc<UnrealizedOp>, Rc<UnrealizedOp>, usize),
}

impl Debug for UnrealizedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnrealizedOp::Add(_, _, id) => write!(f, "Add {id}"),
            UnrealizedOp::Sub(_, _, id) => write!(f, "Sub {id}"),
            UnrealizedOp::Mul(_, _, id) => write!(f, "Mul {id}"),
            UnrealizedOp::Div(_, _, id) => write!(f, "Div {id}"),
            UnrealizedOp::Max(_, id) => write!(f, "Max {id}"),
            UnrealizedOp::Min(_, id) => write!(f, "Min {id}"),
            UnrealizedOp::Sqrt(_, id) => write!(f, "Sqrt {id}"),
            UnrealizedOp::Log(_, id) => write!(f, "Log {id}"),
            UnrealizedOp::Load(_, _, id) => write!(f, "Load {id}"),
            UnrealizedOp::Sigmoid(_, id) => write!(f, "Sigmoid {id}"),
            UnrealizedOp::Relu(_, id) => write!(f, "Relu {id}"),
            UnrealizedOp::Sum(_, _, _, id) => write!(f, "Sum {id}"),
            UnrealizedOp::Pool2D(_, _, _, _, _, id) => write!(f, "Pool2D {id}"),
            UnrealizedOp::Conv2D(_, _, _, _, id) => write!(f, "Conv2D {id}"),
            UnrealizedOp::Pad2D(_, _, _, id) => write!(f, "Pad2D {id}"),
            UnrealizedOp::Reshape(_, _, id) => write!(f, "Reshape {id}"),
            UnrealizedOp::Permute(_, _, id) => write!(f, "Permute {id}"),
            UnrealizedOp::Expand(_, _, id) => write!(f, "Expand {id}"),
            UnrealizedOp::MatMul(_, _, id) => write!(f, "MatMul {id}"),
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
