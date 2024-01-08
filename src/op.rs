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
