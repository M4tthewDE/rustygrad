// TODO: should there be Option in here ever?
// default values could be inserted instead in tensor.rs

use std::{fmt::Display, rc::Rc};

#[derive(Debug, Clone)]
pub enum UnrealizedOp {
    Add(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Sub(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Mul(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Div(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Max(Rc<UnrealizedOp>),
    Min(Rc<UnrealizedOp>),
    Sqrt(Rc<UnrealizedOp>),
    Log(Rc<UnrealizedOp>),
    Load(Vec<f64>, Vec<usize>),
    Sigmoid(Rc<UnrealizedOp>),
    Relu(Rc<UnrealizedOp>),
    Sum(Rc<UnrealizedOp>, Option<Vec<usize>>, bool),
    Pool2D(Rc<UnrealizedOp>, (usize, usize), usize, f64, PoolOp),
    Conv2D(
        Rc<UnrealizedOp>,
        Rc<UnrealizedOp>,
        Option<(usize, usize)>,
        Option<usize>,
    ),
    Pad2D(Rc<UnrealizedOp>, f64, [usize; 4]),
    Reshape(Rc<UnrealizedOp>, Vec<usize>),
    Permute(Rc<UnrealizedOp>, Vec<usize>),
    Expand(Rc<UnrealizedOp>, Vec<usize>),
    MatMul(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
}

impl Display for UnrealizedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnrealizedOp::Add(_, _) => f.write_str("Add"),
            UnrealizedOp::Sub(_, _) => f.write_str("Sub"),
            UnrealizedOp::Mul(_, _) => f.write_str("Mul"),
            UnrealizedOp::Div(_, _) => f.write_str("Div"),
            UnrealizedOp::Max(_) => f.write_str("Max"),
            UnrealizedOp::Min(_) => f.write_str("Min"),
            UnrealizedOp::Sqrt(_) => f.write_str("Sqrt"),
            UnrealizedOp::Log(_) => f.write_str("Log"),
            UnrealizedOp::Load(_, _) => f.write_str("Load"),
            UnrealizedOp::Sigmoid(_) => f.write_str("Sigmoid"),
            UnrealizedOp::Relu(_) => f.write_str("Relu"),
            UnrealizedOp::Sum(_, _, _) => f.write_str("Sum"),
            UnrealizedOp::Pool2D(_, _, _, _, _) => f.write_str("Pool2D"),
            UnrealizedOp::Conv2D(_, _, _, _) => f.write_str("Conv2D"),
            UnrealizedOp::Pad2D(_, _, _) => f.write_str("Pad2D"),
            UnrealizedOp::Reshape(_, _) => f.write_str("Reshape"),
            UnrealizedOp::Permute(_, _) => f.write_str("Permute"),
            UnrealizedOp::Expand(_, _) => f.write_str("Expand"),
            UnrealizedOp::MatMul(_, _) => f.write_str("MatMul"),
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
