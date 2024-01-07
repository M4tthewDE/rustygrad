// TODO: should there be Option in here ever?
// default values could be inserted instead in tensor.rs

use std::rc::Rc;

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
