// TODO: should there be Option in here ever?
// default values could be inserted instead in tensor.rs

#[derive(Debug, Clone)]
pub enum UnrealizedOp {
    Add(Box<UnrealizedOp>, Box<UnrealizedOp>),
    Sub(Box<UnrealizedOp>, Box<UnrealizedOp>),
    Mul(Box<UnrealizedOp>, Box<UnrealizedOp>),
    Div(Box<UnrealizedOp>, Box<UnrealizedOp>),
    Max(Box<UnrealizedOp>),
    Min(Box<UnrealizedOp>),
    Sqrt(Box<UnrealizedOp>),
    Log(Box<UnrealizedOp>),
    Load(Vec<f64>, Vec<usize>),
    Sigmoid(Box<UnrealizedOp>),
    Relu(Box<UnrealizedOp>),
    Sum(Box<UnrealizedOp>, Option<Vec<usize>>, bool),
    Pool2D(Box<UnrealizedOp>, (usize, usize), usize, f64, PoolOp),
    Conv2D(
        Box<UnrealizedOp>,
        Box<UnrealizedOp>,
        Option<(usize, usize)>,
        Option<usize>,
    ),
    Pad2D(Box<UnrealizedOp>, f64, [usize; 4]),
    Reshape(Box<UnrealizedOp>, Vec<usize>),
    Permute(Box<UnrealizedOp>, Vec<usize>),
    Expand(Box<UnrealizedOp>, Vec<usize>),
    MatMul(Box<UnrealizedOp>, Box<UnrealizedOp>),
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
