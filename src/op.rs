use crate::tensor::Tensor;

// TODO: should there be Option in here ever?
// default values could be inserted instead in tensor.rs

#[derive(Debug, Clone)]
pub enum UnrealizedOp {
    Add(Box<Tensor>, Box<Tensor>),
    Sub(Box<Tensor>, Box<Tensor>),
    Mul(Box<Tensor>, Box<Tensor>),
    Div(Box<Tensor>, Box<Tensor>),
    Max(Box<Tensor>),
    Min(Box<Tensor>),
    Sqrt(Box<Tensor>),
    Log(Box<Tensor>),
    Load(Vec<f64>, Vec<usize>),
    Sigmoid(Box<Tensor>),
    Relu(Box<Tensor>),
    Sum(Box<Tensor>, Option<Vec<usize>>, bool),
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
