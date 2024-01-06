use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum UnrealizedOp {
    Add(Box<Tensor>, Box<Tensor>),
    Sub(Box<Tensor>, Box<Tensor>),
    Mul(Box<Tensor>, Box<Tensor>),
    Div(Box<Tensor>, Box<Tensor>),
    Max(Box<Tensor>),
    Min(Box<Tensor>),
    // this is sum_pool_2d, not generic sum!
    // FIXME: generic sum???
    Sum(Box<Tensor>, Option<Vec<usize>>, bool),
    Reshape(Box<Tensor>, Vec<usize>),
    Load(Vec<f64>, Vec<usize>),
}

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
