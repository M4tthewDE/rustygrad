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
*
*/
