use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum UnrealizedOp {
    Add(Box<Tensor>, Box<Tensor>),
    Sub(Box<Tensor>, Box<Tensor>),
    Mul(Box<Tensor>, Box<Tensor>),
    Div(Box<Tensor>, Box<Tensor>),
    Max(Box<Tensor>),
    Min(Box<Tensor>),
    Load(Vec<f64>, Vec<usize>),
}
