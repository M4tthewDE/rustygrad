use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum Op {
    Add(Box<Tensor>),
    Sub(Box<Tensor>),
    Load(Vec<f64>, Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct UnrealizedOp {
    pub op: Op,
    pub next: Box<Option<UnrealizedOp>>,
}

impl UnrealizedOp {
    pub fn new(op: Op) -> UnrealizedOp {
        UnrealizedOp {
            op,
            next: Box::new(None),
        }
    }

    pub fn run(&self) -> Tensor {
        self.op.run()
    }
}
