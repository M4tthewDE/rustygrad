use tracing::trace;

use crate::op::Op;

pub fn realize(op: &Op) -> (Vec<f64>, Vec<usize>) {
    trace!("Realizing {:?}", op);
    match op {
        Op::Add(_, _) => todo!(),
        Op::Sub(_, _) => todo!(),
        Op::Mul(_, _) => todo!(),
        Op::Div(_, _) => todo!(),
        Op::Max(_) => todo!(),
        Op::Min(_) => todo!(),
        Op::Sqrt(_) => todo!(),
        Op::Log(_) => todo!(),
        Op::Load(_, _) => todo!(),
        Op::Sigmoid(_) => todo!(),
        Op::Relu(_) => todo!(),
        Op::Sum(_, _, _) => todo!(),
        Op::Pool2D(_, _, _, _, _) => todo!(),
        Op::Conv2D(_, _, _, _) => todo!(),
        Op::Pad2D(_, _, _) => todo!(),
        Op::Reshape(_, _) => todo!(),
        Op::Permute(_, _) => todo!(),
        Op::Expand(_, _) => todo!(),
        Op::MatMul(_, _) => todo!(),
    }
}
