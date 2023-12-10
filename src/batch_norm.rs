use crate::Tensor;

pub struct BatchNorm2d {}

impl BatchNorm2d {
    pub fn new(_sz: usize) -> Self {
        todo!("BatchNorm2d");
    }

    pub fn run(x: Tensor, training: bool) {
        if training {
            let _batch_mean = x.reduce_mean(Some(vec![0, 2, 3]));
        }
    }
}
