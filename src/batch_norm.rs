use crate::Tensor;

pub struct BatchNorm2d {}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
impl BatchNorm2d {
    pub fn new(_sz: usize) -> Self {
        todo!("BatchNorm2d");
    }

    pub fn forward(input: Tensor, training: bool) {
        if training {
            let _mean = input.reduce_mean(Some(vec![0, 2, 3]), false, None);
            let _var = input.variance(Some(vec![0, 2, 3]));
            let _n = input.numel() / input.size(Some(1)).first().unwrap();
        }
    }
}
