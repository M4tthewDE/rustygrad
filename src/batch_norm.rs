use crate::Tensor;

const EXPO_AVG_FACTOR: f64 = 0.0;

pub struct BatchNorm2d {
    running_mean: Tensor,
    running_var: Tensor,
}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
impl BatchNorm2d {
    pub fn new(_sz: usize) -> Self {
        todo!("BatchNorm2d");
    }

    pub fn forward(&mut self, input: Tensor, training: bool) {
        if training {
            let mean = input.reduce_mean(Some(vec![0, 2, 3]), false, None);
            let var = input.variance(Some(vec![0, 2, 3]));
            let n = (input.numel() / input.size(Some(1)).first().unwrap()) as f64;

            self.running_mean =
                mean * EXPO_AVG_FACTOR + self.running_mean.clone() * (1.0 - EXPO_AVG_FACTOR);
            self.running_var = var * n * EXPO_AVG_FACTOR / (n - 1.0)
                + self.running_var.clone() * (1.0 - EXPO_AVG_FACTOR);
        }
    }
}
