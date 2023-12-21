use crate::Tensor;

const EXPO_AVG_FACTOR: f64 = 0.0;

pub struct BatchNorm2d {
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
    affine: bool,
    weight: Tensor,
    bias: Tensor,
}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
impl BatchNorm2d {
    pub fn new(_num_features: usize) -> BatchNorm2d {
        BatchNorm2d {
            running_mean: Tensor::empty(),
            running_var: Tensor::empty(),
            eps: 1e-5,
            affine: true,
            weight: Tensor::empty(),
            bias: Tensor::empty(),
        }
    }

    pub fn forward(&mut self, input: Tensor, training: bool) -> Tensor {
        let mean: Tensor;
        let var: Tensor;
        if training {
            mean = input.reduce_mean(Some(vec![0, 2, 3]), false, None);
            var = input.variance(Some(vec![0, 2, 3]));
            let n = (input.numel() / input.size(Some(1)).first().unwrap()) as f64;

            self.running_mean = mean.clone() * EXPO_AVG_FACTOR
                + self.running_mean.clone() * (1.0 - EXPO_AVG_FACTOR);
            self.running_var = var.clone() * n * EXPO_AVG_FACTOR / (n - 1.0)
                + self.running_var.clone() * (1.0 - EXPO_AVG_FACTOR);
        } else {
            mean = self.running_mean.clone();
            var = self.running_var.clone();
        }

        // FIXME: reshape mean and var
        let mut input = (input - mean) / (var + self.eps).sqrt();

        if self.affine {
            input = input * self.weight.clone() + self.bias.clone();
        }

        input
    }
}
