use crate::Tensor;

const EXPO_AVG_FACTOR: f64 = 0.0;

pub struct BatchNorm2d {
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
    affine: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    _num_batches_tracked: usize,
}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
impl BatchNorm2d {
    pub fn new(num_features: usize, affine: Option<bool>) -> BatchNorm2d {
        let (weight, bias) = if let Some(affine) = affine {
            if affine {
                (
                    Some(Tensor::ones(num_features)),
                    Some(Tensor::zeros(num_features)),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        BatchNorm2d {
            running_mean: Tensor::zeros(num_features),
            running_var: Tensor::ones(num_features),
            eps: 1e-5,
            affine: true,
            weight,
            bias,
            _num_batches_tracked: 0,
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
            input = input * self.weight.clone().unwrap() + self.bias.clone().unwrap();
        }

        input
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_batchnorm2d() {
        todo!("https://github.com/tinygrad/tinygrad/blob/38554322659fbe7e19c3cc7052465645274db5b9/test/test_nn.py#L25");
    }
}
