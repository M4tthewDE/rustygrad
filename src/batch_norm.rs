use crate::Tensor;

const EXPO_AVG_FACTOR: f64 = 0.0;

#[derive(Debug)]
pub struct BatchNorm2d {
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
    affine: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
impl BatchNorm2d {
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

pub struct BatchNorm2dBuilder {
    num_features: usize,
    eps: f64,
    affine: bool,
}

impl BatchNorm2dBuilder {
    pub fn new(num_features: usize) -> BatchNorm2dBuilder {
        BatchNorm2dBuilder {
            num_features,
            eps: 1e-5,
            affine: true,
        }
    }

    pub fn eps(mut self, eps: f64) -> BatchNorm2dBuilder {
        self.eps = eps;
        self
    }

    pub fn affine(mut self, affine: bool) -> BatchNorm2dBuilder {
        self.affine = affine;
        self
    }

    pub fn build(self) -> BatchNorm2d {
        let (weight, bias) = if self.affine {
            (
                Some(Tensor::ones(self.num_features)),
                Some(Tensor::zeros(self.num_features)),
            )
        } else {
            (None, None)
        };

        BatchNorm2d {
            running_mean: Tensor::zeros(self.num_features),
            running_var: Tensor::ones(self.num_features),
            eps: self.eps,
            affine: self.affine,
            weight,
            bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::batch_norm::BatchNorm2dBuilder;

    #[test]
    fn test_batchnorm2d() {
        for num_features in vec![4, 8, 16, 32] {
            let bn = BatchNorm2dBuilder::new(num_features).eps(1e-5).build();
        }
        todo!("https://github.com/tinygrad/tinygrad/blob/38554322659fbe7e19c3cc7052465645274db5b9/test/test_nn.py#L25");
    }
}
