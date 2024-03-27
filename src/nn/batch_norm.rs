use std::ops::Add;

use crate::tensor::Tensor;

#[derive(Debug)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    eps: f64,
    pub weight: Tensor,
    pub bias: Tensor,
}

// https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/__init__.py
impl BatchNorm2d {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let expanded = self
            .running_var
            .reshape(vec![1, self.num_features, 1, 1])
            .expand(x.shape.clone());
        let batch_invstd = expanded.add(self.eps).rsqrt();

        x.batchnorm(
            Some(&self.weight),
            Some(&self.bias),
            &self.running_mean,
            batch_invstd,
        )
    }
}

pub struct BatchNorm2dBuilder {
    num_features: usize,
    eps: f64,
    weight: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
}

impl BatchNorm2dBuilder {
    pub fn new(num_features: usize) -> BatchNorm2dBuilder {
        BatchNorm2dBuilder {
            num_features,
            eps: 1e-5,
            weight: Tensor::ones(num_features),
            bias: Tensor::ones(num_features),
            running_mean: Tensor::zeros(num_features),
            running_var: Tensor::ones(num_features),
        }
    }

    pub fn eps(mut self, eps: f64) -> BatchNorm2dBuilder {
        self.eps = eps;
        self
    }

    pub fn weight(mut self, weight: Tensor) -> BatchNorm2dBuilder {
        self.weight = weight;
        self
    }

    pub fn bias(mut self, bias: Tensor) -> BatchNorm2dBuilder {
        self.bias = bias;
        self
    }

    pub fn running_mean(mut self, running_mean: Tensor) -> BatchNorm2dBuilder {
        self.running_mean = running_mean;
        self
    }

    pub fn running_var(mut self, running_var: Tensor) -> BatchNorm2dBuilder {
        self.running_var = running_var;
        self
    }

    pub fn build(self) -> BatchNorm2d {
        BatchNorm2d {
            num_features: self.num_features,
            running_mean: self.running_mean,
            running_var: self.running_var,
            eps: self.eps,
            weight: self.weight,
            bias: self.bias,
        }
    }
}
