use std::ops::Add;

use crate::{tensor::Tensor, util};

#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    eps: f64,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

// https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/__init__.py
impl BatchNorm2d {
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        let expanded = self
            .running_var
            .clone()
            .reshape(vec![1, self.num_features, 1, 1])
            .expand(x.shape.clone());
        let (data, _) = expanded.clone().realize();
        let argmax = util::argmax(&data);
        let max = data
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("no min value found");
        println!("bn0 {argmax} {max}");
        let added = expanded.add(self.eps);
        let (data, _) = added.clone().realize();
        let argmax = util::argmax(&data);
        let max = data
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("no min value found");
        println!("bn1 {argmax} {max}");
        let batch_invstd = added.rsqrt();
        let (data, _) = batch_invstd.clone().realize();
        let argmax = util::argmax(&data);
        let max = data
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("no min value found");
        println!("bn2 {argmax} {max}");

        let test = x.batchnorm(
            self.weight.clone(),
            self.bias.clone(),
            self.running_mean.clone(),
            batch_invstd,
        );

        let (data, _) = test.clone().realize();
        let argmax = util::argmax(&data);
        let max = data
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("no min value found");
        println!("bn3 {argmax} {max}");
        test
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
            num_features: self.num_features,
            running_mean: Tensor::zeros(self.num_features),
            running_var: Tensor::ones(self.num_features),
            eps: self.eps,
            weight,
            bias,
        }
    }
}
