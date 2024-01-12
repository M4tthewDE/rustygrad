use std::ops::Add;

use crate::tensor::Tensor;

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
// FIXME: I don't like these clones in here...
impl BatchNorm2d {
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        let batch_invstd = self
            .running_var
            .clone()
            .reshape(vec![1, self.num_features, 1, 1])
            .expand(x.shape.clone())
            .add(self.eps)
            .rsqrt();

        x.batchnorm(
            self.weight.clone(),
            self.bias.clone(),
            self.running_mean.clone(),
            batch_invstd,
        )
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

#[cfg(test)]
mod tests {

    use crate::{nn::batch_norm::BatchNorm2dBuilder, tensor::Tensor, util};

    #[test]
    fn test_batchnorm2d_no_training() {
        let num_features = 4;
        let mut bn = BatchNorm2dBuilder::new(num_features).eps(1e-5).build();
        bn.weight = Some(Tensor::rand(vec![4]));
        bn.bias = Some(Tensor::rand(vec![4]));
        bn.running_mean = Tensor::rand(vec![4]);
        bn.running_var = Tensor::rand(vec![4]);

        let input = Tensor::rand(vec![2, num_features, 3, 3]);
        let tch_input = input.to_tch();
        let mut out = bn.forward(input);
        out.realize();
        let tch_weight = bn.weight.unwrap().to_tch();
        let tch_bias = bn.bias.unwrap().to_tch();
        let tch_running_mean = bn.running_mean.to_tch();
        let tch_running_var = bn.running_var.to_tch();
        let tch_out = tch_input.batch_norm(
            Some(tch_weight),
            Some(tch_bias),
            Some(tch_running_mean),
            Some(tch_running_var),
            false,
            0.1,
            1e-5,
            false,
        );
        let tch_shape = util::tch_shape(&tch_out);
        let tch_output = util::tch_data(&tch_out);

        assert_eq!(out.shape, tch_shape);
        util::assert_aprox_eq_vec(out.data.unwrap(), tch_output, 1e-6);
    }
}
