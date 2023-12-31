use std::ops::Add;

use crate::Tensor;

#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    eps: f64,
    // FIXME: these should probably not be Option
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
// TODO: change this to tinygrad batchnorm implementation?
// seems more elegant and not too hard
// https://github.com/tinygrad/tinygrad/blob/master/tinygrad/nn/__init__.py
impl BatchNorm2d {
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        let batch_mean = self.running_mean.reshape(vec![1, self.num_features, 1, 1]);
        let batch_invstd = self
            .running_var
            .reshape(vec![1, self.num_features, 1, 1])
            .expand(&x.shape)
            .add(self.eps)
            .rsqrt();

        x.batchnorm(
            self.weight.as_ref(),
            self.bias.as_ref(),
            &batch_mean,
            &batch_invstd,
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

pub const INPUT: [f64; 72] = [
    7.2550803e-01,
    2.6468736e-01,
    1.6042733e-01,
    9.1248131e-01,
    1.6140634e-01,
    2.5769627e-01,
    4.6988237e-01,
    4.1404659e-01,
    4.2651403e-01,
    6.2887025e-01,
    2.9633635e-01,
    8.1406182e-01,
    3.3209044e-01,
    2.3387557e-01,
    7.6986372e-02,
    7.9895943e-01,
    4.6219945e-01,
    5.4436874e-01,
    5.0352424e-01,
    7.4825037e-01,
    9.6039361e-01,
    4.1384923e-01,
    6.5281290e-01,
    4.5812815e-01,
    4.9625474e-01,
    8.5442269e-01,
    2.3508650e-01,
    8.1202835e-01,
    5.9161729e-01,
    8.7563610e-01,
    7.8307271e-01,
    2.7927721e-01,
    5.9114403e-01,
    5.6509542e-01,
    6.7465717e-01,
    5.4024965e-01,
    6.9285572e-01,
    7.9642236e-01,
    3.3424616e-02,
    3.6928445e-01,
    3.0962038e-01,
    5.8036125e-01,
    6.5488166e-01,
    2.0787734e-01,
    3.7388825e-01,
    4.8392904e-01,
    8.8160717e-01,
    1.0005343e-01,
    6.3134187e-01,
    9.7084045e-04,
    9.5077854e-01,
    5.2091897e-02,
    8.1263256e-01,
    2.0272434e-02,
    9.7554046e-01,
    4.3832892e-01,
    3.4956914e-01,
    5.9617567e-01,
    1.4704645e-02,
    5.1316148e-01,
    5.0387317e-01,
    4.3340671e-01,
    5.0901908e-01,
    9.7953200e-02,
    3.7877357e-01,
    3.9520144e-01,
    1.4133233e-01,
    1.1132270e-01,
    9.3504202e-01,
    8.0020428e-01,
    4.0364802e-01,
    2.0521462e-02,
];

#[cfg(test)]
mod tests {
    use std::f64::NAN;

    use crate::{
        batch_norm::{BatchNorm2dBuilder, INPUT},
        util, Tensor,
    };

    const WEIGHTS: [f64; 4] = [-0.23390397, 0.80346787, -0.8997578, -0.72561103];
    const BIAS: [f64; 4] = [0.05118884, -1.8861961, -0.48187035, 0.77148885];
    const RUNNING_MEAN: [f64; 4] = [-0.8936946, 0.88154835, 0.50745213, -0.63631004];
    const RUNNING_VAR: [f64; 4] = [1.9885837, 1.4461043, 0.68407005, -0.32585317];
    const OUTPUT_4: [f64; 72] = [
        -0.21738626,
        -0.14095052,
        -0.12365704,
        -0.24839929,
        -0.12381943,
        -0.13979091,
        -0.17498596,
        -0.16572455,
        -0.1677925,
        -2.0550203,
        -2.2771995,
        -1.9312866,
        -2.2533107,
        -2.318932,
        -2.423756,
        -1.941377,
        -2.1663797,
        -2.1114793,
        -0.4775974,
        -0.7438246,
        -0.9746063,
        -0.38004372,
        -0.6400022,
        -0.42821288,
        -0.4696892,
        -0.859325,
        -0.18557526,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        -0.21197027,
        -0.22914873,
        -0.10259125,
        -0.15829991,
        -0.1484035,
        -0.19331095,
        -0.20567155,
        -0.13152751,
        -0.15906353,
        -2.1518614,
        -1.8861568,
        -2.4083438,
        -2.053369,
        -2.4745448,
        -1.8399407,
        -2.440389,
        -1.9322414,
        -2.4616487,
        -0.9910839,
        -0.40667412,
        -0.3101161,
        -0.5783889,
        0.05416886,
        -0.4880813,
        -0.47797695,
        -0.40131947,
        -0.48357496,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
        NAN,
    ];

    #[test]
    fn test_batchnorm2d_no_training() {
        let num_features = 4;
        let mut bn = BatchNorm2dBuilder::new(num_features).eps(1e-5).build();
        bn.weight = Some(Tensor::from_vec(WEIGHTS.to_vec()));
        bn.bias = Some(Tensor::from_vec(BIAS.to_vec()));
        bn.running_mean = Tensor::from_vec(RUNNING_MEAN.to_vec());
        bn.running_var = Tensor::from_vec(RUNNING_VAR.to_vec());

        let input = Tensor::new(INPUT.to_vec(), vec![2, num_features, 3, 3]);
        let out = bn.forward(input);

        util::assert_aprox_eq_vec(out.data, OUTPUT_4.to_vec(), 1e-6);
    }
}
