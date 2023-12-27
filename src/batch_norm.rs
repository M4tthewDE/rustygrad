use crate::Tensor;

#[derive(Debug, Clone)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    eps: f64,
    affine: bool,
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    track_running_stats: bool,
    pub num_batches_tracked: usize,
    momentum: f64,
}

// https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
impl BatchNorm2d {
    pub fn forward(&mut self, input: Tensor, training: bool) -> Tensor {
        let mut mean: Tensor;
        let mut var: Tensor;

        if training && self.track_running_stats {
            self.num_batches_tracked += 1;
        }

        if training {
            mean = input.reduce_mean(Some(&vec![0, 2, 3]), false, None);
            var = input.variance(Some(&vec![0, 2, 3]), Some(0.0));

            let n = (input.numel() / input.size(Some(1)).first().unwrap()) as f64;
            self.running_mean =
                self.momentum * mean.clone() + (1.0 - self.momentum) * self.running_mean.clone();
            self.running_var = var.clone() * n * self.momentum / (n - 1.0)
                + self.running_var.clone() * (1.0 - self.momentum);
        } else {
            mean = self.running_mean.clone();
            var = self.running_var.clone();
        }

        // FIXME: if we use keepdim=true, we shouldn't have to do this here
        mean = mean.reshape(vec![1, self.num_features, 1, 1]);
        var = var.reshape(vec![1, self.num_features, 1, 1]);

        let mut input = (input - mean) / (var + self.eps).sqrt();

        let weight = self
            .weight
            .clone()
            .unwrap()
            .reshape(vec![1, self.num_features, 1, 1]);
        let bias = self
            .bias
            .clone()
            .unwrap()
            .reshape(vec![1, self.num_features, 1, 1]);

        if self.affine {
            input = input * weight + bias;
        }

        input
    }
}

pub struct BatchNorm2dBuilder {
    num_features: usize,
    eps: f64,
    affine: bool,
    track_running_stats: bool,
    num_batches_tracked: usize,
    momentum: f64,
}

impl BatchNorm2dBuilder {
    pub fn new(num_features: usize) -> BatchNorm2dBuilder {
        BatchNorm2dBuilder {
            num_features,
            eps: 1e-5,
            affine: true,
            track_running_stats: false,
            num_batches_tracked: 0,
            momentum: 0.1,
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

    pub fn track_running_stats(mut self, track_running_stats: bool) -> BatchNorm2dBuilder {
        self.track_running_stats = track_running_stats;
        self
    }

    pub fn num_batches_tracked(mut self, num_batches_tracked: usize) -> BatchNorm2dBuilder {
        self.num_batches_tracked = num_batches_tracked;
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
            affine: self.affine,
            weight,
            bias,
            track_running_stats: self.track_running_stats,
            num_batches_tracked: self.num_batches_tracked,
            momentum: self.momentum,
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

    const OUTPUT_4_TRAINING: [f64; 72] = [
        -0.23414426,
        0.21685188,
        0.3188891,
        -0.4171313,
        0.31793097,
        0.2236939,
        0.01603156,
        0.07067694,
        0.0584753,
        -1.433309,
        -2.2809026,
        -0.9612753,
        -2.1897693,
        -2.4401085,
        -2.8400025,
        -0.9997697,
        -1.8581352,
        -1.6486944,
        -0.35483286,
        -1.2984207,
        -2.116379,
        -0.0090739,
        -0.93044347,
        -0.17979965,
        -0.32680392,
        -1.7077881,
        0.6801796,
        -0.03903629,
        0.5331754,
        -0.20416911,
        0.03613578,
        1.344045,
        0.53440404,
        0.60202914,
        0.31759462,
        0.66653156,
        -0.20218807,
        -0.3035467,
        0.44318417,
        0.11448476,
        0.1728768,
        -0.09209195,
        -0.16502361,
        0.27245072,
        0.10997911,
        -1.8027488,
        -0.78910935,
        -2.781207,
        -1.4270091,
        -3.0337582,
        -0.6127988,
        -2.903456,
        -0.9649183,
        -2.9845605,
        -2.1747806,
        -0.10345996,
        0.23877016,
        -0.71206796,
        1.5299035,
        -0.39199105,
        -0.3561782,
        -0.08448144,
        -0.3760192,
        1.8147825,
        1.0857414,
        1.0430928,
        1.7021654,
        1.7800738,
        -0.35839352,
        -0.00833968,
        1.0211647,
        2.0158038,
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
        let out = bn.forward(input, false);

        util::assert_aprox_eq_vec(out.data, OUTPUT_4.to_vec(), 1e-6);
    }

    #[test]
    fn test_batchnorm2d_training() {
        let num_features = 4;
        let mut bn = BatchNorm2dBuilder::new(num_features)
            .eps(1e-5)
            .track_running_stats(true)
            .build();
        bn.weight = Some(Tensor::from_vec(WEIGHTS.to_vec()));
        bn.bias = Some(Tensor::from_vec(BIAS.to_vec()));
        bn.running_mean = Tensor::from_vec(RUNNING_MEAN.to_vec());
        bn.running_var = Tensor::from_vec(RUNNING_VAR.to_vec());

        let input = Tensor::new(INPUT.to_vec(), vec![2, num_features, 3, 3]);
        let out = bn.forward(input, true);

        util::assert_aprox_eq_vec(out.data, OUTPUT_4_TRAINING.to_vec(), 1e-6);
        util::assert_aprox_eq_vec(
            bn.running_mean.data,
            vec![-0.76092917, 0.83851254, 0.51035416, -0.522697],
            1e-6,
        );
        util::assert_aprox_eq_vec(
            bn.running_var.data,
            vec![1.7957723, 1.3120139, 0.62142795, -0.2849974],
            1e-6,
        );
    }
}
