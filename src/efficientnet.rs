use crate::util;
use anyhow::Result;

// https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

static MODEL_URLS: [&str; 8] = [
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth"
];

static PARAMS: [(f64, f64, i64, f64); 8] = [
    (1.0, 1.0, 224, 0.2),
    (1.0, 1.1, 240, 0.2),
    (1.1, 1.2, 260, 0.3),
    (1.2, 1.4, 300, 0.3),
    (1.4, 1.8, 380, 0.4),
    (1.6, 2.2, 456, 0.4),
    (1.8, 2.6, 528, 0.5),
    (2.0, 3.1, 600, 0.5),
];

pub struct GlobalParams {
    pub width_coefficient: f64,
    pub depth_coefficient: f64,
    pub image_size: i64,
    pub dropout_rate: f64,
    pub num_classes: usize,
    pub batch_norm_momentum: f64,
    pub batch_norm_epsilon: f64,
    pub drop_connect_rate: f64,
    pub depth_divisor: usize,
    pub include_top: bool,
}

pub struct Efficientnet {
    pub blocks_args: BlocksArgs,
    pub global_params: GlobalParams,
}

impl Default for Efficientnet {
    fn default() -> Self {
        let (global_params, blocks_args) = get_model_params(0).unwrap();

        Self {
            blocks_args,
            global_params,
        }
    }
}

type BlocksArgs = (f64, f64);

fn get_model_params(number: usize) -> Result<(GlobalParams, BlocksArgs)> {
    let _global_params = get_global_params(number);
    let _model_data = util::load_torch_model(MODEL_URLS[number])?;

    todo!("do whatever it takes");
}

fn get_global_params(number: usize) -> GlobalParams {
    let (width, depth, res, dropout) = PARAMS[number];

    GlobalParams {
        width_coefficient: width,
        depth_coefficient: depth,
        image_size: res,
        dropout_rate: dropout,
        num_classes: 1000,
        batch_norm_momentum: 0.99,
        batch_norm_epsilon: 1e-3,
        drop_connect_rate: 0.2,
        depth_divisor: 8,
        include_top: true,
    }
}
