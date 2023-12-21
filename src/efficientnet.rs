// https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

use crate::{batch_norm::BatchNorm2d, Tensor};

pub static MODEL_URLS: [&str; 8] = [
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

type BlockArgsTuple = (usize, usize, usize, usize, usize, usize, f64, bool);

static BLOCKS_ARGS: [BlockArgsTuple; 7] = [
    (1, 3, 11, 1, 32, 16, 0.25, true),
    (2, 3, 22, 6, 16, 24, 0.25, true),
    (2, 5, 22, 6, 24, 40, 0.25, true),
    (3, 3, 22, 6, 40, 80, 0.25, true),
    (3, 5, 11, 6, 80, 112, 0.25, true),
    (4, 5, 22, 6, 112, 192, 0.25, true),
    (1, 3, 11, 6, 192, 320, 0.25, true),
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

#[derive(Clone)]
pub struct BlockArgs {
    pub num_repeat: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub expand_ratio: usize,
    pub input_filters: usize,
    pub output_filters: usize,
    pub se_ratio: f64,
    pub id_skip: bool,
}

impl BlockArgs {
    fn from_tuple(tuple: BlockArgsTuple) -> Self {
        Self {
            num_repeat: tuple.0,
            kernel_size: tuple.1,
            stride: tuple.2,
            expand_ratio: tuple.3,
            input_filters: tuple.4,
            output_filters: tuple.5,
            se_ratio: tuple.6,
            id_skip: tuple.7,
        }
    }
}

pub struct Efficientnet {
    pub global_params: GlobalParams,
    pub blocks_args: Vec<BlockArgs>,
}

impl Default for Efficientnet {
    fn default() -> Self {
        let global_params = get_global_params(0);
        let blocks_args = BLOCKS_ARGS.map(BlockArgs::from_tuple).to_vec();

        let out_channels = round_filters(32., global_params.width_coefficient);
        // NOTE: are we using the correct arguments?
        let _conv_stem = Tensor::glorot_uniform(3, out_channels, vec![3, 3]);
        let _bn0 = BatchNorm2d::new(out_channels);

        dbg!("TODO: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L183");

        Self {
            global_params,
            blocks_args,
        }
    }
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

fn round_filters(mut filters: f64, multiplier: f64) -> usize {
    let divisor = 8.0;
    filters *= multiplier;

    let mut new_filters = f64::max(
        divisor,
        ((filters + divisor / 2.) / (divisor * divisor)).floor(),
    );

    if new_filters < 0.9 * filters {
        new_filters += divisor;
    }

    new_filters as usize
}
