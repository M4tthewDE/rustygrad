// https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

use crate::{
    batch_norm::{BatchNorm2d, BatchNorm2dBuilder},
    Callable, Tensor,
};

pub static MODEL_URLS: [&str; 8] = [
      "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth", "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
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

type BlockArgsTuple = (usize, usize, [usize; 2], usize, usize, usize, f64, bool);

static BLOCKS_ARGS: [BlockArgsTuple; 7] = [
    (1, 3, [1, 1], 1, 32, 16, 0.25, true),
    (2, 3, [2, 2], 6, 16, 24, 0.25, true),
    (2, 5, [2, 2], 6, 24, 40, 0.25, true),
    (3, 3, [2, 2], 6, 40, 80, 0.25, true),
    (3, 5, [1, 1], 6, 80, 112, 0.25, true),
    (4, 5, [2, 2], 6, 112, 192, 0.25, true),
    (1, 3, [1, 1], 6, 192, 320, 0.25, true),
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
    pub stride: (usize, usize),
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
            stride: tuple.2.into(),
            expand_ratio: tuple.3,
            input_filters: tuple.4,
            output_filters: tuple.5,
            se_ratio: tuple.6,
            id_skip: tuple.7,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MBConvBlock {
    pub expand_conv: Option<Tensor>,
    pub bn0: Option<BatchNorm2d>,
    pub strides: (usize, usize),
    pub pad: Vec<usize>,
    pub bn1: BatchNorm2d,
    pub depthwise_conv: Tensor,
    pub se_reduce: Tensor,
    pub se_reduce_bias: Tensor,
    pub se_expand: Tensor,
    pub se_expand_bias: Tensor,
    pub project_conv: Tensor,
    pub bn2: BatchNorm2d,
}

impl MBConvBlock {
    pub fn new(
        kernel_size: usize,
        strides: (usize, usize),
        expand_ratio: usize,
        input_filters: usize,
        output_filters: usize,
        se_ratio: f64,
    ) -> MBConvBlock {
        let oup = expand_ratio * input_filters;

        let (expand_conv, bn0) = if expand_ratio != 1 {
            (
                Some(Tensor::glorot_uniform(vec![oup, input_filters, 1, 1])),
                Some(BatchNorm2dBuilder::new(oup).build()),
            )
        } else {
            (None, None)
        };

        let pad = if strides == (2, 2) {
            let v0 = ((kernel_size as f64 - 1.0) / 2.0).floor() as usize - 1;
            let v1 = ((kernel_size as f64 - 1.0) / 2.0).floor() as usize;
            vec![v0, v1, v0, v1]
        } else {
            vec![((kernel_size as f64 - 1.0) / 2.0).floor() as usize; 4]
        };

        let depthwise_conv = Tensor::glorot_uniform(vec![oup, 1, kernel_size, kernel_size]);
        let bn1 = BatchNorm2dBuilder::new(oup).build();

        // we always have se!
        let num_squeezed_channels = ((input_filters as f64 * se_ratio) as usize).max(1);
        let se_reduce = Tensor::glorot_uniform(vec![num_squeezed_channels, oup, 1, 1]);
        let se_reduce_bias = Tensor::new(
            vec![0.0; num_squeezed_channels],
            vec![1, num_squeezed_channels, 1, 1],
        );
        let se_expand = Tensor::glorot_uniform(vec![oup, num_squeezed_channels, 1, 1]);
        let se_expand_bias = Tensor::new(vec![0.0; oup], vec![1, oup, 1, 1]);

        let project_conv = Tensor::glorot_uniform(vec![output_filters, oup, 1, 1]);
        let bn2 = BatchNorm2dBuilder::new(output_filters).build();

        MBConvBlock {
            expand_conv,
            bn0,
            strides,
            pad,
            bn1,
            depthwise_conv,
            se_reduce,
            se_reduce_bias,
            se_expand,
            se_expand_bias,
            project_conv,
            bn2,
        }
    }
}

impl Callable for MBConvBlock {
    fn call(&self, input: Tensor) -> Tensor {
        let mut x = input.clone();
        if let Some(expand_conv) = &self.expand_conv {
            x = self
                .bn0
                .clone()
                .unwrap()
                .forward(x.conv2d(expand_conv, None, None, None, None), false)
                .swish();
        }

        x = x.conv2d(
            &self.depthwise_conv,
            None,
            Some(self.pad.clone()),
            Some(self.strides),
            Some(self.depthwise_conv.shape[0]),
        );

        x = self.bn1.clone().forward(x, false).swish();

        let mut x_squeezed = x.avg_pool2d((x.shape[2], x.shape[3]), None);
        x_squeezed = x_squeezed
            .conv2d(
                &self.se_reduce,
                Some(&self.se_reduce_bias),
                None,
                None,
                None,
            )
            .swish();

        x_squeezed = x_squeezed.conv2d(
            &self.se_expand,
            Some(&self.se_expand_bias),
            None,
            None,
            None,
        );

        x = x * x_squeezed.sigmoid();
        x = self
            .bn2
            .clone()
            .forward(x.conv2d(&self.project_conv, None, None, None, None), false);

        if x.shape == input.shape {
            x = x + input;
        }

        x
    }
}

pub struct Efficientnet {
    pub global_params: GlobalParams,
    pub blocks_args: Vec<BlockArgs>,
    pub blocks: Vec<Box<dyn Callable>>,
    conv_stem: Tensor,
    bn0: BatchNorm2d,
}

impl Default for Efficientnet {
    fn default() -> Self {
        let number = 0;
        let input_channels = 3;
        let global_params = get_global_params(number);
        let blocks_args = BLOCKS_ARGS.map(BlockArgs::from_tuple).to_vec();

        let out_channels = round_filters(32., global_params.width_coefficient);
        let conv_stem = Tensor::glorot_uniform(vec![out_channels, input_channels, 3, 3]);
        let bn0 = BatchNorm2dBuilder::new(out_channels).build();

        let mut blocks: Vec<Box<dyn Callable>> = Vec::new();
        for block_arg in &blocks_args {
            let filters = (
                round_filters(
                    block_arg.input_filters as f64,
                    global_params.width_coefficient,
                ),
                round_filters(
                    block_arg.output_filters as f64,
                    global_params.width_coefficient,
                ),
            );

            let mut input_filters = filters.0;

            let mut strides = block_arg.stride;
            for _ in 0..round_repeats(block_arg.num_repeat, global_params.depth_coefficient) {
                blocks.push(Box::new(MBConvBlock::new(
                    block_arg.kernel_size,
                    strides,
                    block_arg.expand_ratio,
                    input_filters,
                    filters.1,
                    block_arg.se_ratio,
                )));

                strides = (1, 1);
                input_filters = filters.1;
            }
        }

        Self {
            global_params,
            blocks_args,
            blocks,
            conv_stem,
            bn0,
        }
    }
}

impl Efficientnet {
    pub fn load_from_pretrained(&mut self) {
        //let _model_data = util::load_torch_model(MODEL_URLS[self.number]).unwrap();
        // TODO: use the data
    }

    pub fn forward(&mut self, x: Tensor) {
        let x = self
            .bn0
            .forward(
                x.conv2d(
                    &self.conv_stem,
                    None,
                    Some(vec![0, 0, 1, 1]),
                    Some((2, 2)),
                    None,
                ),
                false,
            )
            .swish();

        let _x = x.sequential(&self.blocks);
        todo!("remaining forward pass");
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
        ((filters + divisor / 2.) / divisor).floor() * divisor,
    );

    if new_filters < 0.9 * filters {
        new_filters += divisor;
    }

    new_filters as usize
}

fn round_repeats(repeats: usize, depth_coefficient: f64) -> usize {
    (depth_coefficient * repeats as f64).ceil() as usize
}
