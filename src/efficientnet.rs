// https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
// https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py

use std::time::Instant;

use tracing::{debug, info};

use crate::{
    batch_norm::{BatchNorm2d, BatchNorm2dBuilder},
    util, Callable, Tensor,
};

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
    pub pad: [usize; 4],
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
            [v0, v1, v0, v1]
        } else {
            [((kernel_size as f64 - 1.0) / 2.0).floor() as usize; 4]
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
        debug!("0: {}", util::argmax(&x));
        dbg!(&x.data[266398]);
        dbg!(&x.data[266509]);
        if let Some(expand_conv) = &self.expand_conv {
            x = self
                .bn0
                .clone()
                .unwrap()
                .forward(x.conv2d(expand_conv, None, None, None, None))
                .swish();
        }
        debug!("1: {}", util::argmax(&x));
        dbg!(&x.data[266398]);
        dbg!(&x.data[266509]);
        dbg!(
            &x.shape,
            &self.pad,
            self.strides,
            &self.depthwise_conv.shape
        );
        // This is massively wrong!
        x = x.conv2d(
            &self.depthwise_conv,
            None,
            Some(&self.pad.clone()),
            Some(self.strides),
            Some(self.depthwise_conv.shape[0]),
        );
        dbg!(&x.shape);
        dbg!(&x.data[266398]);
        dbg!(&x.data[266509]);
        debug!("2: {}", util::argmax(&x));
        x = self.bn1.clone().forward(x).swish();
        debug!("3: {}", util::argmax(&x));

        let mut x_squeezed = x.avg_pool2d((x.shape[2], x.shape[3]), None);
        debug!("4: {}", util::argmax(&x));
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
            .forward(x.conv2d(&self.project_conv, None, None, None, None));

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
    conv_head: Tensor,
    bn0: BatchNorm2d,
    bn1: BatchNorm2d,
    fc: Tensor,
    fc_bias: Tensor,
}

impl Default for Efficientnet {
    fn default() -> Self {
        let number = 0;
        let global_params = get_global_params(number);
        let blocks_args = BLOCKS_ARGS.map(BlockArgs::from_tuple).to_vec();

        let out_channels = round_filters(32., global_params.width_coefficient);
        let mut bn0 = BatchNorm2dBuilder::new(out_channels).build();

        let mut blocks: Vec<MBConvBlock> = Vec::new();
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
                blocks.push(MBConvBlock::new(
                    block_arg.kernel_size,
                    strides,
                    block_arg.expand_ratio,
                    input_filters,
                    filters.1,
                    block_arg.se_ratio,
                ));

                strides = (1, 1);
                input_filters = filters.1;
            }
        }

        let out_channels = round_filters(1280.0, global_params.width_coefficient);
        let mut bn1 = BatchNorm2dBuilder::new(out_channels).build();

        let start = Instant::now();
        info!("loading model, this might take a while...");

        let model_data = util::load_torch_model(MODEL_URLS[number]).unwrap();
        bn0.weight = Some(Tensor::from_vec(
            util::extract_floats(&model_data["_bn0.weight"]).unwrap(),
        ));
        bn0.bias = Some(Tensor::from_vec(
            util::extract_floats(&model_data["_bn0.bias"]).unwrap(),
        ));
        bn0.running_mean =
            Tensor::from_vec(util::extract_floats(&model_data["_bn0.running_mean"]).unwrap());
        bn0.running_var =
            Tensor::from_vec(util::extract_floats(&model_data["_bn0.running_var"]).unwrap());
        bn1.weight = Some(Tensor::from_vec(
            util::extract_floats(&model_data["_bn1.weight"]).unwrap(),
        ));
        bn1.bias = Some(Tensor::from_vec(
            util::extract_floats(&model_data["_bn1.bias"]).unwrap(),
        ));
        bn1.running_mean =
            Tensor::from_vec(util::extract_floats(&model_data["_bn1.running_mean"]).unwrap());
        bn1.running_var =
            Tensor::from_vec(util::extract_floats(&model_data["_bn1.running_var"]).unwrap());

        let conv_head = util::extract_4d_tensor(&model_data["_conv_head.weight"]).unwrap();
        let conv_stem = util::extract_4d_tensor(&model_data["_conv_stem.weight"]).unwrap();
        // NOTE: is this permute correct? tinygrad changes the shape at some poing, unsure where
        let fc = util::extract_2d_tensor(&model_data["_fc.weight"])
            .unwrap()
            .permute(vec![1, 0]);
        let fc_bias = Tensor::from_vec(util::extract_floats(&model_data["_fc.bias"]).unwrap());

        for (i, block) in blocks.iter_mut().enumerate() {
            block.depthwise_conv = util::extract_4d_tensor(
                &model_data[format!("_blocks.{}._depthwise_conv.weight", i)],
            )
            .unwrap();
            block.project_conv =
                util::extract_4d_tensor(&model_data[format!("_blocks.{}._project_conv.weight", i)])
                    .unwrap();

            block.se_reduce =
                util::extract_4d_tensor(&model_data[format!("_blocks.{}._se_reduce.weight", i)])
                    .unwrap();
            block.se_reduce_bias = Tensor::new(
                util::extract_floats(&model_data[format!("_blocks.{}._se_reduce.bias", i)])
                    .unwrap(),
                block.se_reduce_bias.clone().shape,
            );

            block.se_expand =
                util::extract_4d_tensor(&model_data[format!("_blocks.{}._se_expand.weight", i)])
                    .unwrap();
            block.se_expand_bias = Tensor::new(
                util::extract_floats(&model_data[format!("_blocks.{}._se_expand.bias", i)])
                    .unwrap(),
                block.se_expand_bias.clone().shape,
            );

            for j in 0..3 {
                // this works right?
                let bn = match j {
                    0 => {
                        if block.bn0.is_none() {
                            continue;
                        }
                        block.expand_conv = Some(
                            util::extract_4d_tensor(
                                &model_data[format!("_blocks.{}._expand_conv.weight", i)],
                            )
                            .unwrap(),
                        );

                        block.bn0.as_mut().unwrap()
                    }
                    1 => &mut block.bn1,
                    2 => &mut block.bn2,
                    _ => panic!(),
                };

                bn.weight = Some(Tensor::from_vec(
                    util::extract_floats(&model_data[format!("_blocks.{}._bn{}.weight", i, j)])
                        .unwrap(),
                ));
                bn.bias = Some(Tensor::from_vec(
                    util::extract_floats(&model_data[format!("_blocks.{}._bn{}.bias", i, j)])
                        .unwrap(),
                ));
                bn.running_mean = Tensor::from_vec(
                    util::extract_floats(
                        &model_data[format!("_blocks.{}._bn{}.running_mean", i, j)],
                    )
                    .unwrap(),
                );
                bn.running_var = Tensor::from_vec(
                    util::extract_floats(
                        &model_data[format!("_blocks.{}._bn{}.running_var", i, j)],
                    )
                    .unwrap(),
                );
            }
        }

        info!("loaded model in {:?}", start.elapsed());

        let blocks = blocks
            .into_iter()
            .map(|b| Box::new(b) as Box<dyn Callable>)
            .collect::<Vec<Box<dyn Callable>>>();

        Self {
            global_params,
            blocks_args,
            blocks,
            conv_stem,
            conv_head,
            bn0,
            bn1,
            fc,
            fc_bias,
        }
    }
}

impl Efficientnet {
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        debug!("-2: {}", util::argmax(&x));
        let mut x = self
            .bn0
            .forward(x.conv2d(
                &self.conv_stem,
                None,
                Some(&[0, 1, 0, 1]),
                Some((2, 2)),
                None,
            ))
            .swish();

        debug!("-1: {}", util::argmax(&x));
        x = x.sequential(&self.blocks);
        x = self
            .bn1
            .clone()
            .forward(x.conv2d(&self.conv_head, None, None, None, None))
            .swish();
        x = x.avg_pool2d((x.shape[2], x.shape[3]), None);
        x = x.reshape(vec![1, x.shape[1]]);
        x.linear(&self.fc, Some(&self.fc_bias))
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
