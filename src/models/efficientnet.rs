// https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
// https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py

use crate::nn::batch_norm::BatchNorm2d;
use crate::nn::batch_norm::BatchNorm2dBuilder;

use serde_json::Value;

use crate::{tensor::Tensor, util};

static MODEL_URL: &str= "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth";
const DEPTH_COEFFICIENT: f64 = 1.0;
const WIDTH_COEFFICIENT: f64 = 1.0;

type BlockArgsTuple = (usize, usize, [usize; 2], usize, usize, usize, f64);

static BLOCKS_ARGS: [BlockArgsTuple; 7] = [
    (1, 3, [1, 1], 1, 32, 16, 0.25),
    (2, 3, [2, 2], 6, 16, 24, 0.25),
    (2, 5, [2, 2], 6, 24, 40, 0.25),
    (3, 3, [2, 2], 6, 40, 80, 0.25),
    (3, 5, [1, 1], 6, 80, 112, 0.25),
    (4, 5, [2, 2], 6, 112, 192, 0.25),
    (1, 3, [1, 1], 6, 192, 320, 0.25),
];

#[derive(Clone)]
struct BlockArgs {
    num_repeat: usize,
    kernel_size: usize,
    stride: (usize, usize),
    expand_ratio: usize,
    input_filters: usize,
    output_filters: usize,
    se_ratio: f64,
}

impl BlockArgs {
    fn from_tuple(tuple: BlockArgsTuple) -> BlockArgs {
        BlockArgs {
            num_repeat: tuple.0,
            kernel_size: tuple.1,
            stride: tuple.2.into(),
            expand_ratio: tuple.3,
            input_filters: tuple.4,
            output_filters: tuple.5,
            se_ratio: tuple.6,
        }
    }
}

struct MBConvBlock {
    expand_conv: Option<Tensor>,
    bn0: Option<BatchNorm2d>,
    strides: (usize, usize),
    pad: [usize; 4],
    bn1: BatchNorm2d,
    depthwise_conv: Tensor,
    se_reduce: Tensor,
    se_reduce_bias: Tensor,
    se_expand: Tensor,
    se_expand_bias: Tensor,
    project_conv: Tensor,
    bn2: BatchNorm2d,
}

impl MBConvBlock {
    fn new(
        model_data: &Value,
        i: usize,
        block_arg: &BlockArgs,
        strides: (usize, usize),
        input_filters: usize,
        output_filters: usize,
    ) -> MBConvBlock {
        let oup = block_arg.expand_ratio * input_filters;

        let (expand_conv, bn0) = if block_arg.expand_ratio != 1 {
            (
                Some(
                    util::extract_4d_tensor(
                        &model_data[format!("_blocks.{}._expand_conv.weight", i)],
                    )
                    .unwrap(),
                ),
                Some(
                    BatchNorm2dBuilder::new(oup)
                        .weight(
                            util::extract_1d_tensor(
                                &model_data[format!("_blocks.{}._bn0.weight", i)],
                            )
                            .unwrap(),
                        )
                        .bias(
                            util::extract_1d_tensor(
                                &model_data[format!("_blocks.{}._bn0.bias", i)],
                            )
                            .unwrap(),
                        )
                        .running_mean(
                            util::extract_1d_tensor(
                                &model_data[format!("_blocks.{}._bn0.running_mean", i)],
                            )
                            .unwrap(),
                        )
                        .running_var(
                            util::extract_1d_tensor(
                                &model_data[format!("_blocks.{}._bn0.running_var", i)],
                            )
                            .unwrap(),
                        )
                        .build(),
                ),
            )
        } else {
            (None, None)
        };

        let pad = if strides == (2, 2) {
            let v0 = ((block_arg.kernel_size as f64 - 1.0) / 2.0).floor() as usize - 1;
            let v1 = ((block_arg.kernel_size as f64 - 1.0) / 2.0).floor() as usize;
            [v0, v1, v0, v1]
        } else {
            [((block_arg.kernel_size as f64 - 1.0) / 2.0).floor() as usize; 4]
        };

        MBConvBlock {
            expand_conv,
            bn0,
            strides,
            pad,
            bn1: BatchNorm2dBuilder::new(oup)
                .weight(
                    util::extract_1d_tensor(&model_data[format!("_blocks.{}._bn1.weight", i)])
                        .unwrap(),
                )
                .bias(
                    util::extract_1d_tensor(&model_data[format!("_blocks.{}._bn1.bias", i)])
                        .unwrap(),
                )
                .running_mean(
                    util::extract_1d_tensor(
                        &model_data[format!("_blocks.{}._bn1.running_mean", i)],
                    )
                    .unwrap(),
                )
                .running_var(
                    util::extract_1d_tensor(&model_data[format!("_blocks.{}._bn1.running_var", i)])
                        .unwrap(),
                )
                .build(),
            depthwise_conv: util::extract_4d_tensor(
                &model_data[format!("_blocks.{}._depthwise_conv.weight", i)],
            )
            .unwrap(),
            se_reduce: util::extract_4d_tensor(
                &model_data[format!("_blocks.{}._se_reduce.weight", i)],
            )
            .unwrap(),
            se_reduce_bias: Tensor::from_vec(
                util::extract_floats(&model_data[format!("_blocks.{}._se_reduce.bias", i)])
                    .unwrap(),
                vec![
                    1,
                    ((input_filters as f64 * block_arg.se_ratio) as usize).max(1),
                    1,
                    1,
                ],
            ),
            se_expand: util::extract_4d_tensor(
                &model_data[format!("_blocks.{}._se_expand.weight", i)],
            )
            .unwrap(),
            se_expand_bias: Tensor::from_vec(
                util::extract_floats(&model_data[format!("_blocks.{}._se_expand.bias", i)])
                    .unwrap(),
                vec![1, oup, 1, 1],
            ),
            project_conv: util::extract_4d_tensor(
                &model_data[format!("_blocks.{}._project_conv.weight", i)],
            )
            .unwrap(),
            bn2: BatchNorm2dBuilder::new(output_filters)
                .weight(
                    util::extract_1d_tensor(&model_data[format!("_blocks.{}._bn2.weight", i)])
                        .unwrap(),
                )
                .bias(
                    util::extract_1d_tensor(&model_data[format!("_blocks.{}._bn2.bias", i)])
                        .unwrap(),
                )
                .running_mean(
                    util::extract_1d_tensor(
                        &model_data[format!("_blocks.{}._bn2.running_mean", i)],
                    )
                    .unwrap(),
                )
                .running_var(
                    util::extract_1d_tensor(&model_data[format!("_blocks.{}._bn2.running_var", i)])
                        .unwrap(),
                )
                .build(),
        }
    }
}

impl MBConvBlock {
    fn call(&self, input: &Tensor) -> Tensor {
        let x = if let (Some(expand_conv), Some(bn0)) = (&self.expand_conv, &self.bn0) {
            bn0.forward(&input.conv2d(expand_conv, None, None, None, None))
                .swish()
        } else {
            input.clone()
        };

        let shape = self.depthwise_conv.shape.clone();
        let x = x.conv2d(
            &self.depthwise_conv,
            None,
            Some(self.pad),
            Some(self.strides),
            Some(shape[0]),
        );
        let x = self.bn1.forward(&x).swish();

        let shape = x.shape.clone();
        let old_x = &x;
        let x_squeezed = x.avg_pool_2d((shape[2], shape[3]), None);
        let x_squeezed = x_squeezed
            .conv2d(
                &self.se_reduce,
                Some(&self.se_reduce_bias),
                None,
                None,
                None,
            )
            .swish();
        let x_squeezed = x_squeezed.conv2d(
            &self.se_expand,
            Some(&self.se_expand_bias),
            None,
            None,
            None,
        );

        let x = old_x * x_squeezed.sigmoid();
        let x = self
            .bn2
            .forward(&x.conv2d(&self.project_conv, None, None, None, None));

        if x.shape == input.shape {
            x + input
        } else {
            x
        }
    }
}

pub struct Efficientnet {
    blocks: Vec<MBConvBlock>,
    conv_stem: Tensor,
    conv_head: Tensor,
    bn0: BatchNorm2d,
    bn1: BatchNorm2d,
    fc: Tensor,
    fc_bias: Tensor,
}

impl Default for Efficientnet {
    fn default() -> Self {
        let blocks_args = BLOCKS_ARGS.map(BlockArgs::from_tuple).to_vec();
        let model_data = util::load_torch_model(MODEL_URL);

        let mut blocks: Vec<MBConvBlock> = Vec::new();
        let mut i = 0;
        for block_arg in &blocks_args {
            let mut input_filters = round_filters(block_arg.input_filters);
            let output_filters = round_filters(block_arg.output_filters);

            let mut strides = block_arg.stride;
            for _ in 0..((DEPTH_COEFFICIENT * block_arg.num_repeat as f64).ceil() as usize) {
                blocks.push(MBConvBlock::new(
                    &model_data,
                    i,
                    block_arg,
                    strides,
                    input_filters,
                    output_filters,
                ));

                strides = (1, 1);
                input_filters = output_filters;
                i += 1;
            }
        }

        Self {
            blocks,
            conv_stem: util::extract_4d_tensor(&model_data["_conv_stem.weight"]).unwrap(),
            conv_head: util::extract_4d_tensor(&model_data["_conv_head.weight"]).unwrap(),
            bn0: BatchNorm2dBuilder::new(round_filters(32))
                .weight(util::extract_1d_tensor(&model_data["_bn0.weight"]).unwrap())
                .bias(util::extract_1d_tensor(&model_data["_bn0.bias"]).unwrap())
                .running_mean(util::extract_1d_tensor(&model_data["_bn0.running_mean"]).unwrap())
                .running_var(util::extract_1d_tensor(&model_data["_bn0.running_var"]).unwrap())
                .build(),
            bn1: BatchNorm2dBuilder::new(round_filters(1280))
                .weight(util::extract_1d_tensor(&model_data["_bn1.weight"]).unwrap())
                .bias(util::extract_1d_tensor(&model_data["_bn1.bias"]).unwrap())
                .running_mean(util::extract_1d_tensor(&model_data["_bn1.running_mean"]).unwrap())
                .running_var(util::extract_1d_tensor(&model_data["_bn1.running_var"]).unwrap())
                .build(),
            fc: util::extract_2d_tensor(&model_data["_fc.weight"])
                .unwrap()
                .permute(vec![1, 0]),
            fc_bias: util::extract_1d_tensor(&model_data["_fc.bias"]).unwrap(),
        }
    }
}

impl Efficientnet {
    pub fn forward(&mut self, x: Tensor) -> Tensor {
        let mut x = self
            .bn0
            .forward(&x.conv2d(
                &self.conv_stem,
                None,
                Some([0, 1, 0, 1]),
                Some((2, 2)),
                None,
            ))
            .swish();

        for block in &self.blocks {
            x = block.call(&x);
        }

        let x = self
            .bn1
            .forward(&x.conv2d(&self.conv_head, None, None, None, None))
            .swish();
        let shape = x.shape.clone();
        let x = x.avg_pool_2d((shape[2], shape[3]), None);
        let shape = x.shape.clone();
        let x = x.reshape(vec![1, shape[1]]);
        x.linear(&self.fc, Some(&self.fc_bias))
    }
}

const DIVISOR: f64 = 8.0;

fn round_filters(filters: usize) -> usize {
    let filters = filters as f64 * WIDTH_COEFFICIENT;
    let new_filters = DIVISOR.max(((filters + DIVISOR / 2.) / DIVISOR).floor() * DIVISOR);

    if new_filters < 0.9 * filters {
        (new_filters + DIVISOR) as usize
    } else {
        new_filters as usize
    }
}
