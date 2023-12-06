use crate::util;
use anyhow::Result;

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

pub struct Efficientnet {
    pub blocks_args: Vec<(f64, f64)>,
    pub global_params: (f64, f64),
}

impl Default for Efficientnet {
    fn default() -> Self {
        let (blocks_args, global_params) = get_model_params(0).unwrap();

        Self {
            blocks_args,
            global_params,
        }
    }
}

type GlobalParams = Vec<(f64, f64)>;
type BlocksArgs = (f64, f64);

fn get_model_params(number: usize) -> Result<(GlobalParams, BlocksArgs)> {
    let _model_data = util::load_torch_model(MODEL_URLS[number])?;
    todo!("extract blocks_args and global_params from json");
}
