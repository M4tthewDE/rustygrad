use image::{imageops::FilterType, io::Reader, DynamicImage};
use rustygrad::{efficientnet::Efficientnet, Tensor};
use tracing::info;

fn main() {
    // https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py
    // https://github.com/tinygrad/tinygrad/blob/master/examples/efficientnet.py
    tracing_subscriber::fmt::init();

    let efficientnet = Efficientnet::default();

    let img = Reader::open("examples/chicken.jpg")
        .unwrap()
        .decode()
        .unwrap();

    infer(efficientnet, img);
}

const BIAS: [f64; 3] = [0.485, 0.456, 0.406];
const SCALE: [f64; 3] = [0.229, 0.224, 0.225];

fn infer(mut model: Efficientnet, mut image: DynamicImage) {
    let aspect_ratio = image.width() as f64 / image.height() as f64;
    image = image.resize_exact(
        (224.0 * aspect_ratio.max(1.0)) as u32,
        (224.0 * (1.0 / aspect_ratio).max(1.0)) as u32,
        FilterType::Nearest,
    );

    let x0 = (image.width() - 224) / 2;
    let y0 = (image.height() - 224) / 2;
    image = image.crop_imm(x0, y0, 224, 224);

    let bias = Tensor::new(BIAS.to_vec(), vec![1, 3, 1, 1]);
    let scale = Tensor::new(SCALE.to_vec(), vec![1, 3, 1, 1]);

    let mut input = Tensor::from_image(image);
    input = input.permute(vec![2, 0, 1]);
    input = input / 255.0;
    input = input - bias;
    input = input / scale;
    let out = model.forward(input);
    let max = out.clone().max().data[0];
    let argmax = out
        .data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap();
    info!("{} {} LABEL", argmax, max);
}
