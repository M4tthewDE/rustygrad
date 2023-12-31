use std::collections::HashMap;

use image::{imageops::FilterType, io::Reader, DynamicImage};
use rustygrad::{efficientnet::Efficientnet, util, Tensor};
use tracing::info;

fn main() {
    // https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py
    // https://github.com/tinygrad/tinygrad/blob/master/examples/efficientnet.py
    tracing_subscriber::fmt::init();

    let efficientnet = Efficientnet::default();

    let img = Reader::open("examples/chicken_cropped.jpg")
        .unwrap()
        .decode()
        .unwrap();

    infer(efficientnet, img);
}

const BIAS: [f64; 3] = [0.485, 0.456, 0.406];
const SCALE: [f64; 3] = [0.229, 0.224, 0.225];

fn infer(mut model: Efficientnet, mut image: DynamicImage) {
    let labels = load_labels();

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
    let argmax = util::argmax(&out);

    info!("{} {} {}", argmax, max, labels.get(&argmax).unwrap());
}

fn load_labels() -> HashMap<usize, String> {
    let mut resp = reqwest::blocking::get(
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw",
    )
    .unwrap()
    .text()
    .unwrap();

    resp = resp.replace("'", "");
    let mut result = HashMap::new();
    resp = resp[1..resp.len() - 1].to_string();

    for line in resp.lines() {
        let parts: Vec<&str> = line.split(':').take(2).collect();
        result.insert(
            parts[0].trim().parse::<usize>().unwrap(),
            parts[1][..parts[1].len() - 1].trim().to_string(),
        );
    }

    result
}
