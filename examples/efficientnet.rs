use std::{collections::HashMap, env, time::Instant};

use image::{imageops::FilterType, io::Reader, DynamicImage};
use rustygrad::{efficientnet::Efficientnet, tensor::Tensor, util};
use tracing::info;

// https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py
// https://github.com/tinygrad/tinygrad/blob/master/examples/efficientnet.py
fn main() {
    tracing_subscriber::fmt::init();

    let efficientnet = Efficientnet::default();
    let img_name = env::args().nth(1).unwrap();
    let img = Reader::open(img_name).unwrap().decode().unwrap();

    let start = Instant::now();
    infer(efficientnet, img);
    info!("did inference in {:?}", start.elapsed());
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
    let bias = Tensor::from_vec(BIAS.to_vec(), vec![1, 3, 1, 1]);
    let scale = Tensor::from_vec(SCALE.to_vec(), vec![1, 3, 1, 1]);

    let mut input = Tensor::from_image(image);
    input = input.permute(vec![2, 0, 1]);
    input = input / 255.0;
    input = input - bias;
    input = input / scale;
    let out = model.forward(input);
    let argmax = util::argmax(out.clone());
    let mut max = out.max();
    max.realize();
    let max = max.data.unwrap()[0];

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
