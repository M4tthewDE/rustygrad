use image::{imageops::FilterType, io::Reader, DynamicImage};
use rustygrad::{efficientnet::Efficientnet, Tensor};

fn main() {
    // https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py
    // https://github.com/tinygrad/tinygrad/blob/master/examples/efficientnet.py
    tracing_subscriber::fmt::init();

    let efficientnet = Efficientnet::default();
    // FIXME: this is crazy slow
    //efficientnet.load_from_pretrained();

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

    // cropping towards the top left corner!
    // TODO: crop towards the center
    image = image.crop_imm(0, 0, 224, 224);

    let bias = Tensor::new(BIAS.to_vec(), vec![1, 3, 1, 1]);
    let scale = Tensor::new(SCALE.to_vec(), vec![1, 3, 1, 1]);

    let mut input = Tensor::from_image(image);
    input = input.permute(vec![2, 0, 1]);
    input = input / 255.0;
    input = input - bias;
    input = input / scale;
    model.forward(input);
}
