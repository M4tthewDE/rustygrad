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

fn infer(_model: Efficientnet, mut image: DynamicImage) {
    let aspect_ratio = image.width() as f64 / image.height() as f64;
    image = image.resize_exact(
        (224.0 * aspect_ratio.max(1.0)) as u32,
        (224.0 * (1.0 / aspect_ratio).max(1.0)) as u32,
        FilterType::Nearest,
    );
    image = image.crop_imm(0, 0, 224, 224);
    let input = Tensor::from_image(image);

    dbg!(&input.shape);
    todo!();
}
