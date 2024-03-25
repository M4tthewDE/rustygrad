use image::{imageops::FilterType, io::Reader};
use rustygrad::{models::efficientnet::Efficientnet, tensor::Tensor, util};

const BIAS: [f64; 3] = [0.485, 0.456, 0.406];
const SCALE: [f64; 3] = [0.229, 0.224, 0.225];

#[test]
fn efficientnet() {
    let mut model = Efficientnet::from_model();
    let mut image = Reader::open("static/chicken.jpg".to_string())
        .unwrap()
        .decode()
        .unwrap();

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
    let (data, _) = out.realize();
    let argmax = util::argmax(&data);
    let max = data
        .into_iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .expect("no min value found");

    assert_eq!(argmax, 8);
    assert_eq!(max, 8.869465815599556);
}
