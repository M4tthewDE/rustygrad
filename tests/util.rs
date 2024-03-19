use std::iter::zip;

use rustygrad::tensor::Tensor;

pub fn to_tch(tensor: Tensor) -> tch::Tensor {
    let (data, shape) = tensor.realize();
    tch::Tensor::from_slice(&data).reshape(shape.iter().map(|&d| d as i64).collect::<Vec<i64>>())
}

pub fn tch_data(tch: &tch::Tensor) -> Vec<f64> {
    tch.flatten(0, tch.size().len() as i64 - 1)
        .iter::<f64>()
        .unwrap()
        .collect()
}

pub fn tch_shape(tch: &tch::Tensor) -> Vec<usize> {
    tch.size().iter().map(|&d| d as usize).collect()
}

pub fn assert_aprox_eq_vec(a: Vec<f64>, b: Vec<f64>, tolerance: f64) {
    for (a1, b1) in zip(a, b) {
        if a1.is_nan() {
            assert!(b1.is_nan());
        } else if b1.is_nan() {
            assert!(a1.is_nan());
        } else {
            assert!((a1 - b1).abs() < tolerance);
        }
    }
}
