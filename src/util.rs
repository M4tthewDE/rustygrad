use serde_json::Value;
use std::fs::create_dir;
use std::fs::File;
use std::io::copy;
use std::io::Read;
use std::path::PathBuf;
use std::process::Command;
use tracing::info;

use crate::tensor::Tensor;

pub fn load_torch_model(url: &str) -> Value {
    let file_name = url.split('/').last().unwrap();
    let path = get_cache_dir().join(file_name);

    if path.exists() {
        info!("using cached model in {:?}", path);
    } else {
        download_file(url, &path);
    }

    let mut json_path = path.clone();
    json_path.pop();
    let json_path = json_path.join(format!("{file_name}.json"));
    if json_path.exists() {
        info!("using cached json of model in {:?}", json_path);
    } else {
        info!("converting pth model to json...");
        convert_to_json(path.to_str().unwrap());
    }

    load_model(&json_path)
}

fn load_model(path: &PathBuf) -> Value {
    let mut content = String::new();
    File::open(path)
        .unwrap()
        .read_to_string(&mut content)
        .unwrap();
    let model_data: Value = serde_json::from_str(&content).unwrap();

    model_data
}

fn convert_to_json(path: &str) {
    Command::new("python3")
        .args(["src/pth_to_json.py", path])
        .output()
        .unwrap();
}

pub fn download_file(url: &str, path: &PathBuf) {
    info!("loading torch model from {}", url);
    let response = reqwest::blocking::get(url).unwrap();

    info!("saving model in {:?}", path);

    let mut dest = File::create(path.clone()).unwrap();
    let byte_vec = response.bytes().unwrap().to_vec();
    let mut bytes = byte_vec.as_slice();
    copy(&mut bytes, &mut dest).unwrap();
}

fn get_cache_dir() -> PathBuf {
    let path = dirs::cache_dir().unwrap().join("rustygrad/");

    if !path.exists() {
        create_dir(path.clone()).unwrap();
    }

    path
}

pub fn extract_floats(array: &Value) -> Option<Vec<f64>> {
    array
        .as_array()?
        .iter()
        .filter_map(|x| x.as_f64())
        .collect::<Vec<f64>>()
        .into()
}

pub fn extract_1d_tensor(v: &Value) -> Option<Tensor> {
    let data = extract_floats(v)?;
    let len = data.len();
    Some(Tensor::from_vec(data, vec![len]))
}

pub fn extract_2d_tensor(v: &Value) -> Option<Tensor> {
    let data = v
        .as_array()?
        .iter()
        .map(extract_floats)
        .collect::<Option<Vec<Vec<f64>>>>()?;
    Some(Tensor::from_vec(
        data.clone().into_iter().flatten().collect(),
        vec![data.len(), data[0].len()],
    ))
}

fn extract_4d_array(v: &Value) -> Option<Vec<Vec<Vec<Vec<f64>>>>> {
    v.as_array()?
        .iter()
        .map(|third_dim| {
            third_dim
                .as_array()?
                .iter()
                .map(|second_dim| {
                    second_dim
                        .as_array()?
                        .iter()
                        .map(extract_floats)
                        .collect::<Option<Vec<Vec<f64>>>>()
                })
                .collect::<Option<Vec<Vec<Vec<f64>>>>>()
        })
        .collect::<Option<Vec<Vec<Vec<Vec<f64>>>>>>()
}

pub fn extract_4d_tensor(v: &Value) -> Option<Tensor> {
    let data = extract_4d_array(v)?;
    Some(Tensor::from_vec(
        data.clone()
            .into_iter()
            .flat_map(|x| x.into_iter())
            .flat_map(|x| x.into_iter())
            .flat_map(|x| x.into_iter())
            .collect(),
        vec![
            data.len(),
            data[0].len(),
            data[0][0].len(),
            data[0][0][0].len(),
        ],
    ))
}

pub fn argmax(data: &[f64]) -> usize {
    let mut index = 0;
    for (i, d) in data.iter().enumerate() {
        if d > &data[index] {
            index = i;
        }
    }

    index
}

pub fn index_4d_to_1d(shape: &[usize], n: usize, c: usize, h: usize, w: usize) -> usize {
    let (height, width) = (shape[2], shape[3]);
    let channels = shape[1];
    n * (channels * height * width) + c * (height * width) + h * width + w
}
