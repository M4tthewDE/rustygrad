use anyhow::anyhow;
use anyhow::Result;
use assert_approx_eq::assert_approx_eq;
use serde_json::Value;
use std::fs::create_dir;
use std::fs::File;
use std::io::copy;
use std::io::Read;
use std::iter::zip;
use std::path::PathBuf;
use std::process::Command;
use tracing::info;

use crate::tensor::Tensor;

pub fn load_torch_model(url: &str) -> Result<Value> {
    let file_name = url
        .split('/')
        .last()
        .ok_or(anyhow!("no file name in url"))?;
    let path = get_cache_dir()?.join(file_name);

    if path.exists() {
        info!("using cached model in {:?}", path);
    } else {
        download_file(url, &path)?;
    }

    let mut json_path = path.clone();
    json_path.pop();
    let json_path = json_path.join(format!("{file_name}.json"));
    if json_path.exists() {
        info!("using cached json of model in {:?}", json_path);
    } else {
        info!("converting pth model to json...");
        convert_to_json(
            path.to_str()
                .ok_or(anyhow!("error converting path to string {:?}", path))?,
        )?;
    }

    load_model(&json_path)
}

fn load_model(path: &PathBuf) -> Result<Value> {
    let mut content = String::new();
    File::open(path)?.read_to_string(&mut content)?;
    let model_data: Value = serde_json::from_str(&content)?;

    Ok(model_data)
}

fn convert_to_json(path: &str) -> Result<()> {
    Command::new("python3")
        .args(["src/pth_to_json.py", path])
        .output()?;
    Ok(())
}

pub fn download_file(url: &str, path: &PathBuf) -> Result<()> {
    info!("loading torch model from {}", url);
    let response = reqwest::blocking::get(url)?;

    info!("saving model in {:?}", path);

    let mut dest = File::create(path.clone())?;
    let byte_vec = response.bytes()?.to_vec();
    let mut bytes = byte_vec.as_slice();
    copy(&mut bytes, &mut dest)?;

    Ok(())
}

fn get_cache_dir() -> Result<PathBuf> {
    let path = dirs::cache_dir()
        .ok_or(anyhow!("unable to locate cache dir"))?
        .join("rustygrad/");

    if !path.exists() {
        create_dir(path.clone())?;
    }

    Ok(path)
}

pub fn assert_aprox_eq_vec(a: Vec<f64>, b: Vec<f64>, tolerance: f64) {
    for (a1, b1) in zip(a, b) {
        if a1.is_nan() {
            assert!(b1.is_nan());
        } else if b1.is_nan() {
            assert!(a1.is_nan());
        } else {
            assert_approx_eq!(a1, b1, tolerance);
        }
    }
}

pub fn extract_floats(array: &Value) -> Option<Vec<f64>> {
    array
        .as_array()?
        .iter()
        .filter_map(|x| x.as_f64())
        .collect::<Vec<f64>>()
        .into()
}

pub fn extract_2d_tensor(v: &Value) -> Option<Tensor> {
    let data = extract_2d_array(v)?;
    Some(Tensor::from_vec(
        data.clone().into_iter().flatten().collect(),
        vec![data.len(), data[0].len()],
    ))
}

fn extract_2d_array(v: &Value) -> Option<Vec<Vec<f64>>> {
    v.as_array()?
        .iter()
        .map(extract_floats)
        .collect::<Option<Vec<Vec<f64>>>>()
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

pub fn tch_data(tch: &tch::Tensor) -> Vec<f64> {
    tch.flatten(0, tch.size().len() as i64 - 1)
        .iter::<f64>()
        .unwrap()
        .collect()
}

pub fn tch_shape(tch: &tch::Tensor) -> Vec<usize> {
    tch.size().iter().map(|&d| d as usize).collect()
}

pub fn index_4d_to_1d(shape: &[usize], n: usize, c: usize, h: usize, w: usize) -> usize {
    let (height, width) = (shape[2], shape[3]);
    let channels = shape[1];
    n * (channels * height * width) + c * (height * width) + h * width + w
}
