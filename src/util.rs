use anyhow::anyhow;
use anyhow::Result;
use assert_approx_eq::assert_approx_eq;
use serde::Deserialize;
use std::fs::create_dir;
use std::fs::File;
use std::io::copy;
use std::io::Read;
use std::iter::zip;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use tracing::info;

pub fn load_torch_model(url: &str) -> Result<ModelData> {
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

#[derive(Debug, Deserialize)]
pub struct ModelData {
    #[serde(rename(deserialize = "_bn0.weight"))]
    pub bn0_weight: Vec<f64>,
    #[serde(rename(deserialize = "_bn0.bias"))]
    pub bn0_bias: Vec<f64>,
    #[serde(rename(deserialize = "_bn0.running_mean"))]
    pub bn0_running_mean: Vec<f64>,
    #[serde(rename(deserialize = "_bn0.running_var"))]
    pub bn0_running_var: Vec<f64>,
    #[serde(rename(deserialize = "_bn0.num_batches_tracked"))]
    pub bn0_num_batches_tracked: usize,

    #[serde(rename(deserialize = "_bn1.weight"))]
    pub bn1_weight: Vec<f64>,
    #[serde(rename(deserialize = "_bn1.bias"))]
    pub bn1_bias: Vec<f64>,
    #[serde(rename(deserialize = "_bn1.running_mean"))]
    pub bn1_running_mean: Vec<f64>,
    #[serde(rename(deserialize = "_bn1.running_var"))]
    pub bn1_running_var: Vec<f64>,
    #[serde(rename(deserialize = "_bn1.num_batches_tracked"))]
    pub bn1_num_batches_tracked: usize,
    #[serde(rename(deserialize = "_conv_head.weight"))]
    pub conv_head_weight: Vec<Vec<Vec<Vec<f64>>>>,
    #[serde(rename(deserialize = "_conv_stem.weight"))]
    pub conv_stem_weight: Vec<Vec<Vec<Vec<f64>>>>,
    #[serde(rename(deserialize = "_fc.bias"))]
    pub fc_bias: Vec<f64>,
    #[serde(rename(deserialize = "_fc.weight"))]
    pub fc_weight: Vec<Vec<f64>>,
}

fn load_model(path: &PathBuf) -> Result<ModelData> {
    let start = Instant::now();
    info!("loading model from {path:?}, this might take a while...");

    let mut content = String::new();
    File::open(path)?.read_to_string(&mut content)?;
    let model_data: ModelData = serde_json::from_str(&content)?;

    info!("loaded model in {:?}", start.elapsed());
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
