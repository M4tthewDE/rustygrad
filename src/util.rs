use std::fs;
use std::fs::create_dir;
use std::fs::File;
use std::io::copy;
use std::path::PathBuf;

use anyhow::anyhow;
use anyhow::Result;
use tracing::info;

pub fn load_torch_model(url: &str) -> Result<()> {
    let file_name = url
        .split('/')
        .last()
        .ok_or(anyhow!("no file name in url"))?;
    let mut path = get_cache_dir()?.join(file_name);

    if path.exists() {
        info!("using cached model in {:?}", path);
    } else {
        download_file(url, &path)?;
    }

    path.pop();
    let path = path.join(format!("{file_name}.json"));
    if path.exists() {
        info!("using cached json of model in {:?}", path);
    } else {
        todo!("convert to json");
    }

    load_model(&path)?;
    Ok(())
}

fn load_model(path: &PathBuf) -> Result<()> {
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
