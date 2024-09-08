use std::io::Cursor;

use serde::Deserialize;
use tracing::warn;
use std::collections::HashMap;

pub fn main() {
    tracing_subscriber::fmt::init();

    let bytes = include_bytes!("../de_core_news_sm/de_core_news_sm-3.7.0/tokenizer");
    let tokenizer: Tokenizer = rmp_serde::from_slice(bytes).unwrap();
    dbg!(tokenizer);
}

#[derive(Default, Debug, Deserialize)]
struct Tokenizer {
    prefix_search: String,
    suffix_search: String,
    infix_finditer: String,
    url_match: String,
    exceptions: HashMap<String, Vec<HashMap<u64, String>>>,
    faster_heuristics: bool,
}

