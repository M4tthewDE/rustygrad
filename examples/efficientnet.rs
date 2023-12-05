use rustygrad::efficientnet::Efficientnet;

fn main() {
    // https://github.com/tinygrad/tinygrad/blob/master/extra/models/efficientnet.py
    // https://github.com/tinygrad/tinygrad/blob/master/examples/efficientnet.py
    //
    tracing_subscriber::fmt::init();

    let efficientnet = Efficientnet::default();
    efficientnet.load_pretrained();
    todo!();
}
