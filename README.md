# Tinygrad, but in rust

Because I got tired of using python.

## Useful commands

Build using local pytorch

`LIBTORCH_USE_PYTORCH=1 cargo build --release --example efficientnet`

Run efficientnet

`LD_LIBRARY_PATH=/path/to/python/venv/lib/python3.11/site-packages/torch/lib ./target/release/examples/efficientnet static/chicken.jpg`
