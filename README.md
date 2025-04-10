# Deep learning library

Inspired by [tinygrad](https://github.com/tinygrad/tinygrad)

## Examples

`cargo run --release --example efficientnet static/chicken.jpg`

## Environment variables

* RUST_LOG to adjust log level (ERROR, INFO, DEBUG, TRACE)
* GPU=1 to enable opencl gpu backend
* GRAPH=1 to output compute graph (graph.dot)
