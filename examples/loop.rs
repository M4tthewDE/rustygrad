use std::time::Instant;

use rustygrad::tensor::Tensor;

fn main() {
    let mut x = Tensor::rand(vec![128, 128]);

    for _ in 0..2 {
        x = test(x);
    }

    let start = Instant::now();
    x.realize();
    println!("elapsed: {:?}", start.elapsed());
}

fn test(t: Tensor) -> Tensor {
    let old = &t;

    let a = t.sigmoid();
    a.matmul(old)
}
