use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rustygrad::tensor::Tensor;

fn criterion_benchmark(c: &mut Criterion) {
    let t = Tensor::rand(vec![1, 3, 100, 100]);
    let kernel = Tensor::rand(vec![1, 3, 10, 10]);
    let t = t.conv2d(&kernel, None, None, None, None);

    c.bench_function("conv2d", move |b| {
        b.iter_batched(|| t.clone(), |t| t.realize(), BatchSize::SmallInput)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
