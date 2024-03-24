use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rustygrad::tensor::Tensor;

fn criterion_benchmark(c: &mut Criterion) {
    std::env::set_var("CUDA", "1");

    let t = Tensor::rand(vec![100, 100, 1]).expand(vec![100, 100, 100]);

    c.bench_function("expand_cuda", move |b| {
        b.iter_batched(|| t.clone(), |t| t.realize(), BatchSize::SmallInput)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
