use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rustygrad::tensor::Tensor;

fn criterion_benchmark(c: &mut Criterion) {
    std::env::set_var("CUDA", "1");

    let t = Tensor::rand(vec![1, 1, 100, 100]).avg_pool_2d((2, 2), None);

    c.bench_function("avg_pool_2d_cuda", move |b| {
        b.iter_batched(|| t.clone(), |t| t.realize(), BatchSize::SmallInput)
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
