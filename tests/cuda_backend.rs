use rustygrad::{device::Device, tensor::Tensor};

mod util;

#[test]
fn load() {
    std::env::set_var("CUDA", "1");

    let a = Tensor::from_scalar(2.0);
    let (data, shape) = a.realize();

    assert_eq!(data, vec![2.0]);
    assert_eq!(shape, vec![1]);
}

#[test]
fn addition() {
    std::env::set_var("CUDA", "1");

    let a = Tensor::from_scalar(2.0);
    let b = Tensor::from_scalar(3.0);
    let (data, shape) = (a + b).realize();

    assert_eq!(data, vec![5.0]);
    assert_eq!(shape, vec![1]);
}

#[test]
fn addition_broadcasting() {
    std::env::set_var("CUDA", "1");

    let a = Tensor::from_vec(vec![2.0; 100 * 100], vec![100, 100]);
    let b = Tensor::from_scalar(3.0);
    let (data, shape) = (a + b).realize();

    assert_eq!(data, vec![5.0; 100 * 100]);
    assert_eq!(shape, vec![100, 100]);
}

#[test]
fn sub() {
    std::env::set_var("CUDA", "1");

    let a = Tensor::from_scalar(2.0);
    let b = Tensor::from_scalar(3.0);
    let (data, shape) = (a - b).realize();

    assert_eq!(data, vec![-1.0]);
    assert_eq!(shape, vec![1]);
}

#[test]
fn mul() {
    std::env::set_var("CUDA", "1");

    let a = Tensor::from_scalar(2.0);
    let b = Tensor::from_scalar(3.0);
    let (data, shape) = (a * b).realize();

    assert_eq!(data, vec![6.0]);
    assert_eq!(shape, vec![1]);
}

#[test]
fn div() {
    std::env::set_var("CUDA", "1");

    let a = Tensor::from_scalar(2.0);
    let b = Tensor::from_scalar(3.0);
    let (data, shape) = (a / b).realize();

    assert_eq!(data, vec![2.0 / 3.0]);
    assert_eq!(shape, vec![1]);
}

#[test]
fn sqrt() {
    std::env::set_var("CUDA", "1");

    let input = Tensor::rand(vec![100, 100]);
    let tch_input = util::to_tch(input.clone());

    let output = input.sqrt();
    let (data, shape) = output.realize();
    let tch_result = tch_input.sqrt();
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn log() {
    std::env::set_var("CUDA", "1");

    let input = Tensor::rand(vec![100, 100]);
    let tch_input = util::to_tch(input.clone());

    let output = input.log();
    let (data, shape) = output.realize();
    let tch_result = tch_input.log2();
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn relu() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![100, 100]);
    let tch_input = util::to_tch(input.clone());

    let output = input.relu();
    let (data, shape) = output.realize();
    let tch_result = tch_input.relu();
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn max() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![100, 100]);
    let tch_input = util::to_tch(input.clone());

    let output = input.max();
    let (data, shape) = output.realize();
    let tch_result = tch_input.max();
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn min() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![100, 100]);
    let tch_input = util::to_tch(input.clone());

    let output = input.min();
    let (data, shape) = output.realize();
    let tch_result = tch_input.min();
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn matmul() {
    std::env::set_var("CUDA", "1");
    let a = Tensor::rand(vec![100, 100]);
    let b = Tensor::rand(vec![100, 100]);
    let tch_a = util::to_tch(a.clone());
    let tch_b = util::to_tch(b.clone());

    let output = a.matmul(b);
    let (data, shape) = output.realize();
    let tch_result = tch_a.matmul(&tch_b);
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn expand() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![10, 1]);
    let tch_input = util::to_tch(input.clone());

    let output = input.expand(vec![10, 10]);
    let (data, shape) = output.realize();
    let tch_result = tch_input.expand(vec![10, 10], false);
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    assert_eq!(shape, tch_shape);
}

#[test]
fn pad_2d_cuda() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![1, 1, 3, 3]);
    let tch_input = util::to_tch(input.clone());
    let output = input.pad_2d(0., [1, 1, 1, 1]);
    let (data, shape) = output.realize();
    let tch_output = tch_input.zero_pad2d(1, 1, 1, 1);
    let tch_shape = util::tch_shape(&tch_output);
    let tch_output = util::tch_data(&tch_output);

    assert_eq!(shape, tch_shape);
    assert_eq!(data, tch_output,);
}

#[test]
fn pad_2d_weird_padding() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![1, 3, 16, 16]);
    let tch_input = util::to_tch(input.clone());
    let output = input.pad_2d(0., [1, 2, 3, 4]);
    let (data, shape) = output.realize();
    let tch_output = tch_input.zero_pad2d(1, 2, 3, 4);
    let tch_shape = util::tch_shape(&tch_output);
    let tch_output = util::tch_data(&tch_output);

    assert_eq!(shape, tch_shape);
    assert_eq!(data, tch_output,);
}

#[test]
fn permute() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![5, 15]);
    let tch_input = util::to_tch(input.clone());

    let output = input.permute(vec![1, 0]);
    let (data, shape) = output.realize();
    let tch_result = tch_input.permute(vec![1, 0]);
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    assert_eq!(data, tch_output);
    assert_eq!(shape, tch_shape);
}

#[test]
fn permute_harder() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![224, 224, 3]);
    let tch_input = util::to_tch(input.clone());

    let output = input.permute(vec![2, 0, 1]);
    let (data, shape) = output.realize();
    let tch_result = tch_input.permute(vec![2, 0, 1]);
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    assert_eq!(data, tch_output);
    assert_eq!(shape, tch_shape);
}

#[test]
fn reduce_sum_default_dims() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![2, 4, 3, 3]);
    let tch_input = util::to_tch(input.clone());
    let sum = input.reduce_sum(None, false);
    let (data, shape) = sum.realize();
    let tch_sum = tch_input.sum(None);
    let tch_shape = util::tch_shape(&tch_sum);
    let tch_output = util::tch_data(&tch_sum);
    assert_eq!(shape, tch_shape);
    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
}

#[test]
fn reduce_sum() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![2, 4, 3, 3]);
    let tch_input = util::to_tch(input.clone());
    let sum = input.reduce_sum(Some(vec![0, 2, 3]), false);
    let (data, shape) = sum.realize();
    let tch_sum = tch_input.sum_dim_intlist(vec![0, 2, 3], false, None);
    let tch_shape = util::tch_shape(&tch_sum);
    let tch_output = util::tch_data(&tch_sum);
    assert_eq!(shape, tch_shape);
    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
}

#[test]
fn avg_pool_2d() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![1, 3, 3, 3]);
    let tch_input = util::to_tch(input.clone());
    let output = input.avg_pool_2d((2, 2), None);
    let (data, shape) = output.realize();
    let tch_result = tch_input.avg_pool2d(vec![2, 2], 1, 0, false, true, None);
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);
    assert_eq!(data, tch_output);
    assert_eq!(shape, tch_shape);
}

#[test]
fn max_pool_2d() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![1, 3, 3, 3]);
    let tch_input = util::to_tch(input.clone());

    let output = input.max_pool_2d((2, 2), None);
    let (data, shape) = output.realize();
    let tch_result = tch_input.max_pool2d(vec![2, 2], 1, 0, 1, false);
    let tch_output = util::tch_data(&tch_result);
    let tch_shape = util::tch_shape(&tch_result);

    assert_eq!(data, tch_output);
    assert_eq!(shape, tch_shape);
}

#[test]
fn rand() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![4]);
    let (data, shape) = input.clone().realize();
    assert_eq!(shape, vec![4]);
    assert_ne!(data, vec![0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn conv2d_4d() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![1, 3, 224, 224]);
    let tch_input = util::to_tch(input.clone());
    let kernel = Tensor::rand(vec![32, 3, 3, 3]);
    let tch_kernel = util::to_tch(kernel.clone());

    let output = input.conv2d(kernel, None, Some([1, 1, 1, 1]), Some((2, 2)), None);
    let (data, shape) = output.realize();
    let tch_output = tch_input.conv2d(
        &tch_kernel,
        None::<tch::Tensor>,
        vec![2, 2],
        vec![1, 1],
        1,
        1,
    );

    let tch_shape = util::tch_shape(&tch_output);
    let tch_output = util::tch_data(&tch_output);

    assert_eq!(shape, tch_shape);
    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
}

#[test]
fn conv2d_4d_vs_torch() {
    std::env::set_var("CUDA", "1");
    let input = Tensor::rand(vec![1, 3, 224, 224]);
    let kernel = Tensor::rand(vec![32, 3, 3, 3]);

    let input_cpu = input.clone();
    let kernel_cpu = kernel.clone();

    let output = input.conv2d(kernel, None, Some([1, 1, 1, 1]), Some((2, 2)), None);
    let mut output_cpu = input_cpu.conv2d(kernel_cpu, None, Some([1, 1, 1, 1]), Some((2, 2)), None);
    output_cpu.device = Device::Cpu;
    let cuda_out = output.realize();
    let cpu_out = output_cpu.realize();
    util::assert_aprox_eq_vec(cuda_out.0, cpu_out.0, 1e-15);
}
