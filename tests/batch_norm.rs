use rustygrad::{nn::batch_norm::BatchNorm2dBuilder, tensor::Tensor};

mod util;

#[test]
fn test_batchnorm2d_no_training() {
    let num_features = 4;
    let mut bn = BatchNorm2dBuilder::new(num_features).eps(1e-5).build();
    bn.weight = Some(Tensor::rand(vec![4]));
    bn.bias = Some(Tensor::rand(vec![4]));
    bn.running_mean = Tensor::rand(vec![4]);
    bn.running_var = Tensor::rand(vec![4]);

    let input = Tensor::rand(vec![2, num_features, 3, 3]);
    let tch_input = util::to_tch(input.clone());
    let out = bn.forward(input);
    let (data, shape) = out.realize();
    let tch_weight = util::to_tch(bn.weight.unwrap());
    let tch_bias = util::to_tch(bn.bias.unwrap());
    let tch_running_mean = util::to_tch(bn.running_mean);
    let tch_running_var = util::to_tch(bn.running_var);
    let tch_out = tch_input.batch_norm(
        Some(tch_weight),
        Some(tch_bias),
        Some(tch_running_mean),
        Some(tch_running_var),
        false,
        0.1,
        1e-5,
        false,
    );
    let tch_shape = util::tch_shape(&tch_out);
    let tch_output = util::tch_data(&tch_out);

    assert_eq!(shape, tch_shape);
    util::assert_aprox_eq_vec(data, tch_output, 1e-6);
}
