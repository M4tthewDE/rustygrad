#[cfg(test)]
mod cpu {
    use rustygrad::{tensor::Tensor, util};

    #[test]
    fn addition_scalar() {
        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let result = a + b;
        let (data, shape) = result.realize();

        assert_eq!(data, vec![5.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn addition_vector() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let b = Tensor::from_vec(vec![8.0, 7.0], vec![2]);
        let result = a + b;
        let (data, shape) = result.realize();

        assert_eq!(data, vec![10.0, 10.0]);
        assert_eq!(shape, vec![2]);
    }

    #[test]
    fn addition_f64() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let result = a + 5.0;
        let (data, shape) = result.realize();

        assert_eq!(data, vec![7.0, 8.0]);
        assert_eq!(shape, vec![2]);
    }

    #[test]
    fn addition_f64_left_side() {
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
        let result = 5.0 + a;
        let (data, shape) = result.realize();

        assert_eq!(data, vec![7.0, 8.0]);
        assert_eq!(shape, vec![2]);
    }

    #[test]
    fn addition_broadcasting() {
        let a = Tensor::rand(vec![5, 2, 3]);
        let a_tch = a.to_tch();
        let b = Tensor::rand(vec![5, 2, 1]);
        let b_tch = b.to_tch();

        let result = a + b;
        let (data, shape) = result.realize();
        let tch_result = a_tch + b_tch;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn addition_broadcasting_add_dim() {
        let a = Tensor::rand(vec![5, 2, 3]);
        let a_tch = a.to_tch();
        let b = Tensor::rand(vec![2, 3]);
        let b_tch = b.to_tch();

        let result = a + b;
        let (data, shape) = result.realize();
        let tch_result = a_tch + b_tch;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn div() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();
        let tch_input2 = input.to_tch();

        let output = input.clone() / input;
        let (data, shape) = output.realize();
        let tch_result = tch_input / tch_input2;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn div_scalar() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();

        let output = input / 2.0;
        let (data, shape) = output.realize();

        let tch_result = tch_input / 2.0;
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }
    #[test]
    fn max() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();

        let output = input.max();
        let (data, shape) = output.realize();
        let tch_result = tch_input.max();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn min() {
        let input = Tensor::rand(vec![10, 10, 10]);
        let tch_input = input.to_tch();

        let output = input.min();
        let (data, shape) = output.realize();
        let tch_result = tch_input.min();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn sqrt() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

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
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let output = input.log();
        let (data, shape) = output.realize();
        let tch_result = tch_input.log2();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(data, tch_output, 1e-6);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn swish() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let output = input.swish();
        let (data, shape) = output.realize();
        let tch_result = tch_input.silu();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(data, tch_output, 1e-6);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn relu() {
        let input = Tensor::rand(vec![10, 10]);
        let tch_input = input.to_tch();

        let output = input.relu();
        let (data, shape) = output.realize();
        let tch_result = tch_input.relu();
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(data, tch_output, 1e-6);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn reduce_sum_default_dims() {
        let input = Tensor::rand(vec![2, 4, 3, 3]);
        let tch_input = input.to_tch();
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
        let input = Tensor::rand(vec![2, 4, 3, 3]);
        let tch_input = input.to_tch();
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
        let input = Tensor::rand(vec![1, 1, 10, 10]);
        let tch_input = input.to_tch();

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
        let input = Tensor::rand(vec![1, 1, 10, 10]);
        let tch_input = input.to_tch();

        let output = input.max_pool_2d((2, 2), None);
        let (data, shape) = output.realize();
        let tch_result = tch_input.max_pool2d(vec![2, 2], 1, 0, 1, false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn conv2d_4d() {
        let input = Tensor::rand(vec![1, 3, 224, 224]);
        let tch_input = input.to_tch();
        let kernel = Tensor::rand(vec![32, 3, 3, 3]);
        let tch_kernel = kernel.to_tch();

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
    fn conv2d_shape_before_realize() {
        let input = Tensor::rand(vec![1, 3, 224, 224]);
        let tch_input = input.to_tch();
        let kernel = Tensor::rand(vec![32, 3, 3, 3]);
        let tch_kernel = kernel.to_tch();

        let output = input.conv2d(kernel, None, Some([0, 1, 0, 1]), Some((2, 2)), None);
        assert_eq!(output.shape, vec![1, 32, 112, 112]);
        let (data, shape) = output.realize();
        let tch_input = tch_input.zero_pad2d(0, 1, 0, 1);
        let tch_output = tch_input.conv2d(
            &tch_kernel,
            None::<tch::Tensor>,
            vec![2, 2],
            vec![0, 0],
            1,
            1,
        );
        let tch_shape = util::tch_shape(&tch_output);
        let tch_output = util::tch_data(&tch_output);

        assert_eq!(shape, tch_shape);
        util::assert_aprox_eq_vec(data, tch_output, 1e-6);
    }

    #[test]
    fn pad_2d_weird_padding() {
        let input = Tensor::rand(vec![1, 3, 16, 16]);
        let tch_input = input.to_tch();
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
        let input = Tensor::rand(vec![5, 15]);
        let tch_input = input.to_tch();

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
        let input = Tensor::rand(vec![224, 224, 3]);
        let tch_input = input.to_tch();

        let output = input.permute(vec![2, 0, 1]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.permute(vec![2, 0, 1]);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data, tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_end() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![2, 2, 3]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![2, 2, 3], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_middle() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![2, 3, 2]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![2, 3, 2], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_first() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![3, 2, 2]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![3, 2, 2], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_complicated_front() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![2, 2, 2, 2]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![2, 2, 2, 2], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_complicated_back() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1, 1]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![2, 2, 2, 2]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![2, 2, 2, 2], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_complicated_front_and_back() {
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2, 1, 1]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![2, 2, 2, 2]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![2, 2, 2, 2], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn expand_complicated_middle() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 1, 2]);
        let tch_input = input.to_tch();

        let output = input.expand(vec![2, 2, 2, 2]);
        let (data, shape) = output.realize();
        let tch_result = tch_input.expand(vec![2, 2, 2, 2], false);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        assert_eq!(data.clone(), tch_output);
        assert_eq!(shape, tch_shape);
    }

    #[test]
    fn matmul() {
        let input1 = Tensor::rand(vec![8, 10]);
        let input2 = Tensor::rand(vec![10, 12]);
        let tch_input1 = input1.to_tch();
        let tch_input2 = input2.to_tch();

        let (output_data, shape) = input1.matmul(input2).realize();
        let tch_result = tch_input1.matmul(&tch_input2);
        let tch_output = util::tch_data(&tch_result);
        let tch_shape = util::tch_shape(&tch_result);

        util::assert_aprox_eq_vec(output_data, tch_output, 1e-6);
        assert_eq!(shape, tch_shape);
    }
}
