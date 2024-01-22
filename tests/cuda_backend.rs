#[cfg(test)]
mod cuda {
    use rustygrad::{
        device::{self, Device},
        tensor::Tensor,
        util,
    };

    #[test]
    fn load() {
        device::set_device(Device::Cuda);

        let a = Tensor::from_scalar(2.0);
        let (data, shape) = a.realize();

        assert_eq!(data, vec![2.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn addition() {
        device::set_device(Device::Cuda);

        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let (data, shape) = (a + b).realize();

        assert_eq!(data, vec![5.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn addition_broadcasting() {
        device::set_device(Device::Cuda);

        let a = Tensor::from_vec(vec![2.0; 100 * 100], vec![100, 100]);
        let b = Tensor::from_scalar(3.0);
        let (data, shape) = (a + b).realize();

        assert_eq!(data, vec![5.0; 100 * 100]);
        assert_eq!(shape, vec![100, 100]);
    }

    #[test]
    fn sub() {
        device::set_device(Device::Cuda);

        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let (data, shape) = (a - b).realize();

        assert_eq!(data, vec![-1.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn mul() {
        device::set_device(Device::Cuda);

        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let (data, shape) = (a * b).realize();

        assert_eq!(data, vec![6.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn div() {
        device::set_device(Device::Cuda);

        let a = Tensor::from_scalar(2.0);
        let b = Tensor::from_scalar(3.0);
        let (data, shape) = (a / b).realize();

        assert_eq!(data, vec![2.0 / 3.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn sqrt() {
        device::set_device(Device::Cuda);

        let input = Tensor::rand(vec![100, 100]);
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
        device::set_device(Device::Cuda);

        let input = Tensor::rand(vec![100, 100]);
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
    fn relu() {
        device::set_device(Device::Cuda);
        let input = Tensor::rand(vec![100, 100]);
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
    fn max() {
        device::set_device(Device::Cuda);
        let input = Tensor::rand(vec![100, 100]);
        let tch_input = input.to_tch();

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
        device::set_device(Device::Cuda);
        let input = Tensor::rand(vec![100, 100]);
        let tch_input = input.to_tch();

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
        device::set_device(Device::Cuda);
        let a = Tensor::rand(vec![100, 100]);
        let b = Tensor::rand(vec![100, 100]);
        let tch_a = a.to_tch();
        let tch_b = b.to_tch();

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
        device::set_device(Device::Cuda);
        let input = Tensor::rand(vec![10, 1]);
        let tch_input = input.to_tch();

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
        device::set_device(Device::Cuda);
        let input = Tensor::rand(vec![1, 1, 3, 3]);
        let tch_input = input.to_tch();
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
        device::set_device(Device::Cuda);
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
        device::set_device(Device::Cuda);
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
        device::set_device(Device::Cuda);
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
    fn reduce_sum_default_dims() {
        device::set_device(Device::Cuda);
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
        device::set_device(Device::Cuda);
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
}
