#[cfg(test)]
mod cuda {
    use rustygrad::{
        device::{self, Device},
        tensor::Tensor,
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
}
