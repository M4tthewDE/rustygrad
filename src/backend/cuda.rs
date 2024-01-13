use std::ffi::{c_int, c_void, CStr};

use tracing::trace;

use crate::op::Op;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const i8;
}

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

pub fn realize_cuda(op: &Op, mut dev_ptr: *mut c_void) -> (Vec<usize>, *mut c_void) {
    match op {
        Op::Add(_, _) => todo!(),
        Op::Sub(_, _) => todo!(),
        Op::Mul(_, _) => todo!(),
        Op::Div(_, _) => todo!(),
        Op::Max(_) => todo!(),
        Op::Min(_) => todo!(),
        Op::Sqrt(_) => todo!(),
        Op::Log(_) => todo!(),
        Op::Load(data, shape) => {
            let bytes = data.len() * std::mem::size_of::<f64>();

            unsafe {
                let result = cudaMalloc(&mut dev_ptr as *mut *mut c_void, bytes);
                assert_eq!(result, 0, "cudaMalloc failled with error code {}", result);

                let result = cudaMemcpy(
                    dev_ptr,
                    data.as_ptr() as *const c_void,
                    bytes,
                    HOST_TO_DEVICE,
                );
                if result != 0 {
                    cudaFree(dev_ptr);
                    panic!("Failed to copy data to the GPU {}", result);
                }
            }
            (shape.to_vec(), dev_ptr)
        }
        Op::Sigmoid(_) => todo!(),
        Op::Relu(_) => todo!(),
        Op::Sum(_, _, _) => todo!(),
        Op::Pool2D(_, _, _, _, _) => todo!(),
        Op::Conv2D(_, _, _, _) => todo!(),
        Op::Pad2D(_, _, _) => todo!(),
        Op::Reshape(_, _) => todo!(),
        Op::Permute(_, _) => todo!(),
        Op::Expand(_, _) => todo!(),
        Op::MatMul(_, _) => todo!(),
    }
}

pub fn realize(op: &Op) -> (Vec<f64>, Vec<usize>) {
    trace!("Realizing {:?}", op);

    let dev_ptr: *mut c_void = std::ptr::null_mut();

    let (shape, dev_ptr) = realize_cuda(op, dev_ptr);

    let result_size = shape.iter().product();
    let mut result = vec![0.0; result_size];
    unsafe {
        let code = cudaMemcpy(
            result.as_mut_ptr() as *mut c_void,
            dev_ptr,
            result_size * std::mem::size_of::<f64>(),
            DEVICE_TO_HOST,
        );
        if code != 0 {
            let error_str = CStr::from_ptr(cudaGetErrorString(code))
                .to_string_lossy()
                .into_owned();
            panic!("Failed to copy data from the GPU to host: {}", error_str);
        }
    }

    (result, shape)
}