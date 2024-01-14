use std::ffi::{c_int, c_void, CStr};

use tracing::trace;

use crate::op::Op;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    //fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const i8;
    fn cudaGetLastError() -> c_int;

    fn add(a: *const c_void, b: *const c_void, c: *mut c_void, n: usize);
}

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

unsafe fn realize_cuda(op: &Op) -> (*mut c_void, Vec<usize>) {
    match op {
        Op::Add(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);

            let result_size = shape.iter().product::<usize>();
            let mut result_ptr: *mut c_void = std::ptr::null_mut();
            let code = cudaMalloc(
                &mut result_ptr as *mut *mut c_void,
                result_size * std::mem::size_of::<f64>(),
            );
            if code != 0 {
                panic!("{}", error_string(code));
            }

            add(lhs_ptr, rhs_ptr, result_ptr, result_size);
            let code = cudaGetLastError();
            if code != 0 {
                panic!("{}", error_string(code));
            }
            (result_ptr, shape)
        }
        Op::Sub(_, _) => todo!(),
        Op::Mul(_, _) => todo!(),
        Op::Div(_, _) => todo!(),
        Op::Max(_) => todo!(),
        Op::Min(_) => todo!(),
        Op::Sqrt(_) => todo!(),
        Op::Log(_) => todo!(),
        Op::Load(data, shape) => {
            let bytes = data.len() * std::mem::size_of::<f64>();
            let mut dev_ptr: *mut c_void = std::ptr::null_mut();
            let code = cudaMalloc(&mut dev_ptr as *mut *mut c_void, bytes);
            if code != 0 {
                panic!("{}", error_string(code));
            }
            let code = cudaMemcpy(
                dev_ptr,
                data.as_ptr() as *const c_void,
                bytes,
                HOST_TO_DEVICE,
            );
            if code != 0 {
                panic!("{}", error_string(code));
            }
            (dev_ptr, shape.to_vec())
        }
        Op::Sigmoid(_) => todo!(),
        Op::Relu(_) => todo!(),
        Op::Sum(_, _, _) => todo!(),
        Op::Pool2D(_, _, _, _, _) => todo!(),
        Op::Conv2D(_, _, _, _) => todo!(),
        Op::Pad2D(_, _, _) => todo!(),
        Op::Reshape(t, shape) => {
            let (t_ptr, _) = realize_cuda(&t.op);
            (t_ptr, shape.to_vec())
        }
        Op::Permute(_, _) => todo!(),
        Op::Expand(t, _) => {
            let (t_ptr, _) = realize_cuda(&t.op);
            dbg!(t_ptr);
            todo!();
        }
        Op::MatMul(_, _) => todo!(),
    }
}

pub fn realize(op: &Op) -> (Vec<f64>, Vec<usize>) {
    trace!("Realizing {:?}", op);

    unsafe {
        let (result_ptr, shape) = realize_cuda(op);
        let result_size = shape.iter().product();
        let mut result = vec![0.0; result_size];
        let code = cudaMemcpy(
            result.as_mut_ptr() as *mut c_void,
            result_ptr,
            result_size * std::mem::size_of::<f64>(),
            DEVICE_TO_HOST,
        );
        if code != 0 {
            panic!("{}", error_string(code));
        }

        (result, shape)
    }
}

unsafe fn error_string(code: i32) -> String {
    CStr::from_ptr(cudaGetErrorString(code))
        .to_string_lossy()
        .into_owned()
}
