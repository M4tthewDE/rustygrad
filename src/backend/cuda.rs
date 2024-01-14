use std::ffi::{c_int, c_void, CStr};

use tracing::trace;

use crate::op::Op;

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    // FIXME: we should really use this at some point...
    //fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const i8;
    fn cudaGetLastError() -> c_int;

    fn add(a: *const c_void, b: *const c_void, c: *mut c_void, n: usize);
    fn sub(a: *const c_void, b: *const c_void, c: *mut c_void, n: usize);
}

unsafe fn error_string(code: i32) -> String {
    CStr::from_ptr(cudaGetErrorString(code))
        .to_string_lossy()
        .into_owned()
}

unsafe fn check_last_error() {
    let code = cudaGetLastError();
    if code != 0 {
        panic!("{}", error_string(code));
    }
}

unsafe fn malloc(size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let code = cudaMalloc(
        &mut ptr as *mut *mut c_void,
        size * std::mem::size_of::<f64>(),
    );
    if code != 0 {
        panic!("{}", error_string(code));
    }

    ptr
}

unsafe fn memcpy_to_device(ptr: *mut c_void, data: &Vec<f64>) {
    let code = cudaMemcpy(
        ptr,
        data.as_ptr() as *const c_void,
        data.len() * std::mem::size_of::<f64>(),
        HOST_TO_DEVICE,
    );
    if code != 0 {
        panic!("{}", error_string(code));
    }
}

unsafe fn memcpy_to_host(result: &mut Vec<f64>, ptr: *mut c_void, size: usize) {
    let code = cudaMemcpy(
        result.as_mut_ptr() as *mut c_void,
        ptr,
        size * std::mem::size_of::<f64>(),
        DEVICE_TO_HOST,
    );
    if code != 0 {
        panic!("{}", error_string(code));
    }
}

unsafe fn realize_cuda(op: &Op) -> (*mut c_void, Vec<usize>) {
    match op {
        Op::Add(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size);
            add(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Sub(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size);
            sub(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Mul(_, _) => todo!(),
        Op::Div(_, _) => todo!(),
        Op::Max(_) => todo!(),
        Op::Min(_) => todo!(),
        Op::Sqrt(_) => todo!(),
        Op::Log(_) => todo!(),
        Op::Load(data, shape) => {
            let dev_ptr = malloc(data.len());
            memcpy_to_device(dev_ptr, data);
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
            let (_, _) = realize_cuda(&t.op);
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
        memcpy_to_host(&mut result, result_ptr, result_size);
        (result, shape)
    }
}
