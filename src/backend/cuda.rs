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
    fn mul(a: *const c_void, b: *const c_void, c: *mut c_void, n: usize);
    fn division(a: *const c_void, b: *const c_void, c: *mut c_void, n: usize);
    fn rusty_sqrt(a: *const c_void, c: *mut c_void, n: usize);
    fn rusty_log(a: *const c_void, c: *mut c_void, n: usize);
    fn relu(a: *const c_void, c: *mut c_void, n: usize);
    fn sigmoid(a: *const c_void, c: *mut c_void, n: usize);
    fn rusty_max(a: *const c_void, c: *mut c_void, n: usize);
    fn rusty_min(a: *const c_void, c: *mut c_void, n: usize);
    fn matmul(a: *const c_void, b: *const c_void, c: *const c_void, M: usize, K: usize, N: usize);
    fn expand(
        input: *const c_void,
        output: *const c_void,
        output_length: usize,
        dim_count: usize,
        old_shape: *const c_void,
        new_shape: *const c_void,
    );
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

unsafe fn malloc(amount: usize, size: usize) -> *mut c_void {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let code = cudaMalloc(&mut ptr as *mut *mut c_void, amount * size);
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

unsafe fn memcpy_to_device_usize(ptr: *mut c_void, data: &Vec<usize>) {
    let code = cudaMemcpy(
        ptr,
        data.as_ptr() as *const c_void,
        data.len() * std::mem::size_of::<usize>(),
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
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            add(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Sub(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            sub(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Mul(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            mul(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Div(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            division(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Max(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_ptr = malloc(1, std::mem::size_of::<f64>());
            rusty_max(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            (result_ptr, vec![])
        }
        Op::Min(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_ptr = malloc(1, std::mem::size_of::<f64>());
            rusty_min(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            (result_ptr, vec![])
        }
        Op::Sqrt(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            rusty_sqrt(t_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Log(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            rusty_log(t_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Load(data, shape) => {
            let dev_ptr = malloc(data.len(), std::mem::size_of::<f64>());
            memcpy_to_device(dev_ptr, data);
            (dev_ptr, shape.to_vec())
        }
        Op::Sigmoid(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            sigmoid(t_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Relu(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            relu(t_ptr, result_ptr, result_size);
            check_last_error();
            (result_ptr, shape)
        }
        Op::Sum(_, _, _) => todo!(),
        Op::Pool2D(_, _, _, _, _) => todo!(),
        Op::Conv2D(_, _, _, _) => todo!(),
        Op::Pad2D(_, _, _) => todo!(),
        Op::Reshape(t, shape) => {
            let (t_ptr, _) = realize_cuda(&t.op);
            (t_ptr, shape.to_vec())
        }
        Op::Permute(_, _) => todo!(),
        Op::Expand(t, new_shape) => {
            let (t_ptr, old_shape) = realize_cuda(&t.op);
            assert_eq!(
                old_shape.len(),
                new_shape.len(),
                "Only supporting same size shapes"
            );
            for (old, new) in old_shape.iter().zip(new_shape.clone()) {
                assert!(
                    *old == new || *old == 1,
                    "Old dimension must be either 1 or identical to new dimension"
                );
            }
            let result_size = new_shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            let old_shape_ptr = malloc(old_shape.len(), std::mem::size_of::<usize>());
            memcpy_to_device_usize(old_shape_ptr, &old_shape);
            let new_shape_ptr = malloc(new_shape.len(), std::mem::size_of::<usize>());
            memcpy_to_device_usize(new_shape_ptr, &new_shape);
            expand(
                t_ptr,
                result_ptr,
                result_size,
                new_shape.len(),
                old_shape_ptr,
                new_shape_ptr,
            );
            check_last_error();
            (result_ptr, new_shape.to_vec())
        }
        Op::MatMul(lhs, rhs) => {
            let (lhs_ptr, lhs_shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, rhs_shape) = realize_cuda(&rhs.op);
            assert!(
                lhs_shape.len() == 2 && rhs_shape.len() == 2,
                "only supporting 2d tensors for now"
            );

            assert_eq!(lhs_shape[1], rhs_shape[0]);
            let result_size = lhs_shape[0] * rhs_shape[1];
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            matmul(
                lhs_ptr,
                rhs_ptr,
                result_ptr,
                lhs_shape[0],
                lhs_shape[1],
                rhs_shape[1],
            );
            check_last_error();
            (result_ptr, vec![lhs_shape[0], rhs_shape[1]])
        }
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
