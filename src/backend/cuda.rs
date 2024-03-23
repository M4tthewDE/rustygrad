use lazy_static::lazy_static;
use std::{
    collections::HashMap,
    ffi::{c_int, c_void, CStr},
    sync::Mutex,
};

use tracing::trace;

use crate::op::{Op, UnrealizedOp};

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

type OpCache = Mutex<HashMap<usize, (Vec<f64>, Vec<usize>)>>;

lazy_static! {
    static ref USE_CACHE: bool = std::env::var("NO_CACHE").is_err();
    static ref OP_CACHE: OpCache = Mutex::new(HashMap::new());
}

extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaFree(devPtr: *mut c_void) -> c_int;
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
    fn pad2d(
        input: *const c_void,
        output: *const c_void,
        output_length: usize,
        dim_count: usize,
        shape: *const c_void,
        new_shape: *const c_void,
        padding: *const c_void,
    );
    fn permute(
        input: *const c_void,
        output: *const c_void,
        output_length: usize,
        dim_count: usize,
        shape: *const c_void,
        new_shape: *const c_void,
        dims: *const c_void,
    );
    fn sum(
        input: *const c_void,
        result: *const c_void,
        n: usize,
        input_shape: *const c_void,
        input_shape_len: usize,
        dims: *const c_void,
        dims_len: usize,
        reduced_shape: *const c_void,
        reduced_shape_len: usize,
    );
    fn sum_pool2d(
        input: *const c_void,
        result: *const c_void,
        input_shape: *const c_void,
        result_shape: *const c_void,
        kernel: *const c_void,
        init_val: f64,
        stride: usize,
        batch: usize,
        channels: usize,
        output_height: usize,
    );
    fn max_pool2d(
        input: *const c_void,
        result: *const c_void,
        input_shape: *const c_void,
        result_shape: *const c_void,
        kernel: *const c_void,
        init_val: f64,
        stride: usize,
        output_height: usize,
        batch: usize,
        channels: usize,
    );
    fn conv2d(
        input: *const c_void,
        result: *const c_void,
        input_shape: *const c_void,
        result_shape: *const c_void,
        groups: usize,
        kernel_shape: *const c_void,
        kernel: *const c_void,
        strides: *const c_void,
        n: usize,
        c_out: usize,
        total_output_elements: usize,
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

unsafe fn cpy_to_device<T>(data: &[T]) -> *mut c_void {
    let ptr = malloc(data.len(), std::mem::size_of::<T>());
    let code = cudaMemcpy(
        ptr,
        data.as_ptr() as *const c_void,
        std::mem::size_of_val(data),
        HOST_TO_DEVICE,
    );
    if code != 0 {
        panic!("{}", error_string(code));
    }
    ptr
}

unsafe fn cpy_from_device(ptr: *mut c_void, shape: &[usize]) -> Vec<f64> {
    let mut result = vec![0.0; shape.iter().product()];
    let code = cudaMemcpy(
        result.as_mut_ptr() as *mut c_void,
        ptr,
        std::mem::size_of::<f64>() * result.len(),
        DEVICE_TO_HOST,
    );
    if code != 0 {
        panic!("{}", error_string(code));
    }

    cudaFree(ptr);
    result
}

unsafe fn realize_cuda(unrealized_op: &UnrealizedOp) -> (Vec<f64>, Vec<usize>) {
    trace!("realizing {:?}", unrealized_op);

    match &unrealized_op.op {
        Op::Add(lhs, rhs) => {
            let (lhs, shape) = realize(lhs);
            let (rhs, _) = realize(rhs);
            let lhs_ptr = cpy_to_device(&lhs);
            let rhs_ptr = cpy_to_device(&rhs);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            add(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Sub(lhs, rhs) => {
            let (lhs, shape) = realize(lhs);
            let (rhs, _) = realize(rhs);
            let lhs_ptr = cpy_to_device(&lhs);
            let rhs_ptr = cpy_to_device(&rhs);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            sub(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Mul(lhs, rhs) => {
            let (lhs, shape) = realize(lhs);
            let (rhs, _) = realize(rhs);
            let lhs_ptr = cpy_to_device(&lhs);
            let rhs_ptr = cpy_to_device(&rhs);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            mul(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Div(lhs, rhs) => {
            let (lhs, shape) = realize(lhs);
            let (rhs, _) = realize(rhs);
            let lhs_ptr = cpy_to_device(&lhs);
            let rhs_ptr = cpy_to_device(&rhs);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            division(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Max(t) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let result_ptr = malloc(1, std::mem::size_of::<f64>());
            rusty_max(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (cpy_from_device(result_ptr, &[]), vec![])
        }
        Op::Min(t) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let result_ptr = malloc(1, std::mem::size_of::<f64>());
            rusty_min(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (cpy_from_device(result_ptr, &[]), vec![])
        }
        Op::Sqrt(t) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            rusty_sqrt(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Log(t) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            rusty_log(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Load(data, shape) => (data.to_vec(), shape.to_vec()),
        Op::Sigmoid(t) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            sigmoid(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Relu(t) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            relu(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (cpy_from_device(result_ptr, &shape), shape)
        }
        Op::Sum(t, dims, keepdim) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let mut reduced_shape = shape.clone();
            for (i, dim) in dims.iter().enumerate() {
                reduced_shape.remove(*dim - i);
            }

            let result_size = reduced_shape.iter().product::<usize>();
            let result_ptr = cpy_to_device(&vec![0.0; result_size]);
            let input_shape_ptr = cpy_to_device(&shape);
            let dims_ptr = cpy_to_device(dims);
            let reduced_shape_ptr = cpy_to_device(&reduced_shape);

            sum(
                t_ptr,
                result_ptr,
                shape.iter().product(),
                input_shape_ptr,
                shape.len(),
                dims_ptr,
                dims.len(),
                reduced_shape_ptr,
                reduced_shape.len(),
            );

            let new_shape = if *keepdim {
                shape
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                    .collect()
            } else {
                reduced_shape
            };
            (cpy_from_device(result_ptr, &new_shape), new_shape)
        }
        Op::Pool2D(t, kernel, stride, init_val, pool_op) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            // FIXME: remove this constraint, just reshape or something smarter
            assert_eq!(shape.len(), 4, "only supporting 4d tensors");

            let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let (kernel_height, kernel_width) = (kernel.0, kernel.1);

            let output_height = ((height - kernel_height) / stride) + 1;
            let output_width = ((width - kernel_width) / stride) + 1;

            let result_shape = vec![batch, channels, output_height, output_width];

            let result_ptr = cpy_to_device(&vec![0.0; result_shape.iter().product()]);
            let kernel_ptr = cpy_to_device(&[kernel.0, kernel.1]);
            let shape_ptr = cpy_to_device(&shape);
            let result_shape_ptr = cpy_to_device(&result_shape);

            match pool_op {
                crate::op::PoolOp::Sum => sum_pool2d(
                    t_ptr,
                    result_ptr,
                    shape_ptr,
                    result_shape_ptr,
                    kernel_ptr,
                    *init_val,
                    *stride,
                    shape[0],
                    shape[1],
                    output_height,
                ),
                crate::op::PoolOp::Max => max_pool2d(
                    t_ptr,
                    result_ptr,
                    shape_ptr,
                    result_shape_ptr,
                    kernel_ptr,
                    *init_val,
                    *stride,
                    output_height,
                    batch,
                    channels,
                ),
            }
            check_last_error();

            cudaFree(t_ptr);
            cudaFree(kernel_ptr);
            cudaFree(shape_ptr);
            cudaFree(result_shape_ptr);

            (cpy_from_device(result_ptr, &result_shape), result_shape)
        }
        Op::Conv2D(t, kernel, strides, groups) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            let (kernel, kernel_shape) = realize_cuda(kernel);
            let kernel_ptr = cpy_to_device(&kernel);

            assert_eq!(shape.len(), 4, "only supporting 4d tensors");
            assert_eq!(kernel_shape.len(), 4, "only supporting 4d kernels");
            assert_eq!(
                shape[1] % groups,
                0,
                "input channels must be divisible by groups"
            );
            assert_eq!(
                kernel_shape[0] % groups,
                0,
                "output channels must be divisible by groups"
            );

            let (n, _, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let (c_out, kernel_height, kernel_width) =
                (kernel_shape[0], kernel_shape[2], kernel_shape[3]);

            let output_height = ((height - kernel_height) / strides.0) + 1;
            let output_width = ((width - kernel_width) / strides.1) + 1;
            let new_shape = vec![n, c_out, output_height, output_width];

            let result_ptr = cpy_to_device(&vec![0.0; new_shape.iter().product()]);
            let shape_ptr = cpy_to_device(&shape);
            let new_shape_ptr = cpy_to_device(&new_shape);
            let kernel_shape_ptr = cpy_to_device(&kernel_shape);
            let strides_ptr = cpy_to_device(&[strides.0, strides.1]);

            conv2d(
                t_ptr,
                result_ptr,
                shape_ptr,
                new_shape_ptr,
                *groups,
                kernel_shape_ptr,
                kernel_ptr,
                strides_ptr,
                n,
                c_out,
                output_height * output_width,
            );
            check_last_error();

            cudaFree(t_ptr);
            cudaFree(shape_ptr);
            cudaFree(new_shape_ptr);
            cudaFree(kernel_shape_ptr);
            cudaFree(kernel_ptr);
            cudaFree(strides_ptr);

            (cpy_from_device(result_ptr, &new_shape), new_shape)
        }
        Op::Pad2D(t, value, padding) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            if shape.len() < 2 {
                panic!("Tensor must have at least 2 dimensions for 2D padding.");
            }

            let last_two_dims = shape.len() - 2;
            let mut new_shape: Vec<usize> = shape.clone();

            new_shape[last_two_dims] += padding[2] + padding[3]; // top + bottom
            new_shape[last_two_dims + 1] += padding[0] + padding[1]; // left + right

            let result_ptr = cpy_to_device(&vec![*value; new_shape.iter().product()]);
            let shape_ptr = cpy_to_device(&shape);
            let new_shape_ptr = cpy_to_device(&new_shape);
            let padding_ptr = cpy_to_device(padding.as_ref());

            pad2d(
                t_ptr,
                result_ptr,
                shape.iter().product(),
                shape.len(),
                shape_ptr,
                new_shape_ptr,
                padding_ptr,
            );
            check_last_error();

            cudaFree(t_ptr);
            cudaFree(shape_ptr);
            cudaFree(new_shape_ptr);
            cudaFree(padding_ptr);

            (cpy_from_device(result_ptr, &new_shape), new_shape)
        }
        Op::Reshape(t, shape) => {
            let (t_ptr, _) = realize_cuda(t);
            (t_ptr, shape.to_vec())
        }
        Op::Permute(t, dims) => {
            let (t, shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
            if shape.len() < 2 {
                panic!("Tensor must have at least 2 dimensions for 2D padding.");
            }

            let new_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
            let result_ptr = cpy_to_device(&vec![0.0; new_shape.iter().product()]);
            let shape_ptr = cpy_to_device(&shape);
            let new_shape_ptr = cpy_to_device(&new_shape);
            let dims_ptr = cpy_to_device(&dims.to_vec());

            permute(
                t_ptr,
                result_ptr,
                shape.iter().product(),
                shape.len(),
                shape_ptr,
                new_shape_ptr,
                dims_ptr,
            );
            check_last_error();

            cudaFree(t_ptr);
            cudaFree(shape_ptr);
            cudaFree(new_shape_ptr);
            cudaFree(dims_ptr);

            (cpy_from_device(result_ptr, &new_shape), new_shape)
        }
        Op::Expand(t, new_shape) => {
            let (t, old_shape) = realize_cuda(t);
            let t_ptr = cpy_to_device(&t);
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
            let old_shape_ptr = cpy_to_device(&old_shape);
            let new_shape_ptr = cpy_to_device(new_shape);
            expand(
                t_ptr,
                result_ptr,
                result_size,
                new_shape.len(),
                old_shape_ptr,
                new_shape_ptr,
            );
            cudaFree(t_ptr);
            check_last_error();
            (cpy_from_device(result_ptr, new_shape), new_shape.to_vec())
        }
        Op::MatMul(lhs, rhs) => {
            let (lhs, lhs_shape) = realize_cuda(lhs);
            let lhs_ptr = cpy_to_device(&lhs);
            let (rhs, rhs_shape) = realize_cuda(rhs);
            let rhs_ptr = cpy_to_device(&rhs);
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
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            let new_shape = vec![lhs_shape[0], rhs_shape[1]];
            (cpy_from_device(result_ptr, &new_shape), new_shape.to_vec())
        }
    }
}

pub fn realize(unrealized_op: &UnrealizedOp) -> (Vec<f64>, Vec<usize>) {
    if *USE_CACHE {
        {
            let cache = OP_CACHE.lock().unwrap();
            if let Some(result) = cache.get(&unrealized_op.id) {
                return result.clone();
            }
        }
    }
    unsafe {
        let result = realize_cuda(unrealized_op);
        if *USE_CACHE {
            let mut cache = OP_CACHE.lock().unwrap();
            cache.insert(unrealized_op.id, result.clone());
        }

        result
    }
}
