use std::ffi::{c_int, c_void, CStr};

use tracing::trace;

use crate::op::Op;

const HOST_TO_DEVICE: c_int = 1;
const DEVICE_TO_HOST: c_int = 2;

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
    );
    fn max_pool2d(
        input: *const c_void,
        result: *const c_void,
        input_shape: *const c_void,
        result_shape: *const c_void,
        kernel: *const c_void,
        init_val: f64,
        stride: usize,
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

unsafe fn realize_cuda(op: &Op) -> (*mut c_void, Vec<usize>) {
    match op {
        Op::Add(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            add(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (result_ptr, shape)
        }
        Op::Sub(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            sub(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (result_ptr, shape)
        }
        Op::Mul(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            mul(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (result_ptr, shape)
        }
        Op::Div(lhs, rhs) => {
            let (lhs_ptr, shape) = realize_cuda(&lhs.op);
            let (rhs_ptr, _) = realize_cuda(&rhs.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            division(lhs_ptr, rhs_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
            (result_ptr, shape)
        }
        Op::Max(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_ptr = malloc(1, std::mem::size_of::<f64>());
            rusty_max(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (result_ptr, vec![])
        }
        Op::Min(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_ptr = malloc(1, std::mem::size_of::<f64>());
            rusty_min(t_ptr, result_ptr, shape.iter().product());
            check_last_error();
            cudaFree(t_ptr);
            (result_ptr, vec![])
        }
        Op::Sqrt(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            rusty_sqrt(t_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(t_ptr);
            (result_ptr, shape)
        }
        Op::Log(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            rusty_log(t_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(t_ptr);
            (result_ptr, shape)
        }
        Op::Load(data, shape) => (cpy_to_device(data), shape.to_vec()),
        Op::Sigmoid(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            sigmoid(t_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(t_ptr);
            (result_ptr, shape)
        }
        Op::Relu(t) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
            let result_size = shape.iter().product::<usize>();
            let result_ptr = malloc(result_size, std::mem::size_of::<f64>());
            relu(t_ptr, result_ptr, result_size);
            check_last_error();
            cudaFree(t_ptr);
            (result_ptr, shape)
        }
        Op::Sum(t, dims, keepdim) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
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
            (result_ptr, new_shape)
        }
        Op::Pool2D(t, kernel, stride, init_val, pool_op) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
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
                ),
                crate::op::PoolOp::Max => max_pool2d(
                    t_ptr,
                    result_ptr,
                    shape_ptr,
                    result_shape_ptr,
                    kernel_ptr,
                    *init_val,
                    *stride,
                ),
            }
            check_last_error();

            cudaFree(t_ptr);
            cudaFree(kernel_ptr);
            cudaFree(shape_ptr);
            cudaFree(result_shape_ptr);

            (result_ptr, result_shape.to_vec())
        }
        Op::Conv2D(_, _, _, _) => todo!(),
        Op::Pad2D(t, value, padding) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
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

            (result_ptr, new_shape.to_vec())
        }
        Op::Reshape(t, shape) => {
            let (t_ptr, _) = realize_cuda(&t.op);
            (t_ptr, shape.to_vec())
        }
        Op::Permute(t, dims) => {
            let (t_ptr, shape) = realize_cuda(&t.op);
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

            (result_ptr, new_shape.to_vec())
        }
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
            cudaFree(lhs_ptr);
            cudaFree(rhs_ptr);
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
        let code = cudaMemcpy(
            result.as_mut_ptr() as *mut c_void,
            result_ptr,
            result_size * std::mem::size_of::<f64>(),
            DEVICE_TO_HOST,
        );
        if code != 0 {
            panic!("{}", error_string(code));
        }
        cudaFree(result_ptr);
        (result, shape)
    }
}
