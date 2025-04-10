use std::rc::Rc;

use ocl::{Buffer, ProQue};
use tracing::trace;

use crate::op::{Op, OpCache, UnrealizedOp};

fn readonly_buffer(data: &[f64], que: &ProQue) -> Buffer<f64> {
    Buffer::builder()
        .queue(que.queue().clone())
        .flags(ocl::MemFlags::READ_ONLY)
        .len(data.len())
        .copy_host_slice(data)
        .build()
        .unwrap()
}

fn write_buffer(len: usize, que: &ProQue) -> Buffer<f64> {
    Buffer::builder()
        .queue(que.queue().clone())
        .flags(ocl::MemFlags::WRITE_ONLY)
        .len(len)
        .build()
        .unwrap()
}

pub fn realize(unrealized_op: &UnrealizedOp, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    if let Some(result) = cache.get(&unrealized_op.id) {
        return result.clone();
    }

    trace!("Realizing {:?}", unrealized_op);

    let result = match &unrealized_op.op {
        Op::Add(lhs, rhs) => add(lhs, rhs, cache),
        Op::Sub(lhs, rhs) => sub(lhs, rhs, cache),
        Op::Mul(lhs, rhs) => mul(lhs, rhs, cache),
        Op::Div(lhs, rhs) => div(lhs, rhs, cache),
        Op::Sqrt(t) => sqrt(t, cache),
        Op::Log(t) => log(t, cache),
        Op::Sigmoid(t) => sigmoid(t, cache),
        Op::Relu(t) => relu(t, cache),
        Op::Max(t) => max(t, cache),
        Op::Load(data, shape) => load(data, shape),
        Op::Reshape(t, shape) => reshape(t, shape, cache),
        op => todo!("{}", op),
    };

    cache.insert(unrealized_op.id, result.clone());

    result
}

fn load(data: &[f64], shape: &[usize]) -> (Vec<f64>, Vec<usize>) {
    (data.to_owned(), shape.to_owned())
}

fn reshape(
    t: &Rc<UnrealizedOp>,
    shape: &Vec<usize>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let (data, _) = realize(t, cache);
    (data, shape.to_owned())
}

fn add(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void add_kernel(__global double *a, __global double *b, __global double *c, int n) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }
    "#;

    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(lhs_data.len())
        .build()
        .unwrap();
    let lhs_buffer = readonly_buffer(&lhs_data, &que);
    let rhs_buffer = readonly_buffer(&rhs_data, &que);
    let res_buffer = write_buffer(lhs_data.len(), &que);

    let kernel = que
        .kernel_builder("add_kernel")
        .arg(&lhs_buffer)
        .arg(&rhs_buffer)
        .arg(&res_buffer)
        .arg(lhs_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; lhs_data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn sub(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void sub_kernel(__global double *a, __global double *b, __global double *c, int n) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] - b[i];
            }
        }
    "#;

    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(lhs_data.len())
        .build()
        .unwrap();
    let lhs_buffer = readonly_buffer(&lhs_data, &que);
    let rhs_buffer = readonly_buffer(&rhs_data, &que);
    let res_buffer = write_buffer(lhs_data.len(), &que);

    let kernel = que
        .kernel_builder("sub_kernel")
        .arg(&lhs_buffer)
        .arg(&rhs_buffer)
        .arg(&res_buffer)
        .arg(lhs_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; lhs_data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn mul(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void mul_kernel(__global double *a, __global double *b, __global double *c, int n) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] * b[i];
            }
        }
    "#;

    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(lhs_data.len())
        .build()
        .unwrap();
    let lhs_buffer = readonly_buffer(&lhs_data, &que);
    let rhs_buffer = readonly_buffer(&rhs_data, &que);
    let res_buffer = write_buffer(lhs_data.len(), &que);

    let kernel = que
        .kernel_builder("mul_kernel")
        .arg(&lhs_buffer)
        .arg(&rhs_buffer)
        .arg(&res_buffer)
        .arg(lhs_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; lhs_data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn div(
    lhs: &Rc<UnrealizedOp>,
    rhs: &Rc<UnrealizedOp>,
    cache: &mut OpCache,
) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void div_kernel(__global double *a, __global double *b, __global double *c, int n) {
            int i = get_global_id(0);
            if (i < n) {
                c[i] = a[i] / b[i];
            }
        }
    "#;

    let (lhs_data, lhs_shape) = realize(lhs, cache);
    let (rhs_data, _) = realize(rhs, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(lhs_data.len())
        .build()
        .unwrap();
    let lhs_buffer = readonly_buffer(&lhs_data, &que);
    let rhs_buffer = readonly_buffer(&rhs_data, &que);
    let res_buffer = write_buffer(lhs_data.len(), &que);

    let kernel = que
        .kernel_builder("div_kernel")
        .arg(&lhs_buffer)
        .arg(&rhs_buffer)
        .arg(&res_buffer)
        .arg(lhs_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; lhs_data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn sqrt(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void sqrt_kernel(__global double *a, __global double *b, int n) {
            int i = get_global_id(0);
            if (i < n) {
                b[i] = sqrt(a[i]);
            }
        }
    "#;

    let (data, lhs_shape) = realize(t, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(data.len())
        .build()
        .unwrap();
    let data_buffer = readonly_buffer(&data, &que);
    let res_buffer = write_buffer(data.len(), &que);

    let kernel = que
        .kernel_builder("sqrt_kernel")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .arg(data_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn log(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void log_kernel(__global double *a, __global double *b, int n) {
            int i = get_global_id(0);
            if (i < n) {
                b[i] = log2(a[i]);
            }
        }
    "#;

    let (data, lhs_shape) = realize(t, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(data.len())
        .build()
        .unwrap();
    let data_buffer = readonly_buffer(&data, &que);
    let res_buffer = write_buffer(data.len(), &que);

    let kernel = que
        .kernel_builder("log_kernel")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .arg(data_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn sigmoid(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void sigmoid_kernel(__global double *a, __global double *b, int n) {
            int i = get_global_id(0);
            if (i < n) {
                b[i] = (1.0 / (1.0 + exp(-a[i])));
            }
        }
    "#;

    let (data, lhs_shape) = realize(t, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(data.len())
        .build()
        .unwrap();
    let data_buffer = readonly_buffer(&data, &que);
    let res_buffer = write_buffer(data.len(), &que);

    let kernel = que
        .kernel_builder("sigmoid_kernel")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .arg(data_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn relu(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void relu_kernel(__global double *a, __global double *b, int n) {
            int i = get_global_id(0);
            if (i < n) {
                b[i] = max(a[i], 0.0);
            }
        }
    "#;

    let (data, lhs_shape) = realize(t, cache);

    let que = ProQue::builder()
        .src(kernel)
        .dims(data.len())
        .build()
        .unwrap();
    let data_buffer = readonly_buffer(&data, &que);
    let res_buffer = write_buffer(data.len(), &que);

    let kernel = que
        .kernel_builder("relu_kernel")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .arg(data_buffer.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; data.len()];
    res_buffer.read(&mut res).enq().unwrap();

    (res, lhs_shape)
}

fn max(t: &Rc<UnrealizedOp>, cache: &mut OpCache) -> (Vec<f64>, Vec<usize>) {
    let kernel = r#"
        __kernel void max_kernel(__global const double *a, __global double *b, int n) {
            int global_size = get_global_size(0);
            int global_id = get_global_id(0);

            int chunk_size = n/global_size;

            for (int i = global_id * chunk_size; i < ((global_id+1)*chunk_size); i++) {
                if (a[i] > b[global_id]) {
                    b[global_id] = a[i];
                }
            }

            if (global_id == 0) {
                for (int i = 0; i < global_size; i++) {
                    if (b[i] > b[0]) {
                        b[0] = b[i];
                    }
                }
            }
        }
    "#;

    let (data, _) = realize(t, cache);

    let global_size = if data.len() < 16 { 1 } else { data.len() / 16 };

    let que = ProQue::builder()
        .src(kernel)
        .dims(global_size)
        .build()
        .unwrap();
    let data_buffer = readonly_buffer(&data, &que);
    let res_buffer = write_buffer(global_size, &que);

    let kernel = que
        .kernel_builder("max_kernel")
        .arg(&data_buffer)
        .arg(&res_buffer)
        .arg(data.len() as i32)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut res = vec![0.0; 1];
    res_buffer.read(&mut res).enq().unwrap();
    (res, vec![])
}
