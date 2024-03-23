use std::fmt::Debug;
use std::iter::zip;
use std::rc::Rc;
use std::{cmp, env, ops};

use image::DynamicImage;
use rand::{distributions::Uniform, prelude::Distribution};
use tracing::debug;

use crate::device::Device;
use crate::graph;
use crate::op::{Op, PoolOp, UnrealizedOp};

#[derive(Clone)]
pub struct Tensor {
    pub unrealized_op: Rc<UnrealizedOp>,
    pub data: Option<Vec<f64>>,
    pub shape: Vec<usize>,
    pub device: Device,
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.unrealized_op)
    }
}

impl Tensor {
    fn from_op(op: Op, shape: &[usize]) -> Tensor {
        Tensor {
            unrealized_op: Rc::new(UnrealizedOp::new(op)),
            data: None,
            shape: shape.to_owned(),
            device: Device::default(),
        }
    }

    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(Op::Load(data, shape.clone()), &shape)
    }

    pub fn from_vec_single_dim(data: Vec<f64>) -> Tensor {
        let data_len = data.len();
        Tensor::from_op(Op::Load(data, vec![data_len]), &[data_len])
    }

    pub fn from_scalar(data: f64) -> Tensor {
        Tensor::from_op(Op::Load(vec![data], vec![1]), &[1])
    }

    pub fn from_image(img: DynamicImage) -> Tensor {
        let shape = vec![img.width() as usize, img.height() as usize, 3];
        let data: Vec<f64> = img
            .to_rgb8()
            .pixels()
            .flat_map(|p| p.0.map(|x| x as f64))
            .collect();

        Tensor::from_op(Op::Load(data, shape.clone()), &shape)
    }

    pub fn zeros(size: usize) -> Tensor {
        Tensor::from_op(Op::Load(vec![0.0; size], vec![size]), &[size])
    }

    pub fn ones(size: usize) -> Tensor {
        Tensor::from_op(Op::Load(vec![1.0; size], vec![size]), &[size])
    }

    pub fn glorot_uniform(shape: Vec<usize>) -> Tensor {
        let limit = (6.0 / (shape[0] + shape[1..].iter().product::<usize>()) as f64).sqrt();
        let data = Uniform::new(0.0, limit)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(Op::Load(data, shape.clone()), &shape)
    }

    pub fn rand(shape: Vec<usize>) -> Tensor {
        // feels like this should happen in an op
        let data = Uniform::new(-1.0, 1.0)
            .sample_iter(rand::thread_rng())
            .take(shape.iter().product::<usize>())
            .collect();

        Tensor::from_op(Op::Load(data, shape.clone()), &shape)
    }

    pub fn max(&self) -> Tensor {
        Tensor::from_op(Op::Max(self.unrealized_op.clone()), &[1])
    }

    pub fn min(&self) -> Tensor {
        Tensor::from_op(Op::Min(self.unrealized_op.clone()), &[1])
    }

    pub fn sqrt(&self) -> Tensor {
        Tensor::from_op(Op::Sqrt(self.unrealized_op.clone()), &self.shape)
    }

    pub fn rsqrt(&self) -> Tensor {
        1.0 / self.sqrt()
    }

    pub fn log(&self) -> Tensor {
        Tensor::from_op(Op::Log(self.unrealized_op.clone()), &self.shape)
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor::from_op(Op::Sigmoid(self.unrealized_op.clone()), &self.shape)
    }

    pub fn swish(&self) -> Tensor {
        self.clone() * self.sigmoid()
    }

    pub fn relu(&self) -> Tensor {
        Tensor::from_op(Op::Relu(self.unrealized_op.clone()), &self.shape)
    }

    pub fn reduce_sum(&self, dims: Option<Vec<usize>>, keepdim: bool) -> Tensor {
        let dims = dims.unwrap_or((0..self.shape.len()).collect());
        let shape = if keepdim {
            self.shape
                .iter()
                .enumerate()
                .map(|(i, &d)| if dims.contains(&i) { 1 } else { d })
                .collect()
        } else {
            let mut reduced_shape = self.shape.clone();
            for (i, dim) in dims.iter().enumerate() {
                reduced_shape.remove(*dim - i);
            }

            reduced_shape
        };
        Tensor::from_op(Op::Sum(self.unrealized_op.clone(), dims, keepdim), &shape)
    }

    // 0 0 0 0 0
    // 0 0 0 0 0
    // 0 0 0 0 0
    // 0 0 0 0 0
    // 0 0 0 0 0
    //
    // a b c d e
    // f g h i j
    // k l m n o
    // p q r s t
    // u v w x y
    //
    // [[a,b,f,g], [b,c,g,h], [c,d,h,i], [d,e,i,j]]
    //
    // pool with kernel (2,2) and stride 1
    // [5, 5] ->[4, 4]
    //
    // turning [5, 5] into [16, 4] is the challenge!
    //
    // [16, 4]
    // sum:
    // [16, 1]
    // then just reshape to [4, 4] ?

    pub fn avg_pool_2d(&self, kernel: (usize, usize), stride: Option<usize>) -> Tensor {
        let stride = stride.unwrap_or(1);
        Tensor::from_op(
            Op::Pool2D(self.unrealized_op.clone(), kernel, stride, 0.0, PoolOp::Sum),
            &[
                self.shape[0],
                self.shape[1],
                ((self.shape[2] - kernel.0) / stride) + 1,
                ((self.shape[3] - kernel.1) / stride) + 1,
            ],
        ) / (kernel.0 * kernel.1) as f64
    }

    pub fn max_pool_2d(&self, kernel: (usize, usize), stride: Option<usize>) -> Tensor {
        let stride = stride.unwrap_or(1);
        Tensor::from_op(
            Op::Pool2D(
                self.unrealized_op.clone(),
                kernel,
                stride,
                f64::MIN,
                PoolOp::Max,
            ),
            &[
                self.shape[0],
                self.shape[1],
                ((self.shape[2] - kernel.0) / stride) + 1,
                ((self.shape[3] - kernel.1) / stride) + 1,
            ],
        )
    }

    pub fn pad_2d(&self, value: f64, padding: [usize; 4]) -> Tensor {
        let mut new_shape: Vec<usize> = self.shape.clone();
        new_shape[self.shape.len() - 2] += padding[2] + padding[3];
        new_shape[self.shape.len() - 1] += padding[0] + padding[1];
        Tensor::from_op(
            Op::Pad2D(self.unrealized_op.clone(), value, padding),
            &new_shape,
        )
    }

    pub fn conv2d(
        &self,
        kernel: &Tensor,
        bias: Option<Tensor>,
        padding: Option<[usize; 4]>,
        strides: Option<(usize, usize)>,
        groups: Option<usize>,
    ) -> Tensor {
        let x = if let Some(padding) = padding {
            self.pad_2d(0.0, padding)
        } else {
            self.clone()
        };

        let res = Tensor::from_op(
            Op::Conv2D(
                x.unrealized_op,
                kernel.unrealized_op.clone(),
                strides.unwrap_or((1, 1)),
                groups.unwrap_or(1),
            ),
            &[
                x.shape[0],
                kernel.shape[0],
                ((x.shape[2] - kernel.shape[2]) / strides.unwrap_or((1, 1)).0) + 1,
                ((x.shape[3] - kernel.shape[3]) / strides.unwrap_or((1, 1)).1) + 1,
            ],
        );

        if let Some(bias) = bias {
            res + bias
        } else {
            res
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(
            Op::Reshape(self.unrealized_op.clone(), shape.clone()),
            &shape,
        )
    }

    pub fn permute(&self, dims: Vec<usize>) -> Tensor {
        Tensor::from_op(
            Op::Permute(self.unrealized_op.clone(), dims.clone()),
            &dims.iter().map(|&d| self.shape[d]).collect::<Vec<usize>>(),
        )
    }

    pub fn expand(&self, shape: Vec<usize>) -> Tensor {
        Tensor::from_op(
            Op::Expand(self.unrealized_op.clone(), shape.clone()),
            &shape,
        )
    }

    pub fn matmul(&self, rhs: &Tensor) -> Tensor {
        Tensor::from_op(
            Op::MatMul(self.unrealized_op.clone(), rhs.unrealized_op.clone()),
            &[self.shape[0], rhs.shape[1]],
        )
    }

    pub fn batchnorm(
        &self,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        mean: Tensor,
        invstd: Tensor,
    ) -> Tensor {
        let mean_shape = mean.shape.clone();
        let x = self.clone() - mean.reshape(vec![1, mean_shape[0], 1, 1]);
        let x = if let Some(weight) = weight {
            let shape = weight.shape.clone();
            x * weight.reshape(vec![1, shape[0], 1, 1])
        } else {
            x
        };

        let ret = if invstd.shape.len() == 1 {
            let shape = invstd.shape.clone();
            x * (invstd.reshape(vec![1, shape[1], 1, 1]))
        } else {
            x * invstd
        };

        if let Some(bias) = bias {
            let shape = bias.shape.clone();
            ret + bias.reshape(vec![1, shape[0], 1, 1])
        } else {
            ret
        }
    }

    pub fn linear(&self, weight: &Tensor, bias: Option<Tensor>) -> Tensor {
        match bias {
            Some(bias) => self.matmul(weight) + bias.clone(),
            None => self.matmul(weight),
        }
    }

    pub fn realize(&self) -> (Vec<f64>, Vec<usize>) {
        if env::var("GRAPH").is_ok() {
            graph::build_graph(self);
        }

        debug!("realizing tensor");
        self.unrealized_op.realize(&self.device)
    }
}

impl ops::Add<f64> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_scalar(rhs);
        self + rhs
    }
}

impl ops::Add<Tensor> for f64 {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        let lhs = Tensor::from_scalar(self);
        lhs + rhs
    }
}

impl ops::Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        // FIXME: these reshapes shouldn't be necessary all the time,
        // but running into hard to track down bug without them
        // (shape gets wrong, maybe shape tracking is off somewhere)
        // one fix should solve all binary ops
        let (l, r, shape) = broadcast_shapes(&self.shape, &rhs.shape);
        let lhs = if l != shape {
            self.reshape(l).expand(shape.clone())
        } else {
            self.reshape(l)
        };
        let rhs = if r != shape {
            rhs.reshape(r).expand(shape.clone())
        } else {
            rhs.reshape(r)
        };

        Tensor::from_op(Op::Add(lhs.unrealized_op, rhs.unrealized_op), &shape)
    }
}

impl ops::Sub<f64> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_scalar(rhs);
        self - rhs
    }
}

impl ops::Sub<Tensor> for f64 {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        let lhs = Tensor::from_scalar(self);
        lhs - rhs
    }
}

impl ops::Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let (l, r, shape) = broadcast_shapes(&self.shape, &rhs.shape);
        let lhs = if l != shape {
            self.reshape(l).expand(shape.clone())
        } else {
            self.reshape(l)
        };
        let rhs = if r != shape {
            rhs.reshape(r).expand(shape.clone())
        } else {
            rhs.reshape(r)
        };

        Tensor::from_op(Op::Sub(lhs.unrealized_op, rhs.unrealized_op), &shape)
    }
}

impl ops::Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_scalar(rhs);
        self * rhs
    }
}

impl ops::Mul<Tensor> for f64 {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        let lhs = Tensor::from_scalar(self);
        lhs * rhs
    }
}

impl ops::Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let (l, r, shape) = broadcast_shapes(&self.shape, &rhs.shape);
        let lhs = if l != shape {
            self.reshape(l).expand(shape.clone())
        } else {
            self.reshape(l)
        };
        let rhs = if r != shape {
            rhs.reshape(r).expand(shape.clone())
        } else {
            rhs.reshape(r)
        };

        Tensor::from_op(Op::Mul(lhs.unrealized_op, rhs.unrealized_op), &shape)
    }
}

impl ops::Div<f64> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::from_scalar(rhs);
        self / rhs
    }
}

impl ops::Div<Tensor> for f64 {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        let lhs = Tensor::from_scalar(self);
        lhs / rhs
    }
}

impl ops::Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        let (l, r, shape) = broadcast_shapes(&self.shape, &rhs.shape);
        let lhs = if l != shape {
            self.reshape(l).expand(shape.clone())
        } else {
            self.reshape(l)
        };
        let rhs = if r != shape {
            rhs.reshape(r).expand(shape.clone())
        } else {
            rhs.reshape(r)
        };

        Tensor::from_op(Op::Div(lhs.unrealized_op, rhs.unrealized_op), &shape)
    }
}

fn broadcast_shapes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut lhs_shape = lhs_shape.to_owned();
    let mut rhs_shape = rhs_shape.to_owned();

    assert!(
        lhs_shape
            .iter()
            .rev()
            .zip(rhs_shape.iter().rev())
            .all(|(&left, &right)| left == right || left == 1 || right == 1),
        "{:?} and {:?} aren't broadcastable",
        lhs_shape,
        rhs_shape
    );

    let max_len = lhs_shape.len().max(rhs_shape.len());
    while rhs_shape.len() < max_len {
        rhs_shape.insert(0, 1);
    }

    while lhs_shape.len() < max_len {
        lhs_shape.insert(0, 1);
    }

    let shape: Vec<usize> = zip(lhs_shape.clone(), rhs_shape.clone())
        .map(|(d1, d2)| cmp::max(d1, d2))
        .collect();

    (lhs_shape, rhs_shape, shape)
}
