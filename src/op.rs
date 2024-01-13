use lazy_static::lazy_static;
use std::{
    collections::HashMap,
    env,
    fmt::{Debug, Display},
    hash::Hash,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
};

use crate::{
    backend::{cpu, cuda},
    device::Device,
};

type OpCache = Mutex<HashMap<usize, (Vec<f64>, Vec<usize>)>>;

lazy_static! {
    static ref OP_COUNTER: AtomicUsize = AtomicUsize::new(0);
    static ref OP_CACHE: OpCache = Mutex::new(HashMap::new());
    static ref USE_CACHE: bool = env::var("NO_CACHE").is_err();
}

#[derive(Clone)]
pub struct UnrealizedOp {
    pub id: usize,
    pub op: Op,
    pub device: Device,
}

impl UnrealizedOp {
    pub fn new(op: Op, device: Device) -> UnrealizedOp {
        let id = OP_COUNTER.fetch_add(1, Ordering::Relaxed);

        UnrealizedOp { id, op, device }
    }

    pub fn realize(&self) -> (Vec<f64>, Vec<usize>) {
        if *USE_CACHE {
            {
                let cache = OP_CACHE.lock().unwrap();
                if let Some(result) = cache.get(&self.id) {
                    return result.clone();
                }
            }
        }

        // guaranteed to have at least 32K of stack
        stacker::maybe_grow(32 * 1024, 1024 * 1024, || {
            let result = match self.device {
                Device::Cpu => cpu::realize(&self.op),
                Device::Cuda => cuda::realize(&self.op),
            };

            if *USE_CACHE {
                let mut cache = OP_CACHE.lock().unwrap();
                cache.insert(self.id, result.clone());
            }
            result
        })
    }
}

impl Display for UnrealizedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.op)
    }
}

impl Debug for UnrealizedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.id, self.op)
    }
}

impl Hash for UnrealizedOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
impl Eq for UnrealizedOp {}

impl PartialEq for UnrealizedOp {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

pub type PoolOp = fn(lhs: f64, rhs: f64) -> f64;

#[derive(Clone)]
pub enum Op {
    Add(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Sub(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Mul(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Div(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
    Max(Rc<UnrealizedOp>),
    Min(Rc<UnrealizedOp>),
    Sqrt(Rc<UnrealizedOp>),
    Log(Rc<UnrealizedOp>),
    Load(Vec<f64>, Vec<usize>),
    Sigmoid(Rc<UnrealizedOp>),
    Relu(Rc<UnrealizedOp>),
    Sum(Rc<UnrealizedOp>, Vec<usize>, bool),
    Pool2D(Rc<UnrealizedOp>, (usize, usize), usize, f64, PoolOp),
    Conv2D(Rc<UnrealizedOp>, Rc<UnrealizedOp>, (usize, usize), usize),
    Pad2D(Rc<UnrealizedOp>, f64, [usize; 4]),
    Reshape(Rc<UnrealizedOp>, Vec<usize>),
    Permute(Rc<UnrealizedOp>, Vec<usize>),
    Expand(Rc<UnrealizedOp>, Vec<usize>),
    MatMul(Rc<UnrealizedOp>, Rc<UnrealizedOp>),
}

impl Debug for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::Add(_, _) => write!(f, "Add"),
            Op::Sub(_, _) => write!(f, "Sub"),
            Op::Mul(_, _) => write!(f, "Mul"),
            Op::Div(_, _) => write!(f, "Div"),
            Op::Max(_) => write!(f, "Max"),
            Op::Min(_) => write!(f, "Min"),
            Op::Sqrt(_) => write!(f, "Sqrt"),
            Op::Log(_) => write!(f, "Log"),
            Op::Load(_, _) => write!(f, "Load"),
            Op::Sigmoid(_) => write!(f, "Sigmoid"),
            Op::Relu(_) => write!(f, "Relu"),
            Op::Sum(_, _, _) => write!(f, "Sum"),
            Op::Pool2D(_, _, _, _, _) => write!(f, "Pool2D"),
            Op::Conv2D(_, _, _, _) => write!(f, "Conv2D"),
            Op::Pad2D(_, _, _) => write!(f, "Pad2D"),
            Op::Reshape(_, _) => write!(f, "Reshape"),
            Op::Permute(_, _) => write!(f, "Permute"),
            Op::Expand(_, _) => write!(f, "Expand"),
            Op::MatMul(_, _) => write!(f, "MatMul"),
        }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
