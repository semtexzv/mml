
pub mod graph;
pub (crate) mod tmap;
pub mod eval;
pub mod optim;
mod data;

use std::borrow::Borrow;
use std::ops::{FnOnce, Index, IndexMut};
use std::string::String;
use rand::random;
use smallvec::SmallVec;
use crate::eval::CPU;
use crate::graph::CGraph;
use crate::optim::{Optimizer, SGD};

type Shape = [usize; 4];

pub fn prod(s: Shape) -> usize {
    s.iter().product()
}

pub fn sprod(s: &[usize]) -> usize {
    s.iter().product()
}

pub const B: usize = 0;
pub const F: usize = 1;
pub const H: usize = 2;
pub const W: usize = 3;

#[derive(Default, Debug, PartialEq, Clone, Copy)]
pub enum VKind {
    #[default]
    Zero,
    One,
    Input,
    Param,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TOp {
    Value(VKind),
    /// y = x^e
    Exp,
    /// y = ln(x)
    Log,
    /// y = -x
    Neg,
    /// y = max(0, x)
    Relu,
    /// y = max(0, signum(x))
    Gtz,
    /// y = 1 / x
    Recip,
    /// Repeat tensor in the dimension `dim` `len` times
    Repeat { dim: usize, len: usize },
    /// Max-reduce the dimension `dim` to 1
    MaxReduce { dim: usize },
    /// Sum-reduce the dimension `dim` to 1
    SumReduce { dim: usize },
    /// Concatenate tensors in `dim`, all other dims must be same
    Cat { dim: usize },
    /// Select sub-range of the input tensor in some dimension
    Slice {
        dim: usize,
        from: usize,
        to: usize
    },
    /// Element-wise addtion
    Sum,
    /// Element-wise multiplication
    Prod,
    /// Classic matmul y = a * b
    MatMul,
    /// Transposed matnul y = a * b'
    MatMulT,
    /// 2D convolution, 1st tensor is the image, 2nd is the filter
    Conv2D,
    SumGrad
}

impl Default for TOp {
    fn default() -> Self {
        TOp::Value(VKind::Zero)
    }
}


#[derive(Debug, Default)]
pub struct TData {
    nm: Option<String>,
    sh: Shape,

    sc: SmallVec<Tensor, 2>,
    op: TOp,

    // Whether this tensor needs gradient calculation
    want_grad: bool,
    // Whether this tensor is on the grad path
    is_back: bool,

    grad: Option<Tensor>,
    grad_for: Option<Tensor>,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tensor {
    id: usize,
}

impl Borrow<usize> for Tensor {
    fn borrow(&self) -> &usize {
        &self.id
    }
}
