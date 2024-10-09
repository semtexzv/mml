mod data;
pub mod eval;
pub mod graph;
pub mod optim;
pub(crate) mod tmap;
pub use tmap::Tensor;
use smallvec::SmallVec;
use std::sync::Arc;


/// Number of dimensions supported
pub const DIMS: usize = 4;
/// Batch dimension
pub const B: usize = 0;
/// Feature dimension
pub const F: usize = 1;
/// Height dimension
pub const H: usize = 2;
/// Width dimension
pub const W: usize = 3;


/// All Tensors in MML are 4D. If you need more, pack information into the feature dimension
/// Batch is understood by the framework for minibatches.
pub type Shape = [usize; DIMS];

/// Multiply all the shape dimensions, return total size of the tensor in elements
pub fn prod(s: Shape) -> usize {
    s.iter().product()
}

/// Calculate strides to access the array
///
/// NOTE: dimensions of size 1 have stride of 0 to allow for broadcasted access
pub fn strd(s: Shape) -> [usize; DIMS] {
    let tmp =  [
        sprod(&s[1..]),
        sprod(&s[2..]),
        sprod(&s[3..]),
        1
    ];
    // for i in (0..=2).rev() {
    //     tmp[i] *= tmp[i+1];
    // }
    [
        if s[B] > 1 { tmp[B] } else { 0 },
        if s[F] > 1 { tmp[F] } else { 0 },
        if s[H] > 1 { tmp[H] } else { 0 },
        if s[W] > 1 { tmp[W] } else { 0 },
    ]

}
#[test]
fn test_strides() {
    assert_eq!(strd([1, 1, 1, 1]), [0, 0, 0, 0]);
    assert_eq!(strd([1, 1, 1, 2]), [0, 0, 0, 1]);
    assert_eq!(strd([1, 1, 1, 8]), [0, 0, 0, 1]);
    assert_eq!(strd([1, 1, 2, 1]), [0, 0, 1, 0]);
    assert_eq!(strd([1, 2, 2, 1]), [0, 2, 1, 0]);
    assert_eq!(strd([2, 3, 2, 1]), [6, 2, 1, 0]);
}

pub fn sprod(s: &[usize]) -> usize {
    s.iter().product()
}

/// ValueKind. Different value kinds have different initialization behavior
#[derive(Default, Debug, PartialEq, Clone, Copy)]
pub enum VKind {
    #[default]
    Zero,
    One,
    Val(f64),
    Input,
    Param,
}

/// Tensor operation
///
/// Should be [Copy].
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TOp {
    Value(VKind),
    /// y = -x
    Neg,
    /// y = x^e
    Exp,
    /// y = ln(x)
    Log,
    /// y = sqrt(x)
    Sqrt,
    /// y = max(0, x)
    Relu,
    /// y = max(0, signum(x))
    Gtz,
    /// y = 1 / x
    Recip,

    /// Equality comparison
    Eq,
    /// Element-wise addtion
    Sum,
    /// Element-wise multiplication
    Prod,
    Div,
    /// Power
    Pow,

    /// Classic matmul y = a * b
    MatMul {
        ta: bool,
        tb: bool,
    },

    /// Transposed matmul y = a * b'
    MatMulT,

    /// 2D convolution, 1st tensor is the image, 2nd is the filter
    Conv2D,


    /// Repeat tensor in the dimension `dim` `len` times
    Repeat {
        dim: usize,
        len: usize,
    },

    /// Max-reduce the dimension `dim` to 1
    MaxReduce {
        dim: usize,
    },

    /// Sum-reduce the dimension `dim` to 1
    SumReduce {
        dim: usize,
    },
    /// Concatenate tensors in `dim`, all other dims must be same
    Cat {
        dim: usize,
    },

    /// Select sub-range of the input tensor in some dimension
    Slice {
        dim: usize,
        from: usize,
        to: usize,
    },

    /// Swap two axes
    Transpose {
        d1: usize,
        d2: usize,
    },

    Permute {
        axes: [usize; DIMS]
    }
}

impl Default for TOp {
    fn default() -> Self {
        TOp::Value(VKind::Zero)
    }
}


#[derive(Debug, Default)]
pub struct TData {
    pub nm: Option<Arc<str>>,
    pub sh: Shape,

    // What are sources of this tensor
    pub src: SmallVec<Tensor, 2>,
    // Where data flows from this tensor
    pub dst: SmallVec<Tensor, 2>,
    pub op: TOp,

    // Whether this tensor needs gradient calculation
    want_grad: bool,
    // Whether this tensor is on the grad path
    is_back: bool,

    pub grad: Option<Tensor>,
    pub grad_for: Option<Tensor>,
}
