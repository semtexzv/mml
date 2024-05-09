mod sgd;
pub use sgd::SGD;

use crate::eval::{CPU, Evaluator};
use crate::graph::CGraph;
use crate::Tensor;

pub trait Optimizer {
    fn optimize(&mut self, g: &mut CGraph, e: &mut CPU, parameters: &[Tensor]);
}
