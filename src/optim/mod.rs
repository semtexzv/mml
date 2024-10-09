mod sgd;
mod adam;

pub use sgd::SGD;
pub use adam::{Adam, AdamParams};

use crate::eval::Evaluator;
use crate::graph::CGraph;
use crate::Tensor;

pub trait Optimizer {
    fn optimize<E: Evaluator>(&mut self, g: &mut CGraph, e: &mut E, parameters: &[Tensor]);
}
