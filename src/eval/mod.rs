mod cpu;

use std::borrow::Cow;
pub use cpu::CPU;

#[cfg(feature = "mps")]
mod mps;
#[cfg(feature = "mps")]
pub use mps::MPS;

use crate::graph::CGraph;
use crate::Tensor;

/// Core trait for performing actual computation.
///
/// Internally has to store tensor buffers, perform dependency tracking and invalidation,
/// and correctly schedule tensor evaluation.
pub trait Evaluator {
    /// Step the internal counter. When doing this, all the tensors are invalidated
    /// Typically you [Evalutator::step], then [Evalutator::write] the input/output tensors
    /// and [Evalutator::eval] tensors you're interested in
    fn step(&mut self);
    /// Force-evaluate a tensor (blocking).
    fn eval(&mut self, g: &CGraph, t: Tensor);
    /// Set a tensor to certain value. Marks tensor as valid for this step
    fn write(&mut self, g: &CGraph, t: Tensor, v: &[f32]);
    /// Get the tensor buffer.
    fn read(&self, g: &CGraph, t: Tensor) -> Cow<[f32]>;
    /// Copy values one tensor buffer to another. They must have same dimensions.
    fn copy(&mut self, g: &CGraph, from: Tensor, to: Tensor);
    /// Zero-initialize gradients
    fn zero_grad(&mut self, g: &CGraph, params: &[Tensor]);
}
