mod cpu;
pub use cpu::CPU;


#[cfg(feature = "mps")]
mod mps;
#[cfg(feature = "mps")]
pub use mps::MPS;

use crate::graph::CGraph;
use crate::{Tensor, TOp};

pub trait Evaluator {
    fn step(&mut self);
    /// Set a tensor to certain value
    fn set_value(&mut self, g: &CGraph, t: Tensor, v: &[f32]);
    /// Get the tensor buffer
    fn get_value(&self, t: Tensor) -> &[f32];
    /// Force-evaluate a tensor (blocking).
    fn evaluate(&mut self, g: &CGraph, t: Tensor);
    /// Copy values one tensor buffer to another. They must have same dimensions
    fn copy(&mut self, g: &CGraph, from: Tensor, to: Tensor);
}
