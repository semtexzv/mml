use crate::eval::Evaluator;
use crate::graph::CGraph;
use crate::Tensor;

trait Sample {
    fn apply<E: Evaluator>(&self, g: &mut CGraph, e: &mut E, inp: Tensor, out: Tensor);
}

trait DataSource {
    type Sample: Sample;
    fn len(&self) -> usize;
    fn sample(&mut self, idx: usize) -> Self::Sample;
}
