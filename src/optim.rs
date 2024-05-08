use crate::eval::CPU;
use crate::graph::CGraph;
use crate::{Tensor, TOp};
use crate::tmap::TensorMap;

pub trait Optimizer {
    fn optimize(&mut self, g: &mut CGraph, e: &mut CPU, parameters: &[Tensor]);
}

pub struct SGD {
    lr: f32,
    clip: bool,
    // Tensor diff tensors
    tdiff: TensorMap<Tensor>,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            clip: true,
            tdiff: Default::default(),
        }
    }
}

impl Optimizer for SGD {
    fn optimize(&mut self, g: &mut CGraph, e: &mut CPU, parameters: &[Tensor]) {
        for param in parameters {
            if !self.tdiff.has(*param) {
                self.tdiff.set(*param, g.zeros_like(*param));
            }
            /// diff = (-lr * grad) / max_reduce_all(grad)
            let param = *param;
            let diff = self.tdiff[param];
            let grad = g[param].grad.unwrap();

            e.evaluate(g, grad);
            e.set_value(g, diff, &[-self.lr]);
            e.do_eval(g, diff, &[diff, grad], TOp::Mul);
            e.do_eval(g, param, &[param, diff], TOp::Add);
            e.set_value(g, grad, &[0.0]);
        }
    }
}
