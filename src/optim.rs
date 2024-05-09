use crate::eval::{CPU, Evaluator};
use crate::graph::CGraph;
use crate::{Tensor, TOp, VKind};
use crate::tmap::TensorMap;

pub trait Optimizer {
    fn optimize(&mut self, g: &mut CGraph, e: &mut CPU, parameters: &[Tensor]);
}

pub struct SGD {
    lr: f32,
    clip: bool,
    // Learning rate tensor
    lrt: Tensor,
    // Next value tensors
    next: TensorMap<Tensor>,
}

impl SGD {
    pub fn new(g: &mut CGraph, lr: f32) -> Self {
        Self {
            lr,
            clip: true,
            lrt: g.zeros([1, 1, 1, 1]),
            next: Default::default(),
        }
    }
}

impl Optimizer for SGD {
    fn optimize(&mut self, g: &mut CGraph, e: &mut CPU, parameters: &[Tensor]) {
        e.set_value(g, self.lrt, &[-self.lr]);

        for param in parameters {
            assert_eq!(g[*param].op, TOp::Value(VKind::Param));

            if !self.next.has(*param) {
                let grad = g[*param].grad.unwrap();
                let chng = g.mul(self.lrt, grad);
                let next = g.add(*param, chng);
                self.next.set(*param, next);
            }

            let param = *param;
            let next = self.next[param];

            e.evaluate(g, next);
            e.copy(g, next, param);
        }
    }
}
