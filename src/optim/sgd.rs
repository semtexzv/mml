use crate::eval::{Evaluator, CPU};
use crate::graph::CGraph;
use crate::optim::Optimizer;
use crate::tmap::TensorMap;
use crate::{TOp, Tensor, VKind, B};

pub struct SGD {
    lr: f32,
    clip: bool,
    // Learning rate tensor
    lrt: Tensor,
    bst: Tensor,
    // Next value tensors
    next: TensorMap<Tensor>,
}

impl SGD {
    pub fn new(g: &mut CGraph, lr: f32) -> Self {
        g.scope("_optim", |g| {

            let lrt = g.zeros([1, 1, 1, 1]);
            let bst = g.ones([1, 1, 1, 1]);

            Self {
                lr,
                lrt: g.named("lrt", lrt),
                bst: g.named("bst", bst),
                clip: true,
                next: Default::default(),
            }
        })
    }
}

impl Optimizer for SGD {
    fn optimize<E: Evaluator>(&mut self, g: &mut CGraph, e: &mut E, parameters: &[Tensor]) {
        e.write(g, self.lrt, &[-self.lr]);
        if parameters.len() > 0 {
            e.write(g, self.bst, &[g[parameters[0]].sh[B] as f32]);
        }

        g.scope("_optim", |g| {
            for param in parameters {
                assert_eq!(g[*param].op, TOp::Value(VKind::Param));

                if !self.next.has(*param) {
                    let nm = g[*param].nm.as_ref().cloned().unwrap().to_string();

                    let grad = g[*param].grad.unwrap();

                    let lrt = g.broadcast_to(g[grad].sh, self.lrt);
                    let bst = g.broadcast_to(g[grad].sh, self.bst);

                    let chng = g.mul(lrt, grad);
                    let chng = g.named(&format!("{}.chng", nm), chng);
                    let chng = g.div(chng, bst);
                    let next = g.add(*param, chng);
                    let next = g.named(&format!("{}.next", nm), next);

                    /// P[i+1] = P[i] + (-lr * G[i]) / BATCH_SIZE
                    self.next.set(*param, next);
                }

                let param = *param;
                let next = self.next[param];

                e.eval(g, next);
                e.copy(g, next, param);
            }
        })
    }
}
