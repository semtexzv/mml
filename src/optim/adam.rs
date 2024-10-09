use crate::eval::Evaluator;
use crate::graph::CGraph;
use crate::optim::Optimizer;
use crate::Tensor;
use crate::tmap::TensorMap;


#[derive(Clone, Debug)]
pub struct AdamParams {
    pub lr: f64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f64,
}

impl Default for AdamParams {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

pub struct AdamVars {
    // First moment vector from previous iter
    m0: Tensor,
    // First moment vector in this iter
    m1: Tensor,
    // First moment vector with correction
    m1_cor: Tensor,

    // Second moment vector
    v0: Tensor,
    v1: Tensor,
    v1_cor: Tensor,
    // Next value tensor
    next: Tensor,
}

pub struct Adam {
    param: AdamParams,
    epoch: usize,
    
    lrt: Tensor,
    eps: Tensor,

    b1: Tensor,
    b1_neg: Tensor,
    b1_cor: Tensor,

    b2: Tensor,
    b2_neg: Tensor,
    b2_cor: Tensor,

    params: TensorMap<AdamVars>

}

impl Adam {
    pub fn new(g: &mut CGraph, param: AdamParams) -> Self {
        let o = g.ones([1, 1, 1, 1]);

        let b1 = g.val(param.beta1, [1, 1, 1, 1]);
        let b2 = g.val(param.beta2, [1, 1, 1, 1]);

        let b1_cor = g.input([1, 1, 1, 1]);
        let b2_cor = g.input([1, 1, 1, 1]);

        let b1_neg = g.sub(o, b1);
        let b2_neg = g.sub(o, b2);

        Self {
            epoch: 0,
            params: TensorMap::new(),
            b1,
            b2,
            b1_neg,
            b2_neg,
            lrt: g.val(param.lr as f32, [1, 1, 1, 1]),
            eps: g.val(1e-7, [1, 1, 1, 1]),
            b1_cor,
            b2_cor,

            param,
        }
    }
    fn vars_for_param(&mut self, g: &mut CGraph, param: Tensor) -> &mut AdamVars {
        let grd = g[param].grad.unwrap();
        if !self.params.has(param) {
            let m0 = g.input(g[grd].sh);
            let v0 = g.input(g[grd].sh);

            let sqr = g.mul(grd, grd);

            let b1 = g.broadcast_to(g[m0].sh, self.b1);
            let b2 = g.broadcast_to(g[m0].sh, self.b2);

            let m_a = g.mul(b1, m0);
            let v_a = g.mul(b2, v0);

            let b1_neg = g.broadcast_to(g[m0].sh, self.b1_neg);
            let b2_neg = g.broadcast_to(g[m0].sh, self.b2_neg);

            let m_b = g.mul(b1_neg, grd);
            let v_b = g.mul(b2_neg, sqr);

            let m1 = g.add(m_a, m_b);
            let v1 = g.add(v_a, v_b);

            let b1_cor = g.broadcast_to(g[m1].sh, self.b1_cor);
            let b2_cor = g.broadcast_to(g[v1].sh, self.b2_cor);

            let m1_cor = g.div(m1, b1_cor);
            let v1_cor = g.div(v1, b2_cor);

            let eps = g.broadcast_to(g[v1_cor].sh, self.eps);
            let denom = g.add(v1_cor, eps);

            let denom = g.sqrt(denom);

            let diff = g.div(m1_cor, denom);

            let lrt = g.broadcast_to(g[diff].sh, self.lrt);

            let change = g.mul(lrt, diff);

            let next = g.sub(param, change);

            self.params.set(param, AdamVars {
                m0,
                m1,
                m1_cor,
                v0,
                v1,
                v1_cor,
                next,
            })
        }
        return &mut self.params[param]
    }
}

impl Optimizer for Adam {
    fn optimize<E: Evaluator>(&mut self, g: &mut CGraph, e: &mut E, parameters: &[Tensor]) {
        self.epoch += 1;

        e.write(g, self.lrt, &[self.param.lr as _]);
        e.write(g, self.b1_cor, &[1.0 - self.param.beta1.powi(self.epoch as _)]);
        e.write(g, self.b2_cor, &[1.0 - self.param.beta2.powi(self.epoch as _)]);

        for prm in parameters.iter().cloned() {
            let pvar = self.vars_for_param(g, prm);
            e.eval(g, pvar.next);
            e.copy(g, pvar.next, prm);

            e.copy(g, pvar.m1, pvar.m0);
            e.copy(g, pvar.v1, pvar.v0);
        }
    }
}
