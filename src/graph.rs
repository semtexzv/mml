use std::cmp::max;
use std::mem::take;
use std::ops::{Index, IndexMut};
use smallvec::smallvec;
use crate::{B, F, H, Shape, TData, Tensor, TOp, VKind, W};
use crate::tmap::TensorMap;

pub type BackOp = fn(g: &mut CGraph, out: Tensor, grad: Tensor);

#[derive(Default, Debug)]
pub struct CGraph {
    is_back: bool,
    // Current scope
    scp: String,
    // Tensor data
    ten: Vec<TData>,
    // Backpropagation ops for tensors
    bck: TensorMap<BackOp>,
}

impl Index<Tensor> for CGraph {
    type Output = TData;

    fn index(&self, index: Tensor) -> &Self::Output {
        &self.ten[index.id]
    }
}

impl IndexMut<Tensor> for CGraph {
    fn index_mut(&mut self, index: Tensor) -> &mut Self::Output {
        &mut self.ten[index.id]
    }
}

impl CGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn find(&self, named: &str) -> Tensor {
        Tensor {
            id: self.ten.iter().enumerate().filter(|(i, d)| d.nm.as_deref() == Some(named)).next().unwrap().0
        }
    }

    pub fn scope<T>(&mut self, name: &str, fun: impl FnOnce(&mut Self) -> T) -> T {
        let plen = self.scp.len();
        self.scp.push_str(name);
        self.scp.push('.');

        let ret = fun(self);

        unsafe { self.scp.as_mut_vec().set_len(plen); }
        ret
    }

    pub fn shape(&mut self, t: Tensor) -> Shape {
        self.ten[t.id].sh
    }

    pub fn input(&mut self, name: &str, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            op: TOp::Value(VKind::Input),
            ..Default::default()
        });

        self.named(name, Tensor { id: self.ten.len() - 1 })
    }

    pub fn zeros(&mut self, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            op: TOp::Value(VKind::Zero),
            ..Default::default()
        });
        Tensor { id: self.ten.len() - 1 }
    }
    pub fn zeros_like(&mut self, t: Tensor) -> Tensor {
        let sh = self.ten[t.id].sh;
        self.zeros(sh)
    }

    pub fn ones(&mut self, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            op: TOp::Value(VKind::One),
            ..Default::default()
        });
        Tensor { id: self.ten.len() - 1 }
    }
    pub fn ones_like(&mut self, t: Tensor) -> Tensor {
        let sh = self.ten[t.id].sh;
        self.ones(sh)
    }

    pub fn param(&mut self, name: &str, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            want_grad: true,
            op: TOp::Value(VKind::Param),
            ..Default::default()
        });
        self.named(name, Tensor { id: self.ten.len() - 1 })
    }

    pub fn named(&mut self, name: &str, t: Tensor) -> Tensor {
        let mut nm: String = self.scp.clone();
        nm.push_str(name);
        self.ten[t.id].nm = Some(nm);
        t
    }

    #[inline(never)]
    pub fn grad_for(&mut self, ten: Tensor) -> Tensor {
        if let Some(g) = self[ten].grad {
            return g;
        }
        // TODO: We don't need the batch dim on gradients
        let grd = self.zeros_like(ten);
        self[ten].grad = Some(grd);

        self[grd].is_back = true;
        self[grd].op = TOp::Sum;
        self[grd].grad_for = Some(ten);

        grd
    }
    pub fn add_grad(&mut self, t: Tensor, gdiff: Tensor) {
        let grad = self.grad_for(t);
        assert!(self[grad].is_back);
        assert_eq!(self[grad].op, TOp::Sum);

        self[grad].sc.push(gdiff);
    }

    pub fn backward(&mut self, loss: Tensor) -> Vec<Tensor> {
        self.is_back = true;

        let bck = take(&mut self.bck);
        let mut vis = TensorMap::default();
        let mut idx = Vec::new();

        fn topo(g: &mut CGraph, t: Tensor, vis: &mut TensorMap<()>, idx: &mut Vec<Tensor>) {
            vis.set(t, ());
            for s in g[t].sc.clone() {
                if vis.has(s) {
                    continue;
                }
                topo(g, s, vis, idx);
            }
            idx.push(t);
        }

        topo(self, loss, &mut vis, &mut idx);
        idx.reverse();

        let out_grad = self.ones_like(loss);
        self[loss].grad = Some(out_grad);
        self[out_grad].grad_for = Some(loss);
        self[out_grad].is_back = true;

        let mut out = vec![];
        for t in idx {
            if self[t].op == TOp::Value(VKind::Param) {
                out.push(t);
            }

            if let Some(bck) = bck.get(t) {
                let grad = self.grad_for(t);
                bck(self, t, grad);
            }
        }

        self.bck = bck;
        self.is_back = false;
        out
    }

    pub(crate) fn register_backwards_op(&mut self, t: Tensor, op: BackOp) {
        if !self.is_back && self[t].want_grad {
            self.bck.set(t, op);
        }
    }

    pub(crate) fn _unop(&mut self, op: TOp, t: Tensor) -> Tensor {
        let t1d = &self.ten[t.id];

        self.ten.push(TData {
            nm: None,
            sh: t1d.sh,
            sc: smallvec![t],
            op,
            want_grad: t1d.want_grad,
            is_back: t1d.is_back,
            ..Default::default()
        });

        Tensor { id: self.ten.len() - 1 }
    }

    pub(crate) fn _binop(&mut self, op: TOp, t1: Tensor, t2: Tensor) -> Tensor {
        self._binop_sh(op, self.ten[t1.id].sh, t1, t2)
    }
    pub(crate) fn _binop_sh(&mut self, op: TOp, sh: Shape, t1: Tensor, t2: Tensor) -> Tensor {
        let t1d = &self.ten[t1.id];
        let t2d = &self.ten[t2.id];

        self.ten.push(TData {
            nm: None,
            sh,
            sc: smallvec![t1, t2],
            want_grad: t1d.want_grad | t2d.want_grad,
            op,
            is_back: t1d.is_back | t2d.is_back,
            ..Default::default()
        });

        Tensor { id: self.ten.len() - 1 }
    }
    pub fn maybe_broadcast(&mut self, dim: usize, mut t1: Tensor, mut t2: Tensor) -> [Tensor; 2] {
        let d1 = self[t1].sh[dim];
        let d2 = self[t2].sh[dim];

        if d1 == d2 {
            return [t1, t2];
        }

        let modified = if d1 == 1 {
            t1 = self._unop(TOp::Repeat { dim, len: d2 }, t1);
            self[t1].sh[dim] = d2;
            t1
        } else if d2 == 1 {
            t2 = self._unop(TOp::Repeat { dim, len: d2 }, t2);
            self[t2].sh[dim] = d1;
            t2
        } else {
            panic!()
        };

        self.register_backwards_op(modified, |g, out, outgrad| {
            let TOp::Repeat { dim, .. } = g[out].op else {
                panic!("tOP changed");
            };

            let src = g[out].sc[0];
            let grad = g.sum_reduce(dim, outgrad);

            g.add_grad(src, grad);
        });
        [t1, t2]
    }
    pub fn neg(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Neg, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].sc[0];
            let grad = g.neg(outgrad);

            g.add_grad(src, grad);
        });
        out
    }
    pub fn exp(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Exp, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].sc[0];
            let grad = g.exp(outgrad);
            g.add_grad(src, grad);
        });
        out
    }
    pub fn log(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Log, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].sc[0];
            let grad = g.div(outgrad, src);
            g.add_grad(src, grad);
        });
        out
    }
    pub fn relu(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Relu, t);

        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].sc[0];
            let mask = g.gtz(src);
            let grad = g.mul(mask, outgrad);

            g.add_grad(out, grad);
        });

        out
    }
    
    pub fn gtz(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Gtz, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            panic!("Unimplemented GTZ backwards");
        });
        out
    }
    pub fn recip(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Recip, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            panic!("Unimplemented Recip backwards");
        });
        out
    }
    pub fn sum_reduce(&mut self, dim: usize, t: Tensor) -> Tensor {
        let out = self._unop(TOp::SumReduce { dim }, t);
        self[out].sh[dim] = 1;

        self.register_backwards_op(out, |g, out, outgrad| {
            panic!("Unimplemented SumReduce backwards");
        });

        out
    }

    pub fn sum_reduce_all(&mut self, mut t: Tensor) -> Tensor {
        for d in self[t].sh {
            if d != 1 {
                t = self.sum_reduce(d, t);
            }
        }
        t
    }

    pub fn max_reduce(&mut self, dim: usize, t: Tensor) -> Tensor {
        let out = self._unop(TOp::MaxReduce { dim }, t);
        self[out].sh[dim] = 1;

        self.register_backwards_op(out, |g, out, outgrad| {
            panic!("Unimplemented MaxReduce backwards");
        });

        out
    }

    pub fn max_reduce_all(&mut self, mut t: Tensor) -> Tensor {
        for d in self[t].sh {
            if d != 1 {
                t = self.max_reduce(d, t);
            }
        }
        t
    }

    pub fn min_reduce(&mut self, dim: usize, mut t: Tensor) -> Tensor {
        t = self.neg(t);
        t = self.max_reduce(dim, t);
        self.neg(t)
    }

    pub fn min_reduce_all(&mut self, mut t: Tensor) -> Tensor {
        t = self.neg(t);
        t = self.max_reduce_all(t);
        self.neg(t)
    }

    pub fn add(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let [t1, t2] = self.maybe_broadcast(B, t1, t2);
        let out = self._binop(TOp::Sum, t1, t2);

        self.register_backwards_op(out, |g, out, outgrad| {
            for arg in g[out].sc.clone() {
                g.add_grad(arg, outgrad);
            }
        });
        out
    }

    pub fn sub(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let t2 = self.neg(t2);
        self.add(t1, t2)
    }

    pub fn mul(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let [t1, t2] = self.maybe_broadcast(B, t1, t2);
        let out = self._binop(TOp::Prod, t1, t2);

        self.register_backwards_op(out, |g, out, outgrad| {
            let srcs = g[out].sc.clone();
            assert_eq!(srcs.len(), 2);
            let g0 = g.mul(outgrad, srcs[1]);
            g.add_grad(srcs[0], g0);

            let g1 = g.mul(outgrad, srcs[0]);
            g.add_grad(srcs[1], g1);
        });
        out
    }

    pub fn mul_scl(&mut self, t1: Tensor, scl: f32) -> Tensor {
        let t2 = self.ones_like(t1);
        self.mul(t1, t2)
    }
    pub fn div(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let t2 = self.recip(t2);
        self.mul(t1, t2)
    }

    pub fn mul_mat(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let sh1 = self[t1].sh;
        let sh2 = self[t2].sh;

        assert!(sh1[B] == 1 || sh2[B] == 1);
        assert!(sh1[F] == 1 || sh2[F] == 1);
        assert_eq!(sh1[W], sh2[H], "mul_mat: tensor dimensions mismatch");

        let sho = [
            max(sh1[B], sh2[B]),
            max(sh1[F], sh2[F]),
            sh1[H],
            sh2[W]
        ];
        self._binop_sh(TOp::MatMul, sho, t1, t2)
    }
    pub fn mul_mat_t(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let sh1 = self[t1].sh;
        let sh2 = self[t2].sh;

        assert!(sh1[B] == 1 || sh2[B] == 1);
        assert!(sh1[F] == 1 || sh2[F] == 1);
        assert_eq!(sh1[W], sh2[W], "mul_mat_t {:?} with {:?}", sh1, sh2);

        let sho = [
            max(sh1[B], sh2[B]),
            max(sh1[F], sh2[F]),
            sh1[H],
            sh2[H]
        ];
        self._binop_sh(TOp::MatMulT, sho, t1, t2)
    }

    pub fn linear(&mut self, name: &str, inf: usize, outf: usize) -> impl Fn(&mut Self, Tensor) -> Tensor {
        self.scope(name, |g| {
            let w = g.param("weight", [1, 1, outf, inf]);
            let b = g.param("bias", [1, 1, 1, outf]);

            move |g: &mut Self, x: Tensor| {
                let t = g.mul_mat_t(x, w);
                g.add(t, b)
            }
        })
    }
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        // let mut grd = HashMap::new();
        dot.push_str("digraph G {\n");
        dot.push_str("  node [shape=record];\n");

        for (i, tdata) in self.ten.iter().enumerate() {
            if tdata.grad_for.is_some() {
                continue;
            }
            let name = tdata.nm.clone().unwrap_or_default() + &format!("_{:?}", tdata.op);
            let sub = tdata.grad.map(|g| {
                let gdata = &self[g];
                gdata.nm.clone().unwrap_or_default() + &format!("_{:?}", gdata.op)
            }).unwrap_or_default();

            // Is part of backwards pass
            if tdata.is_back {
                dot.push_str(&format!("  {i} [label=\"<t> {name} | {:?} \" color=red];\n", tdata.sh));
            } else {
                dot.push_str(&format!("  {i} [label=\"{{ <t> {name} | <g> {sub} }}\" color=blue];\n"));
            }
        }

        for (i, tdata) in self.ten.iter().enumerate() {
            if let Some(target) = tdata.grad_for {
                for src in tdata.sc.clone() {
                    dot.push_str(&format!("  {}:<t> -> {}:<g> [color=red];\n", src.id, target.id));
                }
            } else {
                for src in tdata.sc.clone() {
                    dot.push_str(&format!("  {}:<t> -> {}:<t>;\n", src.id, i));
                }
            }
        }

        dot.push_str("}\n");
        dot
    }
}