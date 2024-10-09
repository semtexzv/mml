mod gopt;

use crate::tmap::TensorMap;
use crate::{Shape, TData, TOp, Tensor, VKind, B, F, H, W, eval};
use smallvec::smallvec;
use std::cmp::max;
use std::collections::BTreeMap;
use std::mem::take;
use std::ops::{Deref, Index, IndexMut};
use std::sync::Arc;

/// All Backpropagation operations take form of a simple function
/// NOTE: This is not a closure, all the state is stored in the `out` tensor.
/// If you need the input arguments, pick them out from the sources of the output tensor
pub type BackOp = fn(g: &mut CGraph, out: Tensor, outgrad: Tensor);

/// Core struct for recording the computation graph. Contains all the necessary state
/// to recreate the computation (Does not contain buffers), See [eval::Evaluator] for details
/// About buffers and evaluation.
#[derive(Default, Debug)]
pub struct CGraph {
    is_back: bool,
    // Current scope
    scp: String,
    // Tensor data
    ten: Vec<TData>,
    // Backpropagation ops for tensors
    bck: TensorMap<BackOp>,
    // Tensor names
    nme: BTreeMap<Arc<str>, Tensor>,
}

pub struct Scoped<'a> {
    g: &'a mut CGraph,
    s: &'a str
}

impl Index<Tensor> for CGraph {
    type Output = TData;

    fn index(&self, index: Tensor) -> &Self::Output {
        &self.ten[index.id]
    }
}

impl Index<&str> for CGraph {
    type Output = Tensor;

    fn index(&self, index: &str) -> &Self::Output {
        &self.nme[index]
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
        self.nme[named]
    }

    pub fn scope<T>(&mut self, name: &str, fun: impl FnOnce(&mut Self) -> T) -> T {
        let plen = self.scp.len();
        self.scp.push_str(name);
        self.scp.push('.');

        let ret = fun(self);

        unsafe {
            self.scp.as_mut_vec().set_len(plen);
        }
        ret
    }

    pub fn shape(&mut self, t: Tensor) -> Shape {
        self.ten[t.id].sh
    }

    pub fn input(&mut self, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            op: TOp::Value(VKind::Input),
            ..Default::default()
        });
        Tensor {
            id: self.ten.len() - 1,
        }
    }
    pub fn input_named(&mut self, name: &str, sh: Shape) -> Tensor {
        let t = self.input(sh);
        self.named(name, t)
    }

    pub fn zeros(&mut self, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            op: TOp::Value(VKind::Zero),
            ..Default::default()
        });
        Tensor {
            id: self.ten.len() - 1,
        }
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
        Tensor {
            id: self.ten.len() - 1,
        }
    }
    pub fn ones_like(&mut self, t: Tensor) -> Tensor {
        let sh = self.ten[t.id].sh;
        self.ones(sh)
    }
    pub fn val(&mut self, v: impl Into<f64>, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            op: TOp::Value(VKind::Val(v.into())),
            ..Default::default()
        });
        Tensor {
            id: self.ten.len() - 1,
        }
    }
    pub fn val_like(&mut self, v: f64, t: Tensor) -> Tensor {
        let sh = self.ten[t.id].sh;
        self.val(v, sh)
    }

    pub fn param(&mut self, name: &str, sh: Shape) -> Tensor {
        self.ten.push(TData {
            sh,
            want_grad: true,
            op: TOp::Value(VKind::Param),
            ..Default::default()
        });
        self.named(
            name,
            Tensor {
                id: self.ten.len() - 1,
            },
        )
    }

    pub fn named(&mut self, name: &str, t: Tensor) -> Tensor {
        let plen = self.scp.as_bytes().len();
        self.scp.push_str(name);
        let nm: Arc<str> = Arc::from(self.scp.as_str());
        unsafe { self.scp.as_mut_vec().set_len(plen) };

        if self.nme.contains_key(nm.deref()) {
            panic!("This graph already contains tensor named: \"{nm}\"");
        }

        self.ten[t.id].nm = Some(nm.clone());
        self.nme.insert(nm, t);
        t
    }

    pub fn grad_for(&mut self, ten: Tensor) -> Tensor {
        if let Some(g) = self[ten].grad {
            return g;
        }
        let grd = self.zeros_like(ten);
        self[ten].grad = Some(grd);
        self[grd].is_back = true;
        self[grd].op = TOp::Sum;
        self[grd].grad_for = Some(ten);

        grd
    }
    pub fn add_grad(&mut self, t: Tensor, gdiff: Tensor) {
        if self[t].want_grad {
            let mut grad = self.grad_for(t);
            if let Some(nm) = &self[t].nm {
                if self[grad].nm.is_none() {
                    grad = self.named(&format!("{}._grad", nm), grad)
                }
            }

            assert!(self[grad].is_back);
            assert_eq!(self[grad].op, TOp::Sum);

            self[grad].src.push(gdiff);
            self[gdiff].dst.push(grad);
        }
    }

    pub fn backward(&mut self, loss: Tensor) -> Vec<Tensor> {
        self.is_back = true;

        if self[loss].grad.is_some() {
            panic!("Tensor {loss:?} already has backwards pass registered");
        }

        let bck = take(&mut self.bck);
        let mut vis = TensorMap::default();
        let mut idx = Vec::new();

        fn topo(g: &mut CGraph, t: Tensor, vis: &mut TensorMap<()>, idx: &mut Vec<Tensor>) {
            vis.set(t, ());
            for s in g[t].src.clone() {
                if !vis.has(s) {
                    topo(g, s, vis, idx);
                }
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
            src: smallvec![t],
            op,
            want_grad: t1d.want_grad,
            is_back: t1d.is_back,
            ..Default::default()
        });

        let out = Tensor {
            id: self.ten.len() - 1,
        };
        self[t].dst.push(out);
        out
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
            src: smallvec![t1, t2],
            want_grad: t1d.want_grad | t2d.want_grad,
            op,
            is_back: t1d.is_back | t2d.is_back,
            ..Default::default()
        });

        let out = Tensor {
            id: self.ten.len() - 1,
        };

        self[t1].dst.push(out);
        self[t2].dst.push(out);

        out
    }

    pub fn broadcast(&mut self, dim: usize, len: usize, mut t: Tensor) -> Tensor {
        t = self._unop(
            TOp::Repeat {
                dim,
                len,
            },
            t,
        );
        self[t].sh[dim] = len;
        self.register_backwards_op(t, |g, out, outgrad| {
            let TOp::Repeat { dim, .. } = g[out].op else {
                panic!("tOP changed");
            };

            let src = g[out].src[0];
            let grad = g.sum_reduce(dim, outgrad);

            g.add_grad(src, grad);
        });
        return t
    }
    pub fn broadcast_to(&mut self, shape: Shape, mut t: Tensor) -> Tensor {
        for dim in 0..4 {
            if self[t].sh[dim] == 1 && shape[dim] > 1 {
                t = self.broadcast(dim, shape[dim], t);
            } else if self[t].sh[dim] > 1 && shape[dim] != self[t].sh[dim] {
                panic!(
                    "Cant broadcast {} to {} (dim={})",
                    self[t].sh[dim], shape[dim], dim
                );
            }
        }
        return t;
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

            let src = g[out].src[0];
            let grad = g.sum_reduce(dim, outgrad);

            g.add_grad(src, grad);
        });
        [t1, t2]
    }
    pub fn transpose(&mut self, d1: usize, d2: usize, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Transpose { d1, d2 }, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].src[0];
            let TOp::Transpose { d1, d2 } = g[out].op else {
                panic!("Invalid op")
            };

            let grad = g.transpose(d1, d2, outgrad);
            g.add_grad(src, grad)
        });
        let sh = &mut self[out].sh;
        sh.swap(d1, d2);
        out
    }

    pub fn neg(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Neg, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].src[0];
            let grad = g.neg(outgrad);

            g.add_grad(src, grad);
        });
        out
    }
    pub fn exp(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Exp, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].src[0];
            let grad = g.mul(outgrad, out);
            g.add_grad(src, grad);
        });
        out
    }
    pub fn log(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Log, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].src[0];
            let grad = g.div(outgrad, src);
            g.add_grad(src, grad);
        });
        out
    }
    pub fn sqrt(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Sqrt, t);
        self.register_backwards_op(out, |g, out, outgrad| {
            todo!("Implement sqrt")
        });
        out
    }

    pub fn relu(&mut self, t: Tensor) -> Tensor {
        let out = self._unop(TOp::Relu, t);

        self.register_backwards_op(out, |g, out, outgrad| {
            let src = g[out].src[0];
            let mask = g.gtz(src);
            let grad = g.mul(mask, outgrad);

            g.add_grad(src, grad);
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
            let src = g[out].src[0];
            let sqr = g.mul(src, src);
            let grad = g.div(outgrad, sqr);
            g.add_grad(src, grad)
        });
        out
    }

    pub fn sum_reduce(&mut self, dim: usize, t: Tensor) -> Tensor {
        let out = self._unop(TOp::SumReduce { dim }, t);
        self[out].sh[dim] = 1;

        self.register_backwards_op(out, |g, out, outgrad| {
            let grad = g.broadcast_to(g[g[out].src[0]].sh, outgrad);
            g.add_grad(g[out].src[0], grad);
        });

        out
    }

    pub fn sum_reduce_all(&mut self, mut t: Tensor) -> Tensor {
        for d in 0..4 {
            if self[t].sh[d] != 1 {
                t = self.sum_reduce(d, t);
            }
        }
        t
    }

    pub fn mean_reduce(&mut self, dim: usize, t: Tensor) -> Tensor {
        let len = self[t].sh[dim];
        let t = self.sum_reduce(dim, t);
        let scl = self.val_like(len as _, t);
        self.div(t, scl)
    }

    pub fn mean_reduce_all(&mut self, mut t: Tensor) -> Tensor {
        for d in 0..4 {
            if self[t].sh[d] != 1 {
                t = self.mean_reduce(d, t);
            }
        }
        t
    }

    pub fn max_reduce(&mut self, dim: usize, t: Tensor) -> Tensor {
        let out = self._unop(TOp::MaxReduce { dim }, t);
        self[out].sh[dim] = 1;

        self.register_backwards_op(out, |g, mut out, mut outgrad| {
            let src = g[out].src[0];

            out = g.broadcast_to(g[src].sh, out);
            outgrad = g.broadcast_to(g[src].sh, outgrad);

            let eqvl = g.eq(src, out);
            let grad = g.mul(eqvl, outgrad);
            g.add_grad(src, grad);

        });

        out
    }

    pub fn max_reduce_all(&mut self, mut t: Tensor) -> Tensor {
        for d in 0..4 {
            if self[t].sh[d] != 1 {
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
            for arg in g[out].src.clone() {
                g.add_grad(arg, outgrad);
            }
        });
        out
    }

    pub fn sub(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let t2 = self.neg(t2);
        self.add(t1, t2)
    }

    pub fn eq(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let out = self._binop(TOp::Eq, a, b);
        self.register_backwards_op(out, |g, out, outgrad| {
            panic!("No backwards op for eq");
        });
        out
    }

    pub fn mul(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let [t1, t2] = self.maybe_broadcast(B, t1, t2);
        let out = self._binop(TOp::Prod, t1, t2);

        assert_eq!(self[t1].sh, self[t2].sh, "Shape mismatch");

        self.register_backwards_op(out, |g, out, outgrad| {
            let srcs = g[out].src.clone();
            assert_eq!(srcs.len(), 2);

            let g0 = g.mul(outgrad, srcs[1]);
            g.add_grad(srcs[0], g0);

            let g1 = g.mul(outgrad, srcs[0]);
            g.add_grad(srcs[1], g1);
        });
        out
    }
    pub fn pow(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let [t1, t2] = self.maybe_broadcast(B, t1, t2);
        let out = self._binop(TOp::Pow, t1, t2);
        self.register_backwards_op(out, |g, out, outgrad| {
            todo!("No pow")
        });
        out
    }

    pub fn div(&mut self, t1: Tensor, t2: Tensor) -> Tensor {
        let t2 = self.recip(t2);
        self.mul(t1, t2)
    }

    pub fn mul_mat(&mut self, a: Tensor, ta: bool, b: Tensor, tb: bool) -> Tensor {
        let sha = self[a].sh;
        let shb = self[b].sh;

        assert!(
            sha[B] == 1 || shb[B] == 1 || sha[B] == shb[B],
            "Expected compatible batching {} {}",
            sha[B],
            shb[B]
        );
        assert!(
            sha[F] == 1 || shb[F] == 1,
            "Expected no features {} {}",
            sha[F],
            shb[F]
        );

        let out_h = if !ta { sha[H] } else { sha[W] };
        let inp_a = if !ta { sha[W] } else { sha[H] };

        let out_w = if !tb { shb[W] } else { shb[H] };
        let inp_b = if !tb { shb[H] } else { shb[W] };

        assert_eq!(
            inp_a, inp_b,
            "mul_mat: tensor dimensions mismatch {:?} {:?}",
            sha, shb
        );

        let sho = [max(sha[B], shb[B]), max(sha[F], shb[F]), out_h, out_w];

        let out = self._binop_sh(TOp::MatMul{ ta, tb }, sho, a, b);

        self.register_backwards_op(out, |g, out, outgrad| {
            let srcs = g[out].src.clone();
            let TOp::MatMul { ta, tb } = g[out].op else {
                panic!("Invalid op")
            };

            assert_eq!(srcs.len(), 2);

            let s0 = srcs[0];
            let s1 = srcs[1];

            if g[s0].want_grad {
                let g0 = g.mul_mat(outgrad, false, s1, !tb);
                g.add_grad(s0, g0);
            }
            if g[s1].want_grad {
                let g1 = g.mul_mat(s0, !ta, outgrad, false);
                g.add_grad(s1, g1);
            }
        });
        out
    }

    pub fn log_softmax(&mut self, dim: usize, x: Tensor) -> Tensor {
        let max = self.max_reduce(dim, x);
        let max = self.broadcast_to(self[x].sh, max);

        let elm = self.sub(x, max);

        let exp = self.exp(elm);
        let sum = self.sum_reduce(dim, exp);
        let sum = self.log(sum);

        let sum = self.broadcast_to(self[x].sh, sum);

        let a = self.sub(x, sum);
        self.sub(a, max)
    }

    pub fn softmax(&mut self, dim: usize, inp: Tensor) -> Tensor {
        let max = self.max_reduce(dim, inp);
        let max = self.broadcast_to(self[inp].sh, max);

        let inp = self.sub(inp, max);

        let exp = self.exp(inp);
        let sum = self.sum_reduce(dim, exp);
        let sum = self.broadcast_to(self[inp].sh, sum);
        let eps = self.val_like(1e-8, sum);
        let sum = self.add(sum, eps);

        self.div(exp, sum)
    }


    pub fn nll_loss(&mut self, inp: Tensor, tgt: Tensor) -> Tensor {
        let mul = self.mul(inp, tgt);
        let sum = self.sum_reduce_all(mul);
        self.neg(sum)
    }

    /// inp -> probabilities
    /// y = -sum(tgt * log(inp))
    /// inp = probabilities
    pub fn cross_entropy_loss(&mut self, inp: Tensor, tgt: Tensor) -> Tensor {
        let log = self.log(inp);

        self.nll_loss(log, tgt)
    }

    pub fn dump_named<E: eval::Evaluator>(&mut self, e: &mut E) {
        for (n, t) in &self.nme {
            println!("{:>20}: {:?}", &n[n.len().saturating_sub(20)..], e.read(self, *t))
        }
    }

    pub fn linear(
        &mut self,
        name: &str,
        inf: usize,
        outf: usize,
    ) -> impl Fn(&mut Self, Tensor) -> Tensor {
        self.scope(name, |g| {
            let w = g.param("weight", [1, 1, outf, inf]);
            let b = g.param("bias", [1, 1, 1, outf]);

            let name = name.to_string();
            move |g: &mut Self, x: Tensor| {
                g.scope(&name, |g| {
                    let t = g.mul_mat(x, false, w, true);
                    g.add(t, b)
                })
            }
        })
    }
}
