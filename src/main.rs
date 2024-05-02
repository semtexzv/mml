use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result};
use std::mem::take;
use std::ops::FnOnce;
use std::option::Option;
use std::option::Option::{None, Some};
use std::string::String;
use std::vec::Vec;

// #[repr(transparent)]
// #[derive(Debug, Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
type Shape = [usize; 4];

#[derive(Debug)]
enum TOp {
    Const,

    Exp,
    Log,
    Neg,

    Add,
    Mul,
    MatMul,
    Conv,
}

#[derive(Debug)]
struct TData {
    nm: Option<String>,
    sh: Shape,

    sc: Vec<Tensor>,

    op: TOp,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct Tensor {
    id: usize,
}

type BackOp = fn(g: &GraphBuilder, out: Tensor, grad: Tensor);

#[derive(Default)]
struct GraphBuilder {
    is_back: UnsafeCell<bool>,
    // Current scope
    scp: UnsafeCell<String>,
    // Tensor storage
    ten: UnsafeCell<Vec<TData>>,
    // Gradient mapping
    grd: UnsafeCell<HashMap<Tensor, Tensor>>,
    // Backpropagation ops for tensors
    bck: UnsafeCell<HashMap<Tensor, BackOp>>,
}

impl Debug for GraphBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("GraphBuilder")
            .field("ten", self.ten())
            .field("grd", self.grd())
            .field("bck", self.bck())
            .finish()
    }
}

impl GraphBuilder {
    fn scp(&self) -> &mut String {
        unsafe { &mut *self.scp.get() }
    }
    fn ten(&self) -> &mut Vec<TData> {
        unsafe { &mut *self.ten.get() }
    }
    fn grd(&self) -> &mut HashMap<Tensor, Tensor> {
        unsafe { &mut *self.grd.get() }
    }
    fn bck(&self) -> &mut HashMap<Tensor, BackOp> {
        unsafe { &mut *self.bck.get() }
    }
    fn is_back(&self) -> bool {
        unsafe { *self.is_back.get() }
    }

    fn tdata(&self, t: Tensor) -> &mut TData {
        &mut self.ten()[t.id]
    }

    fn scope<T>(&self, name: &str, fun: impl FnOnce(&Self) -> T) -> T {
        let plen = self.scp().len();
        self.scp().push_str(name);
        let ret = fun(self);
        unsafe { self.scp().as_mut_vec().set_len(plen); }
        ret
    }

    fn shape(&self, t: Tensor) -> Shape {
        self.ten()[t.id].sh
    }

    fn input(&self, sh: Shape) -> Tensor {
        let ten = self.ten();

        ten.push(TData { nm: None, sh, sc: vec![], op: TOp::Const });
        Tensor { id: ten.len() - 1 }
    }

    fn zeros(&self, sh: Shape) -> Tensor {
        let ten = self.ten();

        ten.push(TData { nm: None, sh, sc: vec![], op: TOp::Const });
        Tensor { id: ten.len() - 1 }
    }
    fn zeros_like(&self, t: Tensor) -> Tensor {
        let ten = self.ten();

        ten.push(TData { nm: None, sh: ten[t.id].sh, sc: vec![], op: TOp::Const });
        Tensor { id: ten.len() - 1 }
    }
    fn ones(&self, sh: Shape) -> Tensor {
        let ten = self.ten();

        ten.push(TData { nm: None, sh, sc: vec![], op: TOp::Const });
        Tensor { id: ten.len() - 1 }
    }

    fn param(&self, name: &str, sh: Shape) -> Tensor {
        let ten = self.ten();

        let mut nm: String = self.scp().clone();
        nm.push('.');
        nm.push_str(name);

        ten.push(TData { nm: Some(nm), sh, sc: vec![], op: TOp::Const });
        Tensor { id: ten.len() - 1 }
    }

    fn backward(&self, ten: Tensor) {
        unsafe { *self.is_back.get() = true; }
        let bck = take(self.bck());

        for (t, op) in bck {
            op(self, t, *self.grd().entry(t).or_insert_with(|| self.zeros_like(t)));
        }

        unsafe { *self.is_back.get() = false; }
    }
    fn back_for(&self, t: Tensor, op: BackOp) {
        if !self.is_back() {
            assert!(self.bck().insert(t, op).is_none());
        }
    }

    fn _unop(&self, op: TOp, t: Tensor) -> Tensor {
        let ten = self.ten();

        let t1d = &ten[t.id];

        ten.push(TData {
            nm: None,
            sh: t1d.sh,
            sc: vec![t],
            op,
        });

        Tensor { id: ten.len() - 1 }
    }

    fn _binop(&self, op: TOp, t1: Tensor, t2: Tensor) -> Tensor {
        let ten = self.ten();

        let t1d = &ten[t1.id];
        let t2d = &ten[t2.id];
        // assert_is_broadcastable(t1d.sh, t2d.sh);
        assert_eq!(t1d.sh, t2d.sh);

        ten.push(TData {
            nm: None,
            sh: t1d.sh,
            sc: vec![t1, t2],
            op,
        });

        Tensor { id: ten.len() - 1 }
    }
    fn neg(&self, t: Tensor) -> Tensor {
        self._unop(TOp::Neg, t)
    }
    fn exp(&self, t: Tensor) -> Tensor {
        self._unop(TOp::Exp, t)
    }
    fn log(&self, t: Tensor) -> Tensor {
        self._unop(TOp::Log, t)
    }

    fn add(&self, t1: Tensor, t2: Tensor) -> Tensor {
        let out = self._binop(TOp::Add, t1, t2);
        self.back_for(out, |g, out, grad| {
            for arg in &g.tdata(out).sc {
                let garg = g.grd().entry(*arg).or_insert_with(|| g.zeros_like(*arg));
                *garg = g.add(*garg, grad);
            }
        });
        out
    }

    fn mul(&self, t1: Tensor, t2: Tensor) -> Tensor {
        let out = self._binop(TOp::Mul, t1, t2);
        self.back_for(out, |g, out, grad| {
            let srcs = &g.tdata(out).sc;
            assert_eq!(srcs.len(), 2);

            let ag0 = g.grd().entry(srcs[0]).or_insert_with(|| g.zeros_like(srcs[0]));
            let ag1 = g.grd().entry(srcs[1]).or_insert_with(|| g.zeros_like(srcs[1]));

            *ag0 = g.add(*ag1, srcs[1]);
            *ag1 = g.add(*ag1, srcs[1]);

        });
        out
    }

    fn mul_mat(&self, t1: Tensor, t2: Tensor) -> Tensor {
        self._binop(TOp::MatMul, t1, t2)
    }
}

fn linear(g: &GraphBuilder, name: &str, x: Tensor) -> Tensor {
    g.scope(name, |g| {
        let w = g.param("weight", g.shape(x));
        let b = g.param("bias", g.shape(x));

        g.add(g.mul(x, w), b)
    })
}

fn main() {
    let g = &GraphBuilder::default();
    let a = g.zeros([1, 1, 1, 2]);
    let b = g.zeros([1, 1, 1, 2]);
    let c = linear(g, "lin", a);
    let y = g.add(c, b);

    println!("graph: {g:#?}");
}