#![feature(map_many_mut)]
#![feature(get_many_mut)]
#![feature(slice_ptr_get)]

mod graph;
mod tmap;
mod eval;
mod optim;

use std::borrow::Borrow;
use std::ops::{FnOnce, Index, IndexMut};
use std::string::String;
use rand::random;
use smallvec::SmallVec;
use crate::eval::CPU;
use crate::graph::CGraph;
use crate::optim::{Optimizer, SGD};

type Shape = [usize; 4];

fn prod(s: Shape) -> usize {
    s.iter().product()
}

fn prod2(s: &[usize]) -> usize {
    s.iter().product()
}

const B: usize = 0;
const F: usize = 1;
const H: usize = 2;
const W: usize = 3;

#[derive(Default, Debug, PartialEq, Clone, Copy)]
enum VKind {
    #[default]
    Zero,
    One,
    Input,
    Param,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum TOp {
    Value(VKind),

    /// y = x^e
    Exp,
    /// y = ln(x)
    Log,
    /// y = -x
    Neg,
    /// y = max(0, x)
    Relu,
    /// y = max(0, signum(x))
    Gtz,
    /// y = 1 / x
    Recip,

    Repeat { dim: usize, len: usize },
    MaxReduce { dim: usize },
    SumReduce { dim: usize },

    Add,
    Mul,

    MatMul,
    MatMulT,
    Conv,

    SumGrad,
}

impl Default for TOp {
    fn default() -> Self {
        TOp::Value(VKind::Zero)
    }
}


#[derive(Debug, Default)]
struct TData {
    nm: Option<String>,
    sh: Shape,

    sc: SmallVec<Tensor, 2>,
    op: TOp,

    // Whether this tensor needs gradient calculation
    want_grad: bool,
    // Whether this tensor is on the grad path
    is_back: bool,

    grad: Option<Tensor>,
    grad_for: Option<Tensor>,
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Tensor {
    id: usize,
}

impl Borrow<usize> for Tensor {
    fn borrow(&self) -> &usize {
        &self.id
    }
}

fn target(x: f32) -> f32 {
    32.0 * x + 10.0
}

fn main() {
    const BATCH: usize = 32;
    let g = &mut CGraph::default();
    let e = &mut CPU::new();
    let o = &mut SGD::new(0.001);


    let a = g.input("t", [1, 1, 2, 2]);
    let b = g.input("t", [1, 1, 2, 2]);
    let xx = g.mul_mat(a, b);

    e.set_value(g, a, &[2.0; 4]);
    e.set_value(g, b, &[2.0; 4]);
    e.evaluate(g, xx);

    panic!("{:?}", e.get_value(xx));

    let x = g.input("x", [BATCH, 1, 1, 1]);
    let a = g.param("a", [1, 1, 1, 1]);
    let b = g.param("b", [1, 1, 1, 1]);

    let y1 = g.mul(a, x);
    let y = g.add(y1, b);

    let w = g.input("w", [BATCH, 1, 1, 1]);

    let loss = g.sub(y, w);
    let loss = g.mul(loss, loss);

    let params = g.backward(loss);

    let mut epoch = 0;
    loop {
        let samples: [f32; BATCH] = random();
        e.step();
        e.set_value(g, x, &samples);
        e.set_value(g, w, &samples.map(|s| target(s)));
        e.evaluate(g, loss);

        for p in &params {
            e.evaluate(g, *p);
        }

        o.optimize(g, e, &params);

        println!("Epoch\t{:?}", epoch);
        println!("Loss:\t{:?}", e.get_value(loss)[0]);
        println!("P(a):\t{:?}", e.get_value(a));
        println!("P(b):\t{:?}", e.get_value(b));

        // sleep(Duration::from_millis(1));
        epoch += 1;
    }
    // println!("{:?}", params);
    // let mut e = Evaluator::default();
    // e.set_value(g, x, vec![2.0; 1]);
    // e.set_value(g, y, vec![1.0; 1]);
    // e.evaluate(g, w);
    // println!("{:#?}", e.get_value(w));

    // let dot = g.to_dot();
    // let dot = dot.to_string();
    // std::fs::write("./graph.dot", &*dot).unwrap();
    // let url = urlencoding::encode(&dot);
    // open::that(format!("https://dreampuf.github.io/GraphvizOnline/#{url}"));
}