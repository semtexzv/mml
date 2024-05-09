
// TODO, how to ensure buffer safety

use std::ops::{Index, IndexMut};
use cblas::{Layout, Transpose};
use rand::distributions::Distribution;
use crate::graph::CGraph;
use crate::{B, F, H, prod, sprod, Tensor, TOp, VKind, W};
use crate::eval::Evaluator;
use crate::tmap::TensorMap;

#[derive(Debug)]
pub struct BData {
    epoch: usize,
    buf: Vec<f32>,
}

#[derive(Debug, )]
pub struct CPU {
    epoch: usize,
    bufs: TensorMap<BData>,
}

impl Index<Tensor> for CPU {
    type Output = BData;

    fn index(&self, index: Tensor) -> &Self::Output {
        &self.bufs[index]
    }
}

impl IndexMut<Tensor> for CPU {
    fn index_mut(&mut self, index: Tensor) -> &mut Self::Output {
        &mut self.bufs[index]
    }
}

impl Evaluator for CPU {

    fn step(&mut self) {
        self.epoch += 1;
    }
    fn set_value(&mut self, g: &CGraph, ten: Tensor, val: &[f32]) {
        assert_eq!(prod(g[ten].sh), val.len(), "provided data has wrong size");
        let e = self.bufs.entry(ten)
            .get_or_insert_with(|| BData {
                epoch: self.epoch,
                buf: vec![],
            });
        e.buf = val.to_vec();
        e.epoch = self.epoch;
    }
    fn get_value(&self, ten: Tensor) -> &[f32] {
        self.bufs.get(ten).unwrap().buf.as_slice()
    }

    fn evaluate(&mut self, g: &CGraph, ten: Tensor) {
        if self.bufs.get(ten).map(|s| s.epoch).unwrap_or_default() == self.epoch {
            return;
        }

        let srcs = g[ten].sc.clone();
        for s in &srcs {
            self.evaluate(g, *s);
            self[*s].epoch = self.epoch;
        }

        self.do_eval(g, ten, &srcs, g[ten].op)
    }

    fn copy(&mut self, g: &CGraph, from: Tensor, to: Tensor) {
        if !self.bufs.has(to) {
            self.bufs.set(to, BData {
                epoch: 0,
                buf: vec![0.0; prod(g[to].sh)],
            });
        }
        self[to].epoch = self.epoch;
        let [from, to] = self.bufs.get_many_mut([from, to]).unwrap();
        to.buf.iter_mut().zip(from.buf.iter()).for_each(|(d, s)| *d = *s);
    }
}

impl CPU {
    pub fn new() -> Self {
        Self {
            epoch: 1,
            bufs: TensorMap::new(),
        }
    }
}

impl CPU {
    fn perform_unop(d: &mut [f32], s: &[f32], op: fn(f32) -> f32) {
        for (d, s) in d.iter_mut().zip(s.iter()) {
            *d = op(*s);
        }
    }
    fn perform_binop(d: &mut [f32], s1: &[f32], s2: &[f32], op: impl Fn(f32, f32) -> f32) {
        for (d, (s1, s2)) in d.iter_mut().zip(s1.iter().zip(s2.iter())) {
            *d = op(*s1, *s2);
        }
    }
    fn perform_inop(d: &mut [f32], s1: &[f32], op: impl Fn(&mut f32, f32)) {
        for (d, s1) in d.iter_mut().zip(s1.iter()) {
            op(d, *s1);
        }
    }

    fn unop(&mut self, dst: Tensor, srcs: &[Tensor], op: fn(f32) -> f32) {
        assert_ne!(dst, srcs[0]);
        assert_eq!(srcs.len(), 1);
        let [dst, src1] = self.bufs.get_many_mut([dst, srcs[0]]).unwrap();
        Self::perform_unop(&mut dst.buf, &src1.buf, op);
    }
    fn binop(&mut self, dst: Tensor, srcs: &[Tensor], op: impl Fn(f32, f32) -> f32) {
        assert_ne!(dst, srcs[1]);
        assert_eq!(srcs.len(), 2);

        if dst == srcs[0] {
            let [dst, src2] = self.bufs.get_many_mut([dst, srcs[1]]).unwrap();
            Self::perform_inop(&mut dst.buf, &src2.buf, |dst, src| {
                *dst = op(*dst, src)
            })
        } else {
            let [dst, src1, src2] = self.bufs.get_many_mut([dst, srcs[0], srcs[1]]).unwrap();
            Self::perform_binop(&mut dst.buf, &src1.buf, &src2.buf, op);
        }
    }
    fn inop(&mut self, dst: Tensor, src: Tensor, op: fn(&mut f32, f32)) {
        let [dst, src] = self.bufs.get_many_mut([dst, src]).unwrap();
        Self::perform_inop(&mut dst.buf, &src.buf, op)
    }

    fn do_eval(&mut self, g: &CGraph, dten: Tensor, srcs: &[Tensor], op: TOp) {
        // println!("{:?} {:?} {:?}", ten, srcs, op);

        if !self.bufs.has(dten) {
            self.bufs.set(dten, BData {
                epoch: 0,
                buf: vec![0.0; prod(g[dten].sh)],
            });
        }

        match &op {
            TOp::Value(value) => {
                if self.bufs[dten].epoch > 0 {
                    self.bufs[dten].epoch = self.epoch;
                    return;
                }

                match value {
                    VKind::Param => {
                        let dist = rand::distributions::uniform::Uniform::new(-1.0, 1.0);
                        self.bufs[dten].buf
                            .iter_mut()
                            .zip(dist.sample_iter(rand::thread_rng()))
                            .for_each(|(x, y)| *x = y);
                    }
                    VKind::Zero => {
                        self.set_value(g, dten, &[0.0])
                    }
                    VKind::One => {
                        self.set_value(g, dten, &vec![1.0; prod(g[dten].sh)]);
                    }
                    VKind::Input => {}
                }
                self.bufs[dten].epoch = self.epoch;
            }
            TOp::Neg => {
                self.unop(dten, srcs, |a| -a)
            }
            TOp::Repeat { dim, .. } => {
                assert_eq!(*dim, B);
                assert_ne!(dten, g[dten].sc[0]);

                let [dst, src] = self.bufs.get_many_mut([dten, g[dten].sc[0]]).unwrap();
                for batch in 0..(g[dten].sh[B]) {
                    let product = sprod(&g[dten].sh[B + 1..]);
                    for i in 0..product {
                        dst.buf[batch * product + i] = src.buf[i];
                    }
                }
            }
            TOp::SumReduce { dim } => {
                assert_eq!(*dim, B);
                assert_ne!(dten, g[dten].sc[0]);
                let sten = g[dten].sc[0];

                let [dst, src] = self.bufs.get_many_mut([dten, sten]).unwrap();
                let product = sprod(&g[sten].sh[B + 1..]);
                for i in 0..product {
                    dst.buf[i] = 0.0;
                }
                for batch in 0..(g[sten].sh[B]) {
                    for i in 0..product {
                        dst.buf[i] += src.buf[batch * product + i];
                    }
                }
            }
            TOp::MaxReduce { dim } => {
                assert_eq!(*dim, B);
                assert_ne!(dten, g[dten].sc[0]);

                let sten = g[dten].sc[0];

                let [dst, src] = self.bufs.get_many_mut([dten, sten]).unwrap();
                let product = sprod(&g[sten].sh[B + 1..]);
                for i in 0..product {
                    dst.buf[i] = src.buf[i];
                }

                for batch in 1..(g[sten].sh[B]) {
                    for i in 0..product {
                        dst.buf[i] = f32::max(src.buf[i], src.buf[batch * product + i]);
                    }
                }
            }
            TOp::Prod => {
                if srcs.len() == 2 {
                    self.binop(dten, srcs, |a, b| a * b);
                } else {
                    panic!()
                    // self[dten].buf.iter_mut().for_each(|v| *v = 1.0);
                    //
                    // for s in srcs {
                    //     self.inop(dten, *s, |d, s| *d *= s);
                    // }
                }
            }
            TOp::Sum => {
                if srcs.len() == 2 {
                    self.binop(dten, srcs, |a, b| a + b);
                } else {
                    self[dten].buf.iter_mut().for_each(|v| *v = 0.0);

                    for s in srcs {
                        self.inop(dten, *s, |d, s| *d += s);
                    }
                }
            }

            TOp::MatMul => {
                assert_ne!(dten, g[dten].sc[0]);
                assert_ne!(dten, g[dten].sc[1]);
                let sten1 = g[dten].sc[0];
                let sten2 = g[dten].sc[1];

                let dsh = g[dten].sh;
                let sh1 = g[sten1].sh;
                let sh2 = g[sten2].sh;

                assert_eq!(g[sten1].sh[B], 1);
                assert_eq!(g[sten2].sh[B], 1);

                assert_eq!(g[sten1].sh[F], 1);
                assert_eq!(g[sten2].sh[F], 1);

                assert_eq!(sh1[W], sh2[H]);

                let [dst, src1, src2] = self.bufs.get_many_mut([dten, sten1, sten2]).unwrap();

                unsafe {
                    cblas::sgemm(
                        Layout::RowMajor,
                        Transpose::None,
                        Transpose::None,
                        sh1[H] as _,
                        sh2[W] as _,
                        sh2[H] as _,
                        1.0,
                        src1.buf.as_slice(),
                        sh1[W] as _,
                        src2.buf.as_slice(),
                        sh2[W] as _,
                        0.0,
                        dst.buf.as_mut_slice(),
                        dsh[W] as _,
                    );
                }
            }
            op => unimplemented!("{:?}", op),
        }
    }

}
