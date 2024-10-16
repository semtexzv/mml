use std::borrow::Cow;
use std::cmp::max;
use std::collections::HashMap;
use crate::eval::Evaluator;
use crate::graph::CGraph;
use crate::tmap::TensorMap;
use crate::{prod, TOp, Tensor, VKind, B, F, H, W, strd};
use cblas::{Layout, Transpose};
use rand::distributions::Distribution;
use std::ops::{Index, IndexMut};

#[derive(Debug)]
pub struct BData {
    epoch: usize,
    buf: Vec<f32>,
}

#[derive(Debug)]
pub struct CPU {
    epoch: usize,

    buf: TensorMap<BData>,
}

impl Index<Tensor> for CPU {
    type Output = BData;

    fn index(&self, index: Tensor) -> &Self::Output {
        &self.buf[index]
    }
}

impl IndexMut<Tensor> for CPU {
    fn index_mut(&mut self, index: Tensor) -> &mut Self::Output {
        &mut self.buf[index]
    }
}

impl Evaluator for CPU {
    fn step(&mut self) {
        self.epoch += 1;
    }

    fn write(&mut self, g: &CGraph, ten: Tensor, val: &[f32]) {
        let blen = prod(g[ten].sh);
        assert_eq!(prod(g[ten].sh), val.len(), "provided data has wrong size");
        let e = self.buf.entry(ten).get_or_insert_with(|| BData {
            epoch: self.epoch,
            buf: vec![0.0; blen],
        });
        e.buf.iter_mut().zip(val.iter()).for_each(|(d, s)| *d = *s);
        e.epoch = self.epoch;
    }

    fn read(&self, _: &CGraph, ten: Tensor) -> Cow<[f32]> {
        Cow::Borrowed(self.buf.get(ten).unwrap().buf.as_slice())
    }

    fn eval(&mut self, g: &CGraph, ten: Tensor) {
        if self.buf.get(ten).map(|s| s.epoch).unwrap_or_default() == self.epoch {
            return;
        }

        // find topological order among unevaluated nodes.
        // Then for each node
        // if dep == 0: Schedule it
        // Once finished, go through all dependant nodes, decrease dep
        // If dep == 0; schedule

        let mut vis = TensorMap::default();
        let mut dep = HashMap::default();

        fn traverse(g: &CGraph, e: &CPU, t: Tensor, vis: &mut TensorMap<()>, dep: &mut HashMap<Tensor, usize>) {
            vis.set(t, ());
            let mut cnt = 0;
            for s in g[t].src.clone() {
                if !(e.buf.get(s).map(|s| s.epoch).unwrap_or_default() == e.epoch) {
                    cnt += 1;
                }
                if !vis.has(s) {
                    traverse(g, e, s, vis, dep);
                }
            }
            dep.insert(t, cnt);
        }
        traverse(g, self, ten, &mut vis, &mut dep);

        let srcs = g[ten].src.clone();

        for s in &srcs {
            self.eval(g, *s);
            self[*s].epoch = self.epoch;
        }

        self.do_eval(g, ten, &srcs, g[ten].op)
    }

    fn copy(&mut self, g: &CGraph, from: Tensor, to: Tensor) {
        if !self.buf.has(to) {
            self.buf.set(
                to,
                BData {
                    epoch: 0,
                    buf: vec![0.0; prod(g[to].sh)],
                },
            );
        }

        self[to].epoch = self.epoch;

        let [from, to] = self.buf.get_many_mut([from, to]).unwrap();

        to.buf
            .iter_mut()
            .zip(from.buf.iter())
            .for_each(|(d, s)| *d = *s);
    }

    fn zero_grad(&mut self, g: &CGraph, params: &[Tensor]) {
        for p in params {
            if let Some(g) = g[*p].grad {
                if let Some(g) = self.buf.get_mut(g) {
                    g.buf.iter_mut().for_each(|v| *v = 0.0);
                }
            }
        }
    }
}

impl CPU {
    pub fn new() -> Self {
        Self {
            epoch: 1,
            buf: TensorMap::new(),
        }
    }
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

    #[inline(never)]
    fn unop(&mut self, dst: Tensor, srcs: &[Tensor], op: fn(f32) -> f32) {
        assert_ne!(dst, srcs[0]);
        assert_eq!(srcs.len(), 1);

        let [dst, src1] = self.buf.get_many_mut([dst, srcs[0]]).unwrap();
        Self::perform_unop(&mut dst.buf, &src1.buf, op);
    }
    #[inline(never)]
    fn binop(&mut self, g: &CGraph, dst: Tensor, srcs: &[Tensor], op: impl Fn(f32, f32) -> f32) {
        assert_ne!(dst, srcs[0]);
        assert_ne!(dst, srcs[1]);
        assert_eq!(srcs.len(), 2);

        assert_eq!(g[srcs[0]].sh, g[srcs[1]].sh, "Op: {:?} ", g[dst]);

        let [dst, src1, src2] = self.buf.get_many_mut([dst, srcs[0], srcs[1]]).unwrap();
        Self::perform_binop(&mut dst.buf, &src1.buf, &src2.buf, op);
    }

    #[inline(never)]
    fn inop(&mut self, dst: Tensor, src: Tensor, op: fn(&mut f32, f32)) {
        let [dst, src] = self.buf.get_many_mut([dst, src]).unwrap();
        Self::perform_inop(&mut dst.buf, &src.buf, op)
    }

    #[inline(never)]
    fn do_eval(&mut self, g: &CGraph, dten: Tensor, srcs: &[Tensor], op: TOp) {
        if !self.buf.has(dten) {
            self.buf.set(
                dten,
                BData {
                    epoch: 0,
                    buf: vec![0.0; prod(g[dten].sh)],
                },
            );
        }

        match op {
            TOp::Value(value) => {
                if self.buf[dten].epoch > 0 {
                    self.buf[dten].epoch = self.epoch;
                    return;
                }

                match value {
                    VKind::Param => {
                        let dist = rand::distributions::Uniform::new(-0.01, 0.01);

                        self.buf[dten]
                            .buf
                            .iter_mut()
                            .zip(dist.sample_iter(rand::thread_rng()))
                            .for_each(|(x, y)| *x = y / prod(g[dten].sh) as f32);
                    }
                    VKind::Zero => self.write(g, dten, &[0.0]),
                    VKind::One => {
                        self.write(g, dten, &vec![1.0; prod(g[dten].sh)]);
                    }
                    VKind::Input => {}
                    VKind::Val(v) => {
                        self.write(g, dten, &vec![v as _; prod(g[dten].sh)]);
                        println!("Writing val: {:?}", v);
                    }
                }
                self.buf[dten].epoch = self.epoch;
            }
            TOp::Neg => self.unop(dten, srcs, |a| -a),
            TOp::Relu => self.unop(dten, srcs, |a| f32::max(0.0, a)),
            TOp::Exp => self.unop(dten, srcs, |a| f32::exp(a)),
            TOp::Recip => self.unop(dten, srcs, |a| f32::clamp(f32::recip(a), f32::MIN, f32::MAX)),
            TOp::Log => self.unop(dten, srcs, |a| f32::ln(a + 1e-8)),
            TOp::Sqrt => self.unop(dten, srcs, |a| f32::sqrt(a)),
            TOp::Gtz => self.unop(dten, srcs, |a| if a > 0.0 { 1.0 } else { 0.0 }),
            TOp::SoftMax { dim } => {
                let sten = srcs[0];
                assert_ne!(dten, sten);

                let s_sh = g[sten].sh;
                let d_sh = g[dten].sh;

                let [dst, src] = self.buf.get_many_mut([dten, sten]).unwrap();

                let sbuf = &src.buf;
                let dbuf = &mut dst.buf;

                // Number of elements in the tensor
                let total_elems = prod(d_sh);

                // Compute the size and stride for the softmax dimension
                let dim_size = s_sh[dim];
                let outer_elems = total_elems / dim_size;

                // Iterate over each outer dimension
                for outer_idx in 0..outer_elems {
                    let base_offset = outer_idx * dim_size;

                    // Find the maximum value for numerical stability
                    let mut max_val = std::f32::NEG_INFINITY;
                    for i in 0..dim_size {
                        let idx = base_offset + i;
                        max_val = max_val.max(sbuf[idx]);
                    }


                    // For softmax
                    let mut sum_exp = 0.0;
                    for i in 0..dim_size {
                        let idx = base_offset + i;
                        let val = (sbuf[idx] - max_val).exp();
                        dbuf[idx] = val;
                        sum_exp += val;
                    }
                    for i in 0..dim_size {
                        let idx = base_offset + i;
                        dbuf[idx] /= sum_exp;
                    }
                }
            }
            TOp::Repeat { .. } => {
                let sten = g[dten].src[0];
                assert_ne!(dten, sten);

                let ssh = g[sten].sh;
                let str = strd(ssh);

                let dsh = g[dten].sh;
                let dtr = strd(dsh);

                let [dst, src] = self.buf.get_many_mut([dten, g[dten].src[0]]).unwrap();
                let sbuf = &src.buf;
                let dbuf = &mut dst.buf;

                for batch in 0..dsh[B] {
                    for feature in 0..dsh[F] {
                        for row in 0..dsh[H] {
                            for col in 0..dsh[W] {
                                let didx = batch * dtr[B] + feature * dtr[F] + row * dtr[H] + col * dtr[W];
                                let sidx = batch * str[B] + feature * str[F] + row * str[H] + col * str[W];
                                let delem = &mut dbuf[didx];
                                let selem = &sbuf[sidx];
                                *delem = *selem;
                            }
                        }
                    }
                }
            }
            TOp::SumReduce { dim } => {
                let sten = g[dten].src[0];
                assert_ne!(dten, sten);

                let ssh = g[sten].sh;
                let str = strd(ssh);

                let dsh = g[dten].sh;
                let dtr = strd(dsh);

                let [dst, src] = self.buf.get_many_mut([dten, sten]).unwrap();

                let sbuf = &src.buf;
                let dbuf = &mut dst.buf;
                for batch in 0..dsh[B] {
                    for feature in 0..dsh[F] {
                        for row in 0..dsh[H] {
                            for col in 0..dsh[W] {
                                let didx = batch * dtr[B] + feature * dtr[F] + row * dtr[H] + col * dtr[W];
                                let delem = &mut dbuf[didx];
                                *delem = 0.0;
                            }
                        }
                    }
                }

                for batch in 0..ssh[B] {
                    for feature in 0..ssh[F] {
                        for row in 0..ssh[H] {
                            for col in 0..ssh[W] {
                                let didx = batch * dtr[B] + feature * dtr[F] + row * dtr[H] + col * dtr[W];
                                let sidx = batch * str[B] + feature * str[F] + row * str[H] + col * str[W];
                                let delem = &mut dbuf[didx];
                                let selem = &sbuf[sidx];
                                *delem += *selem;
                            }
                        }
                    }
                }
            }

            TOp::MaxReduce { dim } => {
                let sten = g[dten].src[0];
                assert_ne!(dten, sten);

                let ssh = g[sten].sh;
                let str = strd(ssh);

                let dsh = g[dten].sh;
                let dtr = strd(dsh);

                let [dst, src] = self.buf.get_many_mut([dten, sten]).unwrap();

                let sbuf = &src.buf;
                let dbuf = &mut dst.buf;

                for batch in 0..dsh[B] {
                    for feature in 0..dsh[F] {
                        for row in 0..dsh[H] {
                            for col in 0..dsh[W] {
                                let didx = batch * dtr[B] + feature * dtr[F] + row * dtr[H] + col * dtr[W];
                                let sidx = batch * str[B] + feature * str[F] + row * str[H] + col * str[W];
                                let delem = &mut dbuf[didx];
                                let selem = &sbuf[sidx];

                                *delem = *selem;
                            }
                        }
                    }
                }

                for batch in 0..ssh[B] {
                    for feature in 0..ssh[F] {
                        for row in 0..ssh[H] {
                            for col in 0..ssh[W] {
                                let didx = batch * dtr[B] + feature * dtr[F] + row * dtr[H] + col * dtr[W];
                                let sidx = batch * str[B] + feature * str[F] + row * str[H] + col * str[W];
                                let delem = &mut dbuf[didx];
                                let selem = &sbuf[sidx];
                                *delem = f32::max(*delem, *selem);
                            }
                        }
                    }
                }
            }

            TOp::Eq => self.binop(g, dten, srcs, |a, b| if f32::eq(&a, &b) { 1.0 } else { 0.0 }),
            TOp::Pow => self.binop(g, dten, srcs, |a, b| a.powf(b)),
            TOp::Prod => self.binop(g, dten, srcs, |a, b| a * b),
            TOp::Sum => {
                if srcs.len() == 2 {
                    self.binop(g, dten, srcs, |a, b| a + b);
                } else {
                    self[dten].buf.iter_mut().for_each(|v| *v = 0.0);

                    for s in srcs {
                        self.inop(dten, *s, |d, s| *d += s);
                    }
                }
            }

            TOp::MatMul { ta, tb } => {
                assert_ne!(dten, g[dten].src[0]);
                assert_ne!(dten, g[dten].src[1]);

                let sten1 = g[dten].src[0];
                let sten2 = g[dten].src[1];

                let dsh = g[dten].sh;
                let dstrd = strd(dsh);
                let sh1 = g[sten1].sh;
                let s1str = strd(sh1);
                let sh2 = g[sten2].sh;
                let s2str = strd(sh2);

                assert!(sh1[B] == 1 || sh2[B] == 1 || sh1[B] == sh2[B]);
                assert_eq!(dsh[B], max(sh1[B], sh2[B]));

                assert_eq!(g[sten1].sh[F], 1);
                assert_eq!(g[sten2].sh[F], 1);

                let [dst, src1, src2] = self.buf.get_many_mut([dten, sten1, sten2]).unwrap();

                for b0 in 0..max(g[sten1].sh[B], g[sten2].sh[B]) {
                    let abuf = &src1.buf.as_slice()[s1str[B] * b0..];
                    let bbuf = &src2.buf.as_slice()[s2str[B] * b0..];
                    let dbuf = &mut dst.buf.as_mut_slice()[dstrd[B] * b0..];

                    unsafe {
                        cblas::sgemm(
                            Layout::RowMajor,
                            if ta { Transpose::Ordinary } else { Transpose::None },
                            if tb { Transpose::Ordinary } else { Transpose::None },
                            if !ta { sh1[H] } else { sh1[W] } as _,
                            if !tb { sh2[W] } else { sh2[H] } as _,
                            if !tb { sh2[H] } else { sh2[W] } as _,
                            1.0,
                            abuf,
                            sh1[W] as _,
                            bbuf,
                            sh2[W] as _,
                            0.0,
                            dbuf,
                            dsh[W] as _,
                        );
                    }
                }
            }
            op => unimplemented!("{:?}", op),
        }
    }
}