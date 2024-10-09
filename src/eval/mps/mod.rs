use std::borrow::Cow;
use crate::graph::CGraph;
use crate::tmap::TensorMap;
use crate::{prod, Tensor};
use metal::{Buffer, CommandQueue, CompileOptions, Device, Library, MTLResourceOptions};
use crate::eval::Evaluator;

const SOURCES: &str = include_str!("mps.metal");

pub struct MPS {
    dev: Device,
    cmd: CommandQueue,
    lib: Library,
    buf: TensorMap<Buffer>,
}

impl MPS {
    pub fn new() -> Result<Self, String> {
        let dev = Device::system_default().ok_or_else(|| "No MPS device".to_string())?;
        let cmd = dev.new_command_queue();
        let opt = CompileOptions::new();
        let lib = dev.new_library_with_source(SOURCES, &opt)?;

        Ok(Self {
            dev,
            cmd,
            lib,
            buf: Default::default(),
        })
    }

}


impl Evaluator for MPS {
    fn step(&mut self) {
        todo!()
    }

    fn write(&mut self, g: &CGraph, t: Tensor, v: &[f32]) {
        todo!()
    }

    fn read(&self, g: &CGraph, t: Tensor) -> Cow<[f32]> {
        let buf = self.buf.get(t).unwrap();
        unsafe { Cow::Borrowed(std::slice::from_raw_parts(buf.contents() as *mut f32, buf.length() as _)) }
    }

    fn eval(&mut self, g: &CGraph, dten: Tensor) {
        if !self.buf.has(dten) {
            self.buf.set(
                dten,
                self.dev
                    .new_buffer(prod(g[dten].sh) as _, MTLResourceOptions::empty()),
            );
        }
        match g[dten].op {
            o => panic!("MPS does not support {:?}", o),
        }
    }

    fn copy(&mut self, g: &CGraph, from: Tensor, to: Tensor) {
        todo!()
    }

    fn zero_grad(&mut self, g: &CGraph, params: &[Tensor]) {
        todo!()
    }
}