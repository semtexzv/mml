use metal::{Buffer, CommandQueue, Device, Library, MTLResourceOptions};
use crate::graph::CGraph;
use crate::{prod, Tensor, TOp};
use crate::tmap::TensorMap;

const SHADERS: &[u8] = include_bytes!("mps.rs");

pub struct MPS {
    dev: Device,
    cmd: CommandQueue,
    // lib: Library
    buf: TensorMap<Buffer>,
}

impl MPS {
    pub fn new() -> Result<Self, String> {
        let dev = Device::system_default()
            .ok_or_else(|| "No MPS device".to_string())?;
        let cmd = dev.new_command_queue();
        // let lib = dev.new_library_with_data(SHADERS)?;

        Ok(Self {
            dev,
            cmd,
            // lib
            buf: Default::default(),
        })
    }

    pub fn eval(&mut self, g: &CGraph, dten: Tensor) {
        if !self.buf.has(dten) {
            self.buf.set(dten, self.dev.new_buffer(prod(g[dten].sh) as _, MTLResourceOptions::empty()));
        }
        match g[dten].op {
            TOp::MatMul => {
                let ma = metal::mps::MatrixMultiplication::init(
                    self.dev.as_ref(),
                    1,
                    1,
                    1,
                ).unwrap();
                let buf = self.cmd.new_command_buffer();
                ma.encode_to_command_buffer(buf, todo!(), todo!(), todo!());
                buf.enqueue();
            }
            o => panic!("MPS does not support {:?}", o),
        }
    }
}