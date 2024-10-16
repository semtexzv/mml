use std::borrow::Cow;
use crate::graph::CGraph;
use crate::tmap::TensorMap;
use crate::{prod, TOp, Tensor, VKind};
use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineDescriptor, Device, Library, MTLResourceOptions, MTLSize};
use rand::prelude::Distribution;
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
        // Implement step logic if needed
    }

    fn write(&mut self, g: &CGraph, t: Tensor, v: &[f32]) {
        // Write data to the buffer associated with the tensor
        let size = prod(g[t].sh) * std::mem::size_of::<f32>();
        let buffer = self.dev.new_buffer_with_data(
            v.as_ptr() as *const _,
            size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        self.buf.set(t, buffer);
    }

    fn read(&self, g: &CGraph, t: Tensor) -> Cow<[f32]> {
        let buffer = self.buf.get(t).expect("Buffer not found");
        let ptr = buffer.contents() as *const f32;
        let len = buffer.length() as usize / std::mem::size_of::<f32>();
        unsafe { Cow::Owned(std::slice::from_raw_parts(ptr, len).to_vec()) }
    }

    fn eval(&mut self, g: &CGraph, dten: Tensor) {
        if !self.buf.has(dten) {
            let size = prod(g[dten].sh) * std::mem::size_of::<f32>();
            let buffer = self.dev.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
            self.buf.set(dten, buffer);
        }

        match g[dten].op {
            TOp::Value(value) => {
                // Handle value initialization
                let buffer = self.buf.get(dten).unwrap();
                let ptr = buffer.contents() as *mut f32;
                let len = buffer.length() as usize / std::mem::size_of::<f32>();
                match value {
                    VKind::Zero => {
                        unsafe {
                            std::ptr::write_bytes(ptr, 0, buffer.length() as usize);
                        }
                    }
                    VKind::One => {
                        let ones = vec![1.0f32; len];
                        unsafe {
                            std::ptr::copy_nonoverlapping(ones.as_ptr(), ptr, len);
                        }
                    }
                    VKind::Val(v) => {
                        let val = v as f32;
                        let vals = vec![val; len];
                        unsafe {
                            std::ptr::copy_nonoverlapping(vals.as_ptr(), ptr, len);
                        }
                    }
                    VKind::Input => {
                        // Input tensors are expected to be initialized via `write()`
                    }
                    VKind::Param => {
                        // Initialize parameters randomly
                        let mut rng = rand::thread_rng();
                        let dist = rand::distributions::Uniform::new(-0.01, 0.01);
                        let data: Vec<f32> = dist.sample_iter(&mut rng).take(len).collect();
                        unsafe {
                            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, len);
                        }
                    }
                }
            }
            TOp::Sum => {
                // Element-wise addition
                let srcs = g[dten].src.clone();
                let src1 = srcs[0];
                let src2 = srcs[1];
                self.eval(g, src1);
                self.eval(g, src2);

                let buffer_a = self.buf.get(src1).unwrap();
                let buffer_b = self.buf.get(src2).unwrap();
                let buffer_result = self.buf.get(dten).unwrap();

                // Create compute pipeline state
                let function = self.lib.get_function("vector_add", None).unwrap();
                let desc = ComputePipelineDescriptor::new();
                desc.set_compute_function(Some(&function));
                let pipeline_state = self.dev.new_compute_pipeline_state(&desc).unwrap();

                // Create command buffer and encoder
                let command_buffer = self.cmd.new_command_buffer();
                let encoder = command_buffer.new_compute_command_encoder();

                // Set pipeline state and buffers
                encoder.set_compute_pipeline_state(&pipeline_state);
                encoder.set_buffer(0, Some(buffer_a), 0);
                encoder.set_buffer(1, Some(buffer_b), 0);
                encoder.set_buffer(2, Some(buffer_result), 0);

                // Determine thread groups and thread group size
                let len = prod(g[dten].sh) as u64;
                let thread_group_size = MTLSize {
                    width: pipeline_state.thread_execution_width() as u64,
                    height: 1,
                    depth: 1,
                };
                let num_threadgroups = MTLSize {
                    width: (len + thread_group_size.width - 1) / thread_group_size.width,
                    height: 1,
                    depth: 1,
                };

                // Dispatch threads
                encoder.dispatch_thread_groups(num_threadgroups, thread_group_size);
                encoder.end_encoding();

                // Commit and wait
                command_buffer.commit();
                command_buffer.wait_until_completed();
            }
            // ... Implement other operations similarly ...
            o => panic!("MPS does not support {:?}", o),
        }
    }

    fn copy(&mut self, g: &CGraph, from: Tensor, to: Tensor) {
        // Ensure both buffers exist
        self.eval(g, from);
        self.eval(g, to);

        let buffer_from = self.buf.get(from).unwrap();
        let buffer_to = self.buf.get(to).unwrap();

        // Copy data from buffer_from to buffer_to
        let command_buffer = self.cmd.new_command_buffer();
        let blit_encoder = command_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            buffer_from,
            0,
            buffer_to,
            0,
            buffer_from.length(),
        );
        blit_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    fn zero_grad(&mut self, g: &CGraph, params: &[Tensor]) {
        for p in params {
            if let Some(grad) = g[*p].grad {
                if let Some(buffer) = self.buf.get(grad) {
                    let ptr = buffer.contents();
                    unsafe {
                        std::ptr::write_bytes(ptr, 0, buffer.length() as usize);
                    }
                }
            }
        }
    }
}