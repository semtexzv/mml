use std::fmt::{Debug, Formatter};
use std::mem;
use std::ops::{Index, IndexMut};
use crate::Tensor;

pub struct TensorMap<T> {
    v: Vec<Option<T>>,
}
impl<T: Debug> Debug for TensorMap<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut map = f.debug_map();
        for (i, v) in self.v.iter().enumerate() {
            if let Some(v) = v {
                map.entry(&i, v);
            }
        }
        map.finish()
    }
}

impl<T> Default for TensorMap<T> {
    fn default() -> Self {
        Self { v: Default::default() }
    }
}

impl<T> TensorMap<T> {
    pub fn new() -> Self {
        Self { v: Default::default() }
    }
    pub fn get(&self, t: Tensor) -> Option<&T> {
        if self.v.len() <= t.id {
            return None;
        }
        unsafe { self.v.get_unchecked(t.id).as_ref() }
    }
    pub fn get_mut(&mut self, t: Tensor) -> Option<&mut T> {
        if self.v.len() <= t.id {
            return None;
        }
        unsafe { self.v.get_unchecked_mut(t.id).as_mut() }
    }
    pub fn res(&mut self, t: Tensor) {
        unsafe {
            if self.v.len() <= t.id {
                self.v.reserve(t.id + 1 - self.v.len() + 1);
                self.v.resize_with(t.id + 1, || None);
            }
        }
    }
    pub fn set(&mut self, t: Tensor, val: T) {
        self.res(t);
        unsafe { *self.v.get_unchecked_mut(t.id) = Some(val); }
    }
    pub fn has(&self, t: Tensor) -> bool {
        if t.id >= self.v.len() {
            return false;
        }
        self.v[t.id].is_some()
    }

    pub fn del(&mut self, t: Tensor) {
        if self.v.len() <= t.id {
            return;
        }
        self.v[t.id] = None;
    }
    pub fn entry(&mut self, t: Tensor) -> &mut Option<T> {
        self.res(t);
        self.v.get_mut(t.id).expect("Resize should work")
    }

    pub fn get_many_mut<const N: usize>(
        &mut self,
        indices: [Tensor; N],
    ) -> Option<[&mut T; N]> {
        for i in &indices {
            if self.v[i.id].is_none() {
                panic!("TensorMap::get_many_mut: tensor not found")
            }
        }

        fn get_many_check_valid<const N: usize>(indices: &[Tensor; N], len: usize) -> bool {
            // NB: The optimizer should inline the loops into a sequence
            // of instructions without additional branching.
            let mut valid = true;
            for (i, &idx) in indices.iter().enumerate() {
                valid &= idx.id < len;
                for &idx2 in &indices[..i] {
                    valid &= idx != idx2;
                }
            }
            valid
        }

        get_many_check_valid(&indices, self.v.len());

        let slice: *mut [Option<T>] = self.v.as_mut_slice();
        let mut arr: mem::MaybeUninit<[&mut T; N]> = mem::MaybeUninit::uninit();
        let arr_ptr = arr.as_mut_ptr();

        // SAFETY: We expect `indices` to contain disjunct values that are
        // in bounds of `self`.
        unsafe {
            for i in 0..N {
                let idx = *indices.get_unchecked(i);
                *(*arr_ptr).get_unchecked_mut(i) = (&mut *slice.get_unchecked_mut(idx.id)).as_mut().unwrap();
            }
            Some(arr.assume_init())
        }
    }
}

impl<T> Index<Tensor> for TensorMap<T> {
    type Output = T;

    fn index(&self, index: Tensor) -> &Self::Output {
        self.v[index.id].as_ref().unwrap()
    }
}

impl<T> IndexMut<Tensor> for TensorMap<T> {
    fn index_mut(&mut self, index: Tensor) -> &mut Self::Output {
        self.v[index.id].as_mut().unwrap()
    }
}
