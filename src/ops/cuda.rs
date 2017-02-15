use prelude::*;
use ffi::*;
use ops::*;

use densearray::prelude::*;
use devicemem_cuda::prelude::*;

use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashSet};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

impl<'a, T> CursorBufExt<'a> for CursorBuf<DeviceMem<T>> where T: 'a + Copy {
  type Ref = DeviceMemRef<'a, T>;
  type Mut = DeviceMemRefMut<'a, T>;

  fn read_buf(&'a mut self, length: usize) -> DeviceMemRef<'a, T> {
    let start = self.offset;
    let end = self.offset + length;
    self.offset += length;
    self.buffer.as_ref().slice(start, end)
  }

  fn write_buf(&'a mut self, length: usize) -> DeviceMemRefMut<'a, T> {
    let start = self.offset;
    let end = self.offset + length;
    self.offset += length;
    self.buffer.as_mut().slice_mut(start, end)
  }
}

impl ArrayOp<DeviceArray1d<f32>> for ArraySrc<DeviceArray1d<f32>> {
  fn data(&self) -> Rc<ArrayData<DeviceArray1d<f32>>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<DeviceArray1d<f32>> {
  fn _load(&self, txn: TxnId, ref_set: &mut DataRefSet, reader: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if self.data.val.write(txn, node) {
        if reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
          let mut val = self.data.val.get_mut(txn, node);
          let val_len = val.dim();
          let reader = reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
          val.as_view_mut().copy(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
        } else {
          unimplemented!();
        }
      }
    }
  }

  fn _store(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.grad._ref()) {
      if writer.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
        let mut grad = self.data.grad.get(txn, node);
        let grad_len = grad.dim();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.data.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_backward(&self, _txn: TxnId) {
  }

  fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }
}

impl ArrayOp<DeviceArray2d<f32>> for ArraySrc<DeviceArray2d<f32>> {
  fn data(&self) -> Rc<ArrayData<DeviceArray2d<f32>>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<DeviceArray2d<f32>> {
  fn _load(&self, txn: TxnId, ref_set: &mut DataRefSet, reader: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if self.data.val.write(txn, node) {
        if reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
          let mut val = self.data.val.get_mut(txn, node);
          let val_len = val.dim().flat_len();
          let reader = reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
          val.as_view_mut().flatten_mut().copy(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
        } else {
          unimplemented!();
        }
      }
    }
  }

  fn _store(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim().flat_len();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view().flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.grad._ref()) {
      if writer.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
        let mut grad = self.data.grad.get(txn, node);
        let grad_len = grad.dim().flat_len();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view().flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.data.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_backward(&self, _txn: TxnId) {
  }

  fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }
}

impl ArrayOp<DeviceArray4d<f32>> for ArraySrc<DeviceArray4d<f32>> {
  fn data(&self) -> Rc<ArrayData<DeviceArray4d<f32>>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<DeviceArray4d<f32>> {
  fn _load(&self, txn: TxnId, ref_set: &mut DataRefSet, reader: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if self.data.val.write(txn, node) {
        if reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
          let mut val = self.data.val.get_mut(txn, node);
          let val_len = val.dim().flat_len();
          let reader = reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
          val.as_view_mut().flatten_mut().copy(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
        } else {
          unimplemented!();
        }
      }
    }
  }

  fn _store(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim().flat_len();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view().flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.grad._ref()) {
      if writer.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
        let mut grad = self.data.grad.get(txn, node);
        let grad_len = grad.dim().flat_len();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view().flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.data.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_backward(&self, _txn: TxnId) {
  }

  fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }
}
