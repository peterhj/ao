use ffi::*;
use ops::*;

use async_execution::*;
use cuda_dnn::v5::*;
use cuda_dnn::v5::ffi::*;
//use densearray::prelude::*;
use devicemem_cuda::prelude::*;
use fnv::{FnvHashMap};

use std::any::{Any};
use std::cell::{Cell, RefCell};
use std::cmp::{max};
//use std::collections::{HashMap};
//use std::marker::{PhantomData};
use std::ops::{Deref};
use std::rc::{Rc};
use std::sync::{Arc};

pub trait LoadFromDevMemory {
  fn load_from_dev_memory(dst: &mut Self, reader: &mut Any) -> bool;
}

pub trait StoreToDevMemory {
  fn store_to_dev_memory(src: &Self, writer: &mut Any) -> bool;
}

impl LoadFromMemory for DeviceArray1d<f32> {
  fn load_from_memory(dst: &mut Self, reader: &mut Any) -> bool {
    if reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
      let buf_len = dst.dim();
      let reader = reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
      dst.as_view_mut().load_sync(reader.read_buf(buf_len).flatten(), DeviceStream::implicit().conn());
    } else {
      return false;
    }
    true
  }
}

impl StoreToMemory for DeviceArray1d<f32> {
  fn store_to_memory(src: &Self, writer: &mut Any) -> bool {
    if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
      let buf_len = src.dim();
      let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
      src.as_view().store_sync(writer.write_buf(buf_len).flatten_mut(), DeviceStream::implicit().conn());
    } else {
      return false;
    }
    true
  }
}

impl LoadFromMemory for DeviceArray2d<f32> {
  fn load_from_memory(dst: &mut Self, reader: &mut Any) -> bool {
    if reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
      let buf_len = dst.dim().flat_len();
      let reader = reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
      dst.as_view_mut().flatten_mut().load_sync(reader.read_buf(buf_len).flatten(), DeviceStream::implicit().conn());
    } else {
      return false;
    }
    true
  }
}

impl LoadFromDevMemory for DeviceArray2d<f32> {
  fn load_from_dev_memory(dst: &mut Self, reader: &mut Any) -> bool {
    unimplemented!();
  }
}

impl StoreToMemory for DeviceArray2d<f32> {
  fn store_to_memory(src: &Self, writer: &mut Any) -> bool {
    if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
      let buf_len = src.dim().flat_len();
      let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
      src.as_view().flatten().store_sync(writer.write_buf(buf_len).flatten_mut(), DeviceStream::implicit().conn());
    } else {
      return false;
    }
    true
  }
}

impl<'a, T> CursorIoBufExt<'a> for CursorIoBuf<DeviceMem<T>> where T: 'a + Copy {
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

/*impl<T> ArrayOp<DeviceIoBatch<T>> for ArraySrc<DeviceIoBatch<T>> where T: 'static + Copy {
  fn data(&self) -> ArrayData<DeviceIoBatch<T>> {
    self.data.clone()
  }
}*/

impl<T> AutodiffOp for ArraySrc<DeviceIoBatch<T>> where T: 'static + Copy {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      if reader.downcast_mut::<Vec<T>>().is_some() {
        let src_buf = reader.downcast_mut::<Vec<T>>().unwrap();
        let batch_sz = src_buf.len();
        val.set_batch_size(batch_sz);
        val.load(&*src_buf, DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      if writer.downcast_mut::<Vec<T>>().is_some() {
        let dst_buf = writer.downcast_mut::<Vec<T>>().unwrap();
        let batch_sz = val.batch_size();
        assert_eq!(batch_sz, dst_buf.len());
        val.as_ref().store_sync(&mut *dst_buf, DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      unimplemented!();
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      // TODO: zero out the memory.
      unimplemented!();
    }
  }

  fn _r_backward(&self, _txn: TxnId) {
  }

  fn _backward2(&self, _txn: TxnId) {
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

/*impl ArrayOp<DeviceBatchIoMem<u8>> for ArraySrc<DeviceBatchIoMem<u8>> {
  fn data(&self) -> ArrayData<DeviceBatchIoMem<u8>> {
    self.data.clone()
  }
}*/

impl AutodiffOp for ArraySrc<DeviceBatchIoMem<u8>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      if reader.downcast_mut::<Vec<Arc<Deref<Target=[u8]>>>>().is_some() {
        let src_bufs = reader.downcast_mut::<Vec<Arc<Deref<Target=[u8]>>>>().unwrap();
        let batch_sz = src_bufs.len();
        val.set_batch_size(batch_sz, &*DeviceStream::implicit());
        for idx in 0 .. batch_sz {
          val.load(idx, &**src_bufs[idx], DeviceStream::implicit().conn());
        }
        /*let mut tmp = Vec::with_capacity(val.stride());
        tmp.resize(val.stride(), 0);
        val[0].as_ref().store_sync(&mut tmp, DeviceStream::implicit().conn());
        println!("DEBUG: DeviceBatchIoMem input: {:?} readback: {:?}", &src_bufs[0][290 .. 295], &tmp[290 .. 295]);*/
      /*} else if reader.downcast_mut::<(usize, usize, Arc<Deref<Target=[u8]>>)>().is_some() {
        let &mut (ref batch_idx, ref batch_sz, ref src_mem) = reader.downcast_mut::<(usize, usize, Arc<Deref<Target=[u8]>>)>().unwrap();
        let mut val = self.data.val.get_mut(txn, node);
        let val_len = val.stride();
        val.set_batch_size(*batch_sz, &*DeviceStream::implicit());
        val.load(*batch_idx, &**src_mem, DeviceStream::implicit().conn());*/
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    //unimplemented!();
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    //unimplemented!();
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      // TODO: zero out the memory.
      //self.data.r_val.get_excl(txn, node)
      unimplemented!();
    }
  }

  fn _r_backward(&self, _txn: TxnId) {
  }

  fn _backward2(&self, _txn: TxnId) {
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

impl AutodiffOp for PassOp<DeviceBatchArray1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      unimplemented!();
      /*if reader.downcast_mut::<Vec<Arc<Deref<Target=[u8]>>>>().is_some() {
        let src_bufs = reader.downcast_mut::<Vec<Arc<Deref<Target=[u8]>>>>().unwrap();
        let mut val = self.data.val.get_excl(txn, node);
        let batch_sz = src_bufs.len();
        val.set_batch_size(batch_sz, &*DeviceStream::implicit());
        for idx in 0 .. batch_sz {
          val.load(idx, &**src_bufs[idx], DeviceStream::implicit().conn());
        }
      } else {
        unimplemented!();
      }*/
    }
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      let val = self.data.val.get_excl(txn, node);
      if writer.downcast_mut::<Vec<f32>>().is_some() {
        let dst_buf = writer.downcast_mut::<Vec<f32>>().unwrap();
        let x_dim = val.dim();
        let batch_sz = val.batch_size();
        val.as_view().store_sync(dst_buf.reshape_mut((x_dim, batch_sz)), DeviceStream::implicit().conn());
        //println!("DEBUG: PassOp: storing value: {:?}", val.as_view().dim());
        //dst_buf.reshape_mut((x_dim, batch_sz)).set_constant(3.14);
        DeviceStream::implicit().conn().sync();
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    //unimplemented!();
  }

  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      let x_ = self.x_.borrow();
      apply(self);
      x_.as_ref().unwrap()._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_backward(&self, _txn: TxnId) {
  }

  fn _backward2(&self, _txn: TxnId) {
  }
}

/*impl ArrayOp<DeviceArray1d<f32>> for ArraySrc<DeviceArray1d<f32>> {
  fn data(&self) -> ArrayData<DeviceArray1d<f32>> {
    self.data.clone()
  }
}*/

impl AutodiffOp for ArraySrc<DeviceArray1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      /*if reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let val_len = val.dim();
        let reader = reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        val.as_view_mut().load_sync(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
      } else if reader.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let val_len = val.dim();
        let reader = reader.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        val.as_view_mut().copy(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }*/
      assert!(LoadFromMemory::load_from_memory(&mut *val, reader));
    }
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      let val = self.data.val.get_excl(txn, node);
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        val.as_view().store_sync(writer.write_buf(val_len).flatten_mut(), DeviceStream::implicit().conn());
      } else if writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }*/
      assert!(StoreToMemory::store_to_memory(&*val, writer));
    }
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let grad_len = grad.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        grad.as_view().store_sync(writer.write_buf(grad_len).flatten_mut(), DeviceStream::implicit().conn());
      } else if writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let grad_len = grad.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }*/
      assert!(StoreToMemory::store_to_memory(&*grad, writer));
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
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

/*impl ArrayOp<DeviceArray2d<f32>> for ArraySrc<DeviceArray2d<f32>> {
  fn data(&self) -> ArrayData<DeviceArray2d<f32>> {
    self.data.clone()
  }
}*/

impl AutodiffOp for ArraySrc<DeviceArray2d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      /*if reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let val_len = val.dim().flat_len();
        let reader = reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        val.as_view_mut().flatten_mut().load_sync(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
      } else if reader.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let val_len = val.dim().flat_len();
        let reader = reader.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        val.as_view_mut().flatten_mut().copy(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }*/
      assert!(
          LoadFromMemory::load_from_memory(&mut *val, reader) ||
          LoadFromDevMemory::load_from_dev_memory(&mut *val, reader)
      );
    }
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      let val = self.data.val.get_excl(txn, node);
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let val_len = val.dim().flat_len();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        val.as_view().flatten().store_sync(writer.write_buf(val_len).flatten_mut(), DeviceStream::implicit().conn());
      } else if writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let val_len = val.dim().flat_len();
        let writer = writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view().flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }*/
      assert!(StoreToMemory::store_to_memory(&*val, writer));
    }
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let grad_len = grad.dim().flat_len();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        grad.as_view().flatten().store_sync(writer.write_buf(grad_len).flatten_mut(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }*/
      assert!(StoreToMemory::store_to_memory(&*grad, writer));
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
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

/*impl ArrayOp<DeviceArray4d<f32>> for ArraySrc<DeviceArray4d<f32>> {
  fn _data(&self) -> &ArrayData<DeviceArray4d<f32>> {
    &self.data
  }
}*/

impl AutodiffOp for ArraySrc<DeviceArray4d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      if reader.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let val_len = val.dim().flat_len();
        let reader = reader.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        val.as_view_mut().flatten_mut().copy(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      let val = self.data.val.get_excl(txn, node);
      if writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().is_some() {
        let val_len = val.dim().flat_len();
        let writer = writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view().flatten(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let grad_len = grad.dim().flat_len();
        let writer = writer.downcast_mut::<CursorIoBuf<DeviceMem<f32>>>().unwrap();
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
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

//impl<F> AutodiffOp for InitializeOp<DeviceArray1d<f32>, Rc<F>> where F: Fn(Rc<RefCell<ChaChaRng>>, &mut DeviceArray1d<f32>) {
impl AutodiffOp for InitializeOp<DeviceArray1d<f32>, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<DeviceArray1d<f32>>)>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    // Do nothing, `data` belongs to `x`.
  }

  fn _init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {
    let node = self._id();
    /*if self.data.val.overwrite(txn, node) {
      (self.kernel)(seed_rng, &mut *self.data.val.get_excl(txn, node));
    }*/
    (self.kernel)(txn, node, seed_rng, self.data.clone());
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

//impl<F> AutodiffOp for InitializeOp<DeviceArray2d<f32>, Rc<F>> where F: Fn(Rc<RefCell<ChaChaRng>>, &mut DeviceArray2d<f32>) {
impl AutodiffOp for InitializeOp<DeviceArray2d<f32>, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<DeviceArray2d<f32>>)>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    // Do nothing, `data` belongs to `x`.
  }

  fn _init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {
    let node = self._id();
    /*if self.data.val.overwrite(txn, node) {
      (self.kernel)(seed_rng, &mut *self.data.val.get_excl(txn, node));
    }*/
    (self.kernel)(txn, node, seed_rng, self.data.clone());
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

impl<Op> SpecialMapExt</*f32,*/ DeviceArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceArray1d<f32>> {
  fn rect(&self) -> Rc<MapOp<DeviceArray1d<f32>, RectMapKernel>> {
    let clk_horizon = self.data().horizon();
    MapOp::new(RectMapKernel, self.clone(), clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        DeviceArray1d::zeros(dim, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for MapOp<DeviceArray1d<f32>, RectMapKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_fwd_f32(
          x_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let y_dim = self.y.grad.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      let x_dim = self.x.r_val.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.x.r_val.get(txn, node).as_view().wait(&conn);
      self.y.r_val.get_excl(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_bwd_f32(
          x_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.x.r_val.get(txn, node).as_view().as_ptr(),
          self.y.r_val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().post(&conn);
      self.x.r_val.get(txn, node).as_view().post(&conn);
      self.y.r_val.get_excl(txn, node).as_view_mut().post(&conn);
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let y_dim = self.y.r_grad.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.r_grad.get(txn, node).as_view().wait(&conn);
      self.x.r_grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.r_grad.get(txn, node).as_view().as_ptr(),
          self.x.r_grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          DeviceStream::implicit().conn().raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.r_grad.get(txn, node).as_view().post(&conn);
      self.x.r_grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
  }

  fn _backward2(&self, txn: TxnId) {
    let node = self._id();
    if self.x.grad2.accumulate(txn, node, |grad2| grad2.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let y_dim = self.y.grad2.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.grad2.get(txn, node).as_view().wait(&conn);
      self.x.grad2.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad2.get(txn, node).as_view().as_ptr(),
          self.x.grad2.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          DeviceStream::implicit().conn().raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.grad2.get(txn, node).as_view().post(&conn);
      self.x.grad2.get_mut(txn, node).as_view_mut().post(&conn);
    }
  }
}

impl<Op> CastExt<DeviceBatchArray3d<u8>, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<u8>> {
  fn cast(&self) -> Rc<TransformOp<DeviceBatchArray3d<u8>, DeviceBatchArray3d<f32>, CastTransform>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), CastTransform, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray3d<f32>> for TransformOp<DeviceBatchArray3d<u8>, DeviceBatchArray3d<f32>, CastTransform> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray3d<f32>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchArray3d<u8>, DeviceBatchArray3d<f32>, CastTransform> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      {
        let conn = DeviceStream::implicit().conn();
        self.x.val.get(txn, node).as_view().wait(&conn);
        self.y.val.get_excl(txn, node).as_view().wait(&conn);
        unsafe { arraydiff_cuda_kernel_cast_u8_to_f32(
            x_dim.flat_len() * batch_sz,
            self.x.val.get(txn, node).as_view().flatten().as_ptr(),
            self.y.val.get_excl(txn, node).as_view_mut().flatten_mut().as_mut_ptr(),
            conn.raw_stream().as_ptr(),
        ) };
        self.x.val.get(txn, node).as_view().post(&conn);
        self.y.val.get_excl(txn, node).as_view().post(&conn);
      }
      /*let mut tmp_x = Vec::with_capacity(x_dim.flat_len() * batch_sz);
      tmp_x.resize(x_dim.flat_len() * batch_sz, 0);
      let mut tmp_y = Vec::with_capacity(x_dim.flat_len() * batch_sz);
      tmp_y.resize(x_dim.flat_len() * batch_sz, 0.0);
      self.x.val.get(txn, node).as_view().flatten()
        .store_sync(tmp_x.flatten_mut(), DeviceStream::implicit().conn());
      self.y.val.get_excl(txn, node).as_view().flatten()
        .store_sync(tmp_y.flatten_mut(), DeviceStream::implicit().conn());
      println!("DEBUG: cast: x: {:?} y: {:?}", &tmp_x[290 .. 295], &tmp_y[290 .. 295]);*/
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
    /*let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.y.grad.get(txn, node).as_view());
    }*/
  }

  /*fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      self.y.r_val.get_excl(txn, node).as_view_mut().copy(self.x.r_val.get(txn, node).as_view().flatten());
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      self.x.r_grad.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.y.r_grad.get(txn, node).as_view());
    }
  }*/
}

impl<Op> FlattenExt<DeviceBatchArray3d<f32>, DeviceBatchArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn flatten(&self) -> Rc<TransformOp<DeviceBatchArray3d<f32>, DeviceBatchArray1d<f32>, FlattenTransform>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), FlattenTransform, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray1d::zeros(x_dim.flat_len(), batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray1d<f32>> for TransformOp<DeviceBatchArray3d<f32>, DeviceBatchArray1d<f32>, FlattenTransform> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray1d<f32>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchArray3d<f32>, DeviceBatchArray1d<f32>, FlattenTransform> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      self.y.val.get_excl(txn, node).as_view_mut().flatten_mut()
        .copy(self.x.val.get(txn, node).as_view().flatten(), conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let batch_sz = self.y.grad.get(txn, node).batch_size();
      self.x.grad.get_mut(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut()
        .add(1.0, self.y.grad.get(txn, node).as_view().flatten(), conn);
    }
  }
}

impl<Op> ReifyExt<(usize, usize, usize), DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchIoMem<u8>> {
  fn reify(&self, dim: (usize, usize, usize)) -> Rc<TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>, ReifyTransform<(usize, usize, usize)>>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), ReifyTransform{dim: dim}, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        // TODO: DeviceBatchIoMem has no present capacity, only a current size.
        let batch_cap = x.val.get(txn, node).batch_size();
        //let batch_cap = x.val.get(txn, node).batch_capacity();
        //println!("DEBUG: reify: batch cap: {}", batch_cap);
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray3d<u8>> for TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>, ReifyTransform<(usize, usize, usize)>> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray3d<u8>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>, ReifyTransform<(usize, usize, usize)>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let y_dim = self.y.val.get_excl(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      for idx in 0 .. batch_sz {
        let conn = DeviceStream::implicit().conn();
        self.y.val.get_excl(txn, node).as_view_mut()
          .view_mut((0, 0, 0, idx), (y_dim.0, y_dim.1, y_dim.2, idx + 1))
          .flatten_mut()
          .copy(self.x.val.get(txn, node)[idx].as_ref().flatten(), conn);
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
  }
}

impl<Op> MultExt<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceArray1d<f32>> {
  fn mult(&self, x_: Rc<ArrayOp<DeviceArray1d<f32>>>) -> Rc<LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, None, clk_horizon, Rc::new(|_, _| DeviceMem::<f32>::zeros(1, DeviceStream::implicit().conn())))
  }

  fn mult_add(&self, x_: Rc<ArrayOp<DeviceArray1d<f32>>>, b_: Rc<ArrayOp<DeviceMem<f32>>>) -> Rc<LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, Some(b_), clk_horizon, Rc::new(|_, _| DeviceMem::<f32>::zeros(1, DeviceStream::implicit().conn())))
  }
}

impl ArrayOp<DeviceMem<f32>> for LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> {
  fn _data(&self) -> &ArrayData<DeviceMem<f32>> {
    &self.y
  }
}

/*impl AutodiffObjective for LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if self.y.grad.accumulate(txn, node, |grad| grad.as_mut().set_constant(1.0, DeviceStream::implicit().conn())) {
    } else {
      // TODO
      //assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}*/

impl AutodiffOp for LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref b_) = self.b_ {
        b_._pop(epoch, apply);
      }
      self.x_._pop(epoch, apply);
      self.a_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).as_mut().inner_prod(self.a.val.get(txn, node).as_view(), self.x.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
      if let Some(ref b) = self.b {
        self.y.val.get_excl(txn, node).as_mut().reshape_mut(1).add(1.0, b.val.get(txn, node).as_ref().reshape(1), DeviceStream::implicit().conn());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      // FIXME
      self.a.grad.get_mut(txn, node).as_view_mut().add(1.0, self.x.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
      //self.a.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), self.x.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      // FIXME
      self.x.grad.get_mut(txn, node).as_view_mut().add(1.0, self.a.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
      //self.x.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), self.a.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
    }
    if let Some(ref b) = self.b {
      /*if b.grad.accumulate(txn, node, |g| *g = 0.0) {
        *b.grad.get_mut(txn, node) += *self.y.grad.get(txn, node);
      }*/
      unimplemented!();
    }
  }
}

impl<Op> MultExt<DeviceArray2d<f32>, DeviceBatchArray1d<f32>, DeviceBatchArray1d<f32>, DeviceArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceArray2d<f32>> {
  fn mult(&self, x_: Rc<ArrayOp<DeviceBatchArray1d<f32>>>) -> Rc<LinearOp<DeviceArray2d<f32>, DeviceBatchArray1d<f32>, DeviceBatchArray1d<f32>, DeviceArray1d<f32>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), None, clk_horizon, {
      let a = self.data();
      let x = x_.data();
      Rc::new(move |txn, node| {
        let a_dim = a.val.get(txn, node).dim();
        let x_dim = x.val.get(txn, node).dim();
        assert_eq!(a_dim.1, x_dim);
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray1d::zeros(a_dim.0, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }

  fn mult_add(&self, x_: Rc<ArrayOp<DeviceBatchArray1d<f32>>>, b_: Rc<ArrayOp<DeviceArray1d<f32>>>) -> Rc<LinearOp<DeviceArray2d<f32>, DeviceBatchArray1d<f32>, DeviceBatchArray1d<f32>, DeviceArray1d<f32>>> {
    unimplemented!();
  }
}

impl ArrayOp<DeviceBatchArray1d<f32>> for LinearOp<DeviceArray2d<f32>, DeviceBatchArray1d<f32>, DeviceBatchArray1d<f32>, DeviceArray1d<f32>> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray1d<f32>> {
    &self.y
  }
}

impl AutodiffOp for LinearOp<DeviceArray2d<f32>, DeviceBatchArray1d<f32>, DeviceBatchArray1d<f32>, DeviceArray1d<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref b_) = self.b_ {
        b_._pop(epoch, apply);
      }
      self.x_._pop(epoch, apply);
      self.a_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      self.y.val.get_excl(txn, node).as_view_mut()
        .matrix_prod(
            1.0,
            self.a.val.get(txn, node).as_view(), Transpose::N,
            self.x.val.get(txn, node).as_view(), Transpose::N,
            0.0,
            DeviceStream::implicit().conn(),
        );
      if let Some(ref b) = self.b {
        unimplemented!();
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.a.grad.get_mut(txn, node).as_view_mut()
        .matrix_prod(
            1.0,
            self.y.grad.get(txn, node).as_view(), Transpose::N,
            self.x.val.get(txn, node).as_view(), Transpose::T,
            1.0,
            DeviceStream::implicit().conn(),
        );
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.x.grad.get_mut(txn, node).as_view_mut()
        .matrix_prod(
            1.0,
            self.a.val.get(txn, node).as_view(), Transpose::T,
            self.y.grad.get(txn, node).as_view(), Transpose::N,
            1.0,
            DeviceStream::implicit().conn(),
        );
    }
    if let Some(ref b) = self.b {
      unimplemented!();
    }
  }
}

impl<Op> ScalarLinearExt<f32, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<f32> {
  fn scale(&self, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<ElemLinearOp<f32, DeviceBatchArray3d<f32>, ScaleElemKernel>> {
    let clk_horizon = x_.data().horizon();
    ElemLinearOp::new(self.clone(), x_.clone(), None, ScaleElemKernel, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for ElemLinearOp<f32, DeviceBatchArray3d<f32>, ScaleElemKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref b_) = self.b_ {
        b_._pop(epoch, apply);
      }
      self.x_._pop(epoch, apply);
      self.a_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      self.y.val.get_excl(txn, node).as_view_mut()
        .flatten_mut()
        .copy(self.x.val.get(txn, node).as_view().flatten(), DeviceStream::implicit().conn());
      self.y.val.get_excl(txn, node).as_view_mut()
        .flatten_mut()
        .scale(*self.a.val.get(txn, node), DeviceStream::implicit().conn());
      if let Some(ref b) = self.b {
        unimplemented!();
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
    //unimplemented!();
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    // TODO
    unimplemented!();
  }

  fn _backward2(&self, txn: TxnId) {
    // TODO
    unimplemented!();
  }
}

pub struct CudnnConvBackendSize {
  batch_sz:     usize,
  scratch_req:  usize,
  fwd:      CudnnConvFwdOp,
  bwd_w:    CudnnConvBwdFilterOp,
  bwd_d:    CudnnConvBwdDataOp,
  add:      CudnnAddOp,
}

pub struct CudnnConvBackend {
  scratch_sz:   Cell<usize>,
  scratch:  RefCell<DeviceMem<u8>>,
  sizes:    RefCell<FnvHashMap<usize, CudnnConvBackendSize>>,
}

impl<Op> ConvExt<(usize, usize), DeviceArray4d<f32>, DeviceArray1d<f32>, DeviceBatchArray3d<f32>, CudnnConvBackend> for Rc<Op> where Op: 'static + ArrayOp<DeviceArray4d<f32>> {
  fn conv(&self, shape: ConvShape<(usize, usize)>, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<ConvOp<(usize, usize), DeviceArray4d<f32>, DeviceArray1d<f32>, DeviceBatchArray3d<f32>, CudnnConvBackend>> {
    let clk_horizon = x_.data().horizon();
    // TODO: the default of 4096 might need to be decreased.
    let backend = CudnnConvBackend{
      scratch_sz:   Cell::new(4096),
      scratch:      RefCell::new(DeviceMem::zeros(4096, DeviceStream::implicit().conn())),
      sizes:        RefCell::new(FnvHashMap::default()),
    };
    ConvOp::new(shape, self.clone(), x_.clone(), None, backend, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }

  fn conv_add(&self, shape: ConvShape<(usize, usize)>, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>, b_: Rc<ArrayOp<DeviceArray1d<f32>>>) -> Rc<ConvOp<(usize, usize), DeviceArray4d<f32>, DeviceArray1d<f32>, DeviceBatchArray3d<f32>, CudnnConvBackend>> {
    unimplemented!();
  }
}

impl ArrayOp<DeviceBatchArray3d<f32>> for ConvOp<(usize, usize), DeviceArray4d<f32>, DeviceArray1d<f32>, DeviceBatchArray3d<f32>, CudnnConvBackend> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray3d<f32>> {
    &self.y
  }
}

impl AutodiffOp for ConvOp<(usize, usize), DeviceArray4d<f32>, DeviceArray1d<f32>, DeviceBatchArray3d<f32>, CudnnConvBackend> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref b_) = self.b_ {
        b_._pop(epoch, apply);
      }
      self.x_._pop(epoch, apply);
      self.a_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let mut sizes = self.backend.sizes.borrow_mut();
      if !sizes.contains_key(&batch_sz) {
        let mut workspace_size = 0;
        let (in_w, in_h, in_chan) = x_dim;
        let (out_w, out_h, out_chan) = self.shape.conv2d_output_dim(x_dim);
        let (kernel_w, kernel_h) = self.shape.kernel;
        let (stride_w, stride_h) = self.shape.stride;
        let (pad_w, pad_h) = self.shape.zero_pad;
        let conn = DeviceStream::implicit().conn();
        let fwd = CudnnConvFwdOp::create_fastest(
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
            CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
            CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
            &*conn.cudnn(),
        ).unwrap();
        workspace_size = max(workspace_size, fwd.work_size);
        let bwd_w = CudnnConvBwdFilterOp::create_fastest(
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
            CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
            CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
            &*conn.cudnn(),
        ).unwrap();
        workspace_size = max(workspace_size, bwd_w.work_size);
        let bwd_d = CudnnConvBwdDataOp::create_fastest(
            CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
            CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
            &*conn.cudnn(),
        ).unwrap();
        workspace_size = max(workspace_size, bwd_d.work_size);
        let add = CudnnAddOp::new(
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
        );
        let conv = CudnnConvBackendSize{
          batch_sz:     batch_sz,
          scratch_req:  workspace_size,
          fwd:      fwd,
          bwd_w:    bwd_w,
          bwd_d:    bwd_d,
          add:      add,
        };
        sizes.insert(batch_sz, conv);
        if workspace_size > self.backend.scratch_sz.get() {
          self.backend.scratch_sz.set(workspace_size);
          *self.backend.scratch.borrow_mut() = DeviceMem::zeros(workspace_size, conn);
        }
      }
      let conn = DeviceStream::implicit().conn();
      self.a.val.get(txn, node).as_view().wait(&conn);
      if let Some(ref b) = self.b {
        b.val.get(txn, node).as_view().wait(&conn);
      }
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      self.backend.scratch.borrow_mut().as_ref().wait(&conn);
      let conv = sizes.get(&batch_sz).unwrap();
      unsafe { conv.fwd.forward(
          1.0,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.a.val.get(txn, node).as_view().as_ptr(),
          0.0,
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          self.backend.scratch.borrow_mut().as_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      if let Some(ref b) = self.b {
        unsafe { conv.add.forward(
            1.0,
            b.val.get(txn, node).as_view().as_ptr(),
            1.0,
            self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
            &*conn.cudnn(),
        ) }.unwrap();
      }
      self.a.val.get(txn, node).as_view().post(&conn);
      if let Some(ref b) = self.b {
        b.val.get(txn, node).as_view().post(&conn);
      }
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view().post(&conn);
      self.backend.scratch.borrow_mut().as_ref().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    let mut sizes = self.backend.sizes.borrow_mut();
    let conv = sizes.get(&batch_sz).unwrap();
    // TODO: wait-post.
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.a.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { conv.bwd_w.backward_filter(
          1.0,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          1.0,
          self.a.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          self.backend.scratch.borrow_mut().as_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.a.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      self.a.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { conv.bwd_d.backward_data(
          1.0,
          self.a.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          1.0,
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          self.backend.scratch.borrow_mut().as_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      self.a.val.get(txn, node).as_view().post(&conn);
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
    if let Some(ref b) = self.b {
      if b.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
        let conn = DeviceStream::implicit().conn();
        self.y.grad.get(txn, node).as_view().wait(&conn);
        b.grad.get_mut(txn, node).as_view_mut().wait(&conn);
        unsafe { conv.bwd_w.backward_bias(
            1.0,
            self.y.grad.get(txn, node).as_view().as_ptr(),
            1.0,
            b.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
            &*conn.cudnn(),
        ).unwrap() };
        self.y.grad.get(txn, node).as_view().post(&conn);
        b.grad.get_mut(txn, node).as_view_mut().post(&conn);
      }
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      unimplemented!();
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.a.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
  }

  fn _backward2(&self, txn: TxnId) {
    let node = self._id();
    if self.a.grad2.accumulate(txn, node, |grad2| grad2.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
    if self.x.grad2.accumulate(txn, node, |grad2| grad2.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
  }
}

pub struct CudnnPoolBackendSize {
  batch_sz: usize,
  pooling:  CudnnPoolingOp,
}

pub struct CudnnPoolBackend {
  sizes:    RefCell<FnvHashMap<usize, CudnnPoolBackendSize>>,
  //scratch:  RefCell<DeviceMem<u8>>,
}

impl<Kernel, Backend> ArrayOp<DeviceBatchArray3d<f32>> for PoolOp<(usize, usize), DeviceBatchArray3d<f32>, Kernel, Backend> where PoolOp<(usize, usize), DeviceBatchArray3d<f32>, Kernel, Backend>: AutodiffOp {
  fn _data(&self) -> &ArrayData<DeviceBatchArray3d<f32>> {
    &self.y
  }
}

impl AutodiffOp for PoolOp<(usize, usize), DeviceBatchArray3d<f32>, AvgPool, CudnnPoolBackend> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let mut sizes = self.backend.sizes.borrow_mut();
      if !sizes.contains_key(&batch_sz) {
        let (in_w, in_h, chan) = x_dim;
        let (out_w, out_h, _) = self.shape.conv2d_output_dim(x_dim);
        let (kern_w, kern_h) = self.shape.kernel;
        let (stride_w, stride_h) = self.shape.stride;
        let (pad_w, pad_h) = self.shape.zero_pad;
        let pooling = match CudnnPoolingOp::create_2d(
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, chan, batch_sz).unwrap(),
            kern_w,   kern_h,
            stride_w, stride_h,
            pad_w,    pad_h,
            cudnnPoolingMode_t::AverageCountIncludingPadding,
            //cudnnPoolingMode_t::Max,
        ) {
          Err(e) => panic!("failed to create CudnnPoolingOp: {:?}", e),
          Ok(pooling) => pooling,
        };
        let pool = CudnnPoolBackendSize{
          batch_sz: batch_sz,
          pooling:  pooling,
        };
        sizes.insert(batch_sz, pool);
      }
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      let pool = sizes.get(&batch_sz).unwrap();
      unsafe { pool.pooling.forward(
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let mut sizes = self.backend.sizes.borrow_mut();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view().wait(&conn);
      let pool = sizes.get(&batch_sz).unwrap();
      unsafe { pool.pooling.backward(
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view().post(&conn);
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.x.grad.get_mut(txn, node).as_view().post(&conn);
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      unimplemented!();
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
  }

  fn _backward2(&self, txn: TxnId) {
    let node = self._id();
    if self.x.grad2.accumulate(txn, node, |grad2| grad2.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
  }
}

impl AutodiffOp for PoolOp<(usize, usize), DeviceBatchArray3d<f32>, MaxPool, CudnnPoolBackend> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let mut sizes = self.backend.sizes.borrow_mut();
      if !sizes.contains_key(&batch_sz) {
        let (in_w, in_h, chan) = x_dim;
        let (out_w, out_h, _) = self.shape.conv2d_output_dim(x_dim);
        let (kern_w, kern_h) = self.shape.kernel;
        let (stride_w, stride_h) = self.shape.stride;
        let (pad_w, pad_h) = self.shape.zero_pad;
        let pooling = match CudnnPoolingOp::create_2d(
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, chan, batch_sz).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, chan, batch_sz).unwrap(),
            kern_w,   kern_h,
            stride_w, stride_h,
            pad_w,    pad_h,
            cudnnPoolingMode_t::Max,
        ) {
          Err(e) => panic!("failed to create CudnnPoolingOp: {:?}", e),
          Ok(pooling) => pooling,
        };
        let pool = CudnnPoolBackendSize{
          batch_sz: batch_sz,
          pooling:  pooling,
        };
        sizes.insert(batch_sz, pool);
      }
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      let pool = sizes.get(&batch_sz).unwrap();
      unsafe { pool.pooling.forward(
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let mut sizes = self.backend.sizes.borrow_mut();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view().wait(&conn);
      let pool = sizes.get(&batch_sz).unwrap();
      unsafe { pool.pooling.backward(
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          &*conn.cudnn(),
      ) }.unwrap();
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view().post(&conn);
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.x.grad.get_mut(txn, node).as_view().post(&conn);
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      unimplemented!();
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
  }

  fn _backward2(&self, txn: TxnId) {
    let node = self._id();
    if self.x.grad2.accumulate(txn, node, |grad2| grad2.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      unimplemented!();
    }
  }
}

impl<Op> AutodiffSink<Op> for ArraySink<Op, DeviceMem<f32>> where Op: ArrayOp<DeviceMem<f32>> {
  fn _op(&self) -> &AutodiffOp {
    &*self.x_
  }

  fn _set_source(&self, txn: TxnId) {
    let node = self.node;
    if self.x.grad.overwrite(txn, node) {
      self.x.grad.get_excl(txn, node).as_mut()
        .set_constant(1.0, DeviceStream::implicit().conn());
    }
  }
}

impl<Op> BatchSumExt<Op, DeviceIoBatch<f32>, DeviceMem<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceIoBatch<f32>> {
  fn batch_sum(x_: Rc<Op>) -> Rc<BatchJoinOp<DeviceIoBatch<f32>, DeviceMem<f32>, SumJoinKernel>> {
    let clk_horizon = x_.data().horizon();
    BatchJoinOp::new(x_.clone(), SumJoinKernel, clk_horizon, {
      Rc::new(move |_, _| {
        DeviceMem::zeros(1, DeviceStream::implicit().conn())
      })
    })
  }
}

/*impl AutodiffObjective for BatchJoinOp<DeviceIoBatch<f32>, DeviceMem<f32>, SumJoinKernel> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    // TODO
    /*if !self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }*/
    unimplemented!();
  }
}*/

impl AutodiffOp for BatchJoinOp<DeviceIoBatch<f32>, DeviceMem<f32>, SumJoinKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      assert!(batch_sz <= 1024);
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_ref().wait(&conn);
      self.y.val.get_excl(txn, node).as_ref().wait(&conn);
      unsafe { arraydiff_cuda_kernel_blockreduce_sum_f32(
          batch_sz,
          1,
          self.x.val.get(txn, node).as_ref().as_ptr(),
          self.y.val.get_excl(txn, node).as_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_ref().post(&conn);
      self.y.val.get_excl(txn, node).as_ref().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.x.grad.get_mut(txn, node).as_mut()
        .flatten_mut()
        .add_scalar(
            self.y.grad.get(txn, node).as_ref(),
            DeviceStream::implicit().conn(),
        );
    }
  }
}

impl<Op> SoftmaxNLLLossExt<Op, DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  //fn softmax_nll_loss(x_: Rc<Op>, target_: Rc<ArrayOp<DeviceIoBatch<u32>>>) -> (Rc<SoftmaxLoss<DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>, NLLLossLink>>, Rc<PassOp<DeviceBatchArray1d<f32>>>, Rc<PassOp<DeviceIoBatch<f32>>>) {
  fn softmax_nll_loss(x_: Rc<Op>, target_: Rc<ArrayOp<DeviceIoBatch<u32>>>) -> (Rc<PassOp<DeviceBatchArray1d<f32>>>, Rc<PassOp<DeviceIoBatch<f32>>>) {
    let clk_horizon = x_.data().horizon();
    let (_, prob, loss) = SoftmaxLoss::new(x_.clone(), Some(target_.clone()), NLLLossLink, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray1d::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    }, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceIoBatch::zeros(batch_cap, DeviceStream::implicit().conn())
      })
    });
    (prob, loss)
  }
}

/*impl ArrayOp<DeviceIoBatch<f32>> for SoftmaxLoss<DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>, NLLLossLink> {
  fn data(&self) -> ArrayData<DeviceIoBatch<f32>> {
    self.loss.clone()
  }
}*/

impl AutodiffOp for SoftmaxLoss<DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>, NLLLossLink> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      if let Some(ref target_) = self.target_ {
        target_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref target_) = self.target_ {
        target_._pop(epoch, apply);
      }
      self.x_._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.loss.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.loss.val.overwrite(txn, node) {
      assert!(self.prob.val.overwrite(txn, node));
      self.prob.val.get_excl(txn, node).set_batch_size(batch_sz);
      if x_dim <= 1024 {
        let conn = DeviceStream::implicit().conn();
        self.x.val.get(txn, node).as_view().wait(&conn);
        self.prob.val.get_excl(txn, node).as_view().wait(&conn);
        unsafe { arraydiff_cuda_kernel_block_softmax_fwd_f32(
            x_dim,
            batch_sz,
            self.x.val.get(txn, node).as_view().as_ptr(),
            self.prob.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
            conn.raw_stream().as_ptr(),
        ) };
        self.x.val.get(txn, node).as_view().post(&conn);
        self.prob.val.get_excl(txn, node).as_view().post(&conn);
      } else {
        unimplemented!();
      }
      let target = match self.target.as_ref() {
        None    => panic!("SoftmaxLoss with NLL link requires a target"),
        Some(t) => t,
      };
      self.loss.val.get_excl(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      self.prob.val.get_excl(txn, node).as_view().wait(&conn);
      target.val.get(txn, node).as_ref().wait(&conn);
      self.loss.val.get_excl(txn, node).as_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_softmax_nll_loss_fwd_f32(
          x_dim,
          batch_sz,
          self.prob.val.get_excl(txn, node).as_view().as_ptr(),
          target.val.get(txn, node).as_ref().as_ptr(),
          self.loss.val.get_excl(txn, node).as_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.prob.val.get_excl(txn, node).as_view().post(&conn);
      target.val.get(txn, node).as_ref().post(&conn);
      self.loss.val.get_excl(txn, node).as_mut().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.loss.grad.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let target = match self.target.as_ref() {
        None    => panic!("SoftmaxLoss with NLL link requires a target"),
        Some(t) => t,
      };
      let conn = DeviceStream::implicit().conn();
      self.prob.val.get_excl(txn, node).as_view().wait(&conn);
      target.val.get(txn, node).as_ref().wait(&conn);
      self.loss.grad.get(txn, node).as_ref().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_softmax_nll_loss_bwd_f32(
          x_dim,
          batch_sz,
          self.prob.val.get_excl(txn, node).as_view().as_ptr(),
          target.val.get(txn, node).as_ref().as_ptr(),
          self.loss.grad.get(txn, node).as_ref().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.prob.val.get_excl(txn, node).as_view().post(&conn);
      target.val.get(txn, node).as_ref().post(&conn);
      self.loss.grad.get(txn, node).as_ref().post(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }
}
