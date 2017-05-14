/*
Copyright 2017 the arraydiff authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use ffi::*;
use ops::*;
use prelude::*;

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
use std::ptr::{null_mut};
use std::rc::{Rc};
use std::sync::{Arc};

impl IoBuf for DeviceMem<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.len();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_mut().load_sync(&reader[offset .. offset + buf_len], DeviceStream::implicit().conn());
      offset += buf_len;
    } else if reader.downcast_mut::<DeviceMem<f32>>().is_some() {
      let reader = reader.downcast_mut::<DeviceMem<f32>>().unwrap();
      dst.as_mut().copy(reader.as_ref().slice(offset, offset + buf_len), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented reader type: {:?}", reader);
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.len();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      src.as_ref().store_sync(&mut writer[offset .. offset + buf_len], DeviceStream::implicit().conn());
      offset += buf_len;
    } else if writer.downcast_mut::<DeviceMem<f32>>().is_some() {
      let writer = writer.downcast_mut::<DeviceMem<f32>>().unwrap();
      writer.as_mut().slice_mut(offset, offset + buf_len).copy(src.as_ref(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented writer type: {:?}", writer);
    }
    offset
  }
}

impl IoBuf for DeviceArray1d<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.dim();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_view_mut().load_sync(reader[offset .. offset + buf_len].flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else if reader.downcast_mut::<DeviceMem<f32>>().is_some() {
      let reader = reader.downcast_mut::<DeviceMem<f32>>().unwrap();
      dst.as_view_mut().copy(reader.as_ref().slice(offset, offset + buf_len).flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented reader type: {:?}", reader);
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.dim();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      src.as_view().store_sync(writer[offset .. offset + buf_len].flatten_mut(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else if writer.downcast_mut::<DeviceMem<f32>>().is_some() {
      let writer = writer.downcast_mut::<DeviceMem<f32>>().unwrap();
      writer.as_mut().slice_mut(offset, offset + buf_len).flatten_mut().copy(src.as_view(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented writer type: {:?}", writer);
    }
    offset
  }
}

impl IoBuf for DeviceArray2d<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.dim().flat_len();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_view_mut().flatten_mut().load_sync(reader[offset .. offset + buf_len].flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else if reader.downcast_mut::<DeviceMem<f32>>().is_some() {
      let reader = reader.downcast_mut::<DeviceMem<f32>>().unwrap();
      dst.as_view_mut().flatten_mut().copy(reader.as_ref().slice(offset, offset + buf_len).flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented reader type: {:?}", reader);
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.dim().flat_len();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      src.as_view().flatten().store_sync(writer[offset .. offset + buf_len].flatten_mut(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else if writer.downcast_mut::<DeviceMem<f32>>().is_some() {
      let writer = writer.downcast_mut::<DeviceMem<f32>>().unwrap();
      writer.as_mut().slice_mut(offset, offset + buf_len).flatten_mut().copy(src.as_view().flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented writer type: {:?}", writer);
    }
    offset
  }
}

impl IoBuf for DeviceArray4d<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.dim().flat_len();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_view_mut().flatten_mut().load_sync(reader[offset .. offset + buf_len].flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else if reader.downcast_mut::<DeviceMem<f32>>().is_some() {
      let reader = reader.downcast_mut::<DeviceMem<f32>>().unwrap();
      dst.as_view_mut().flatten_mut().copy(reader.as_ref().slice(offset, offset + buf_len).flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented reader type: {:?}", reader);
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.dim().flat_len();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      src.as_view().flatten().store_sync(writer[offset .. offset + buf_len].flatten_mut(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else if writer.downcast_mut::<DeviceMem<f32>>().is_some() {
      let writer = writer.downcast_mut::<DeviceMem<f32>>().unwrap();
      writer.as_mut().slice_mut(offset, offset + buf_len).flatten_mut().copy(src.as_view().flatten(), DeviceStream::implicit().conn());
      offset += buf_len;
    } else {
      panic!("store: unimplemented writer type: {:?}", writer);
    }
    offset
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
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      if reader.downcast_mut::<Vec<T>>().is_some() {
        let src_buf = reader.downcast_mut::<Vec<T>>().unwrap();
        let batch_sz = src_buf.len();
        //println!("DEBUG: src: load_val: old bsz: {} new bsz: {}", val.batch_size(), batch_sz);
        val.set_batch_size(batch_sz)
          .load(&*src_buf, DeviceStream::implicit().conn());
        offset += batch_sz;
      } else {
        panic!("store: unimplemented reader type: {:?}", reader);
      }
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      if writer.downcast_mut::<Vec<T>>().is_some() {
        let dst_buf = writer.downcast_mut::<Vec<T>>().unwrap();
        let batch_sz = val.batch_size();
        assert_eq!(batch_sz, dst_buf.len());
        val.as_ref().store_sync(&mut *dst_buf, DeviceStream::implicit().conn());
        offset += batch_sz;
      } else {
        panic!("store: unimplemented writer type: {:?}", writer);
      }
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      unimplemented!();
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      if reader.downcast_mut::<NullIo>().is_some() {
        unimplemented!();
      } else if reader.downcast_mut::<ZeroIo>().is_some() {
        let batch_sz = val.batch_capacity().unwrap_or(val.batch_size());
        val.set_batch_size(batch_sz, &*DeviceStream::implicit());
        for idx in 0 .. batch_sz {
          val[idx].as_mut().set_constant(0, DeviceStream::implicit().conn());
        }
        offset += val.stride() * val.batch_size();
      } else if reader.downcast_mut::<Vec<Arc<Deref<Target=[u8]>>>>().is_some() {
        let src_bufs = reader.downcast_mut::<Vec<Arc<Deref<Target=[u8]>>>>().unwrap();
        let batch_sz = src_bufs.len();
        val.set_batch_size(batch_sz, &*DeviceStream::implicit());
        for idx in 0 .. batch_sz {
          val.load_one(idx, &**src_bufs[idx], DeviceStream::implicit().conn());
        }
        offset += val.stride() * val.batch_size();
        /*let mut tmp = Vec::with_capacity(val.stride());
        tmp.resize(val.stride(), 0);
        val[0].as_ref().store_sync(&mut tmp, DeviceStream::implicit().conn());
        println!("DEBUG: DeviceBatchIoMem input: {:?} readback: {:?}", &src_bufs[0][290 .. 295], &tmp[290 .. 295]);*/
      } else if reader.downcast_mut::<Vec<Arc<Extract<[u8]>>>>().is_some() {
        let src_bufs = reader.downcast_mut::<Vec<Arc<Extract<[u8]>>>>().unwrap();
        let batch_sz = src_bufs.len();
        val.set_batch_size(batch_sz, &*DeviceStream::implicit());
        for idx in 0 .. batch_sz {
          val.extract_load_one(idx, &*src_bufs[idx], DeviceStream::implicit().conn());
        }
        offset += val.stride() * val.batch_size();
      } else {
        panic!("store: unimplemented reader type: {:?}", reader);
      }
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      // TODO
      unimplemented!();
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      // TODO
      unimplemented!();
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for PassOp<DeviceIoBatch<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      unimplemented!();
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      if writer.downcast_mut::<Vec<f32>>().is_some() {
        let dst_buf = writer.downcast_mut::<Vec<f32>>().unwrap();
        let batch_sz = val.batch_size();
        val.as_ref().store_sync(&mut dst_buf[ .. batch_sz], DeviceStream::implicit().conn());
        offset += batch_sz;
      } else {
        panic!("store: unimplemented writer type: {:?}", writer);
      }
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      unimplemented!();
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for PassOp<DeviceBatchArray1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
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
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      if writer.downcast_mut::<Vec<f32>>().is_some() {
        let dst_buf = writer.downcast_mut::<Vec<f32>>().unwrap();
        let x_dim = val.dim();
        let batch_sz = val.batch_size();
        val.as_view().store_sync(dst_buf.reshape_mut((x_dim, batch_sz)), DeviceStream::implicit().conn());
        //println!("DEBUG: PassOp: storing value: {:?}", val.as_view().dim());
        //dst_buf.reshape_mut((x_dim, batch_sz)).set_constant(3.14);
        //DeviceStream::implicit().conn().sync();
        offset += x_dim * batch_sz;
      } else {
        panic!("store: unimplemented writer type: {:?}", writer);
      }
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      unimplemented!();
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for PassOp<DeviceArray1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      offset = IoBuf::load(&mut *val, offset, reader);
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      let next_offset = IoBuf::store(&*val, offset, writer);
      //println!("DEBUG: PassOp 1d: dim: {}", next_offset - offset);
      offset = next_offset;
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      offset = IoBuf::store(&*grad, offset, writer);
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for ArraySrc<DeviceMem<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      offset = IoBuf::load(&mut *val, offset, reader);
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      offset = IoBuf::store(&*val, offset, writer);
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      offset = IoBuf::store(&*grad, offset, writer);
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
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

impl AutodiffOp for ArraySrc<DeviceArray1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      offset = IoBuf::load(&mut *val, offset, reader);
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      offset = IoBuf::store(&*val, offset, writer);
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      offset = IoBuf::store(&*grad, offset, writer);
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for ArraySrc<DeviceArray2d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      offset = IoBuf::load(&mut *val, offset, reader);
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      offset = IoBuf::store(&*val, offset, writer);
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      offset = IoBuf::store(&*grad, offset, writer);
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for ArraySrc<DeviceArray4d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      assert!(self.data.val.overwrite(txn, node));
      let mut val = self.data.val.get_excl(txn, node);
      offset = IoBuf::load(&mut *val, offset, reader);
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      let val = self.data.val.get(txn, node);
      offset = IoBuf::store(&*val, offset, writer);
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      let grad = self.data.grad.get(txn, node);
      offset = IoBuf::store(&*grad, offset, writer);
    }
    offset
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

pub fn normal_linear_init_gpu<R>(mean: f64, std: f64) -> impl Fn(Rc<RefCell<R>>, &mut DeviceArray2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut DeviceArray2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let dist = Normal::new(mean, std);
    let mut h_a = Array2d::zeros(a.dim());
    for e in h_a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
    a.as_view_mut().load_sync(h_a.as_view(), DeviceStream::implicit().conn());
  }
}

pub fn xavier_linear_init_gpu<R>() -> impl Fn(Rc<RefCell<R>>, &mut DeviceArray2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut DeviceArray2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let half_range = (6.0 / (a.dim().0 + a.dim().1) as f64).sqrt();
    let dist = Range::new(-half_range, half_range);
    let mut h_a = Array2d::zeros(a.dim());
    for e in h_a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
    a.as_view_mut().load_sync(h_a.as_view(), DeviceStream::implicit().conn());
  }
}

pub fn kaiming_linear_init_gpu<R>() -> impl Fn(Rc<RefCell<R>>, &mut DeviceArray2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut DeviceArray2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let std = (2.0 / a.dim().1 as f64).sqrt();
    let dist = Normal::new(0.0, std);
    let mut h_a = Array2d::zeros(a.dim());
    for e in h_a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
    a.as_view_mut().load_sync(h_a.as_view(), DeviceStream::implicit().conn());
  }
}

pub fn xavier_conv2d_init_gpu<R>(axes: Axes<(usize, usize)>) -> impl Fn(Rc<RefCell<R>>, &mut DeviceArray4d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut DeviceArray4d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let half_range = match axes {
      Axes((0, 1)) => (6.0 / (a.dim().0 * a.dim().1 * a.dim().2 + a.dim().3) as f64).sqrt(),
      _ => unimplemented!(),
    };
    let dist = Range::new(-half_range, half_range);
    let mut h_a = Array4d::zeros(a.dim());
    for e in h_a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
    a.as_view_mut().load_sync(h_a.as_view(), DeviceStream::implicit().conn());
  }
}

pub fn kaiming_conv2d_init_gpu<R>(axes: Axes<(usize, usize)>) -> impl Fn(Rc<RefCell<R>>, &mut DeviceArray4d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut DeviceArray4d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let std = match axes {
      Axes((0, 1)) => (2.0 / (a.dim().0 * a.dim().1 * a.dim().2) as f64).sqrt(),
      _ => unimplemented!(),
    };
    let dist = Normal::new(0.0, std);
    let mut h_a = Array4d::zeros(a.dim());
    for e in h_a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
    a.as_view_mut().load_sync(h_a.as_view(), DeviceStream::implicit().conn());
  }
}

impl AutodiffOp for InitializeOp<DeviceMem<f32>, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<DeviceMem<f32>>)>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl AutodiffOp for InitializeOp<DeviceArray4d<f32>, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<DeviceArray4d<f32>>)>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    // Do nothing, `data` belongs to `x`.
  }

  fn _init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {
    let node = self._id();
    (self.kernel)(txn, node, seed_rng, self.data.clone());
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

impl ArrayOp<DeviceArray1d<f32>> for BranchOp<Rc<CopyConstant<bool>>, Rc<ArrayOp<DeviceArray1d<f32>>>, Rc<ArrayOp<DeviceArray1d<f32>>>, ArrayData<DeviceArray1d<f32>>> {
  fn _data(&self) -> &ArrayData<DeviceArray1d<f32>> {
    &self.output
  }
}

impl AutodiffOp for BranchOp<Rc<CopyConstant<bool>>, Rc<ArrayOp<DeviceArray1d<f32>>>, Rc<ArrayOp<DeviceArray1d<f32>>>, ArrayData<DeviceArray1d<f32>>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.off_._push(epoch, apply);
      self.on_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.on_._pop(epoch, apply);
      self.off_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.output.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.output.val.overwrite(txn, node) {
      match self.cond.var.get(txn) {
        false => {
          self.output.val.get_excl(txn, node).as_view_mut()
            .copy(self.off.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
        }
        true  => {
          self.output.val.get_excl(txn, node).as_view_mut()
            .copy(self.on.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
        }
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    match self.cond.var.get(txn) {
      false => {
        if self.off.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          self.off.grad.get_mut(txn, node).as_view_mut()
            .add(1.0, self.output.grad.get(txn, node).as_view(), DeviceStream::implicit().conn());
        }
        if self.on.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          // Do nothing.
        }
      }
      true  => {
        if self.off.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          // Do nothing.
        }
        if self.on.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          self.on.grad.get_mut(txn, node).as_view_mut()
            .add(1.0, self.output.grad.get(txn, node).as_view(), DeviceStream::implicit().conn());
        }
      }
    }
  }
}

impl<Op> SpecialMapExt</*f32,*/ DeviceBatchArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  fn rect(&self) -> Rc<MapOp<DeviceBatchArray1d<f32>, RectMapKernel>> {
    let clk_horizon = self.data().horizon();
    MapOp::new(RectMapKernel, self.clone(), clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let batch_cap = x.val.get(txn, node).batch_capacity();
        let x_dim = x.val.get(txn, node).dim();
        DeviceBatchArray1d::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl<Op> SpecialMapExt<DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn rect(&self) -> Rc<MapOp<DeviceBatchArray3d<f32>, RectMapKernel>> {
    let clk_horizon = self.data().horizon();
    MapOp::new(RectMapKernel, self.clone(), clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let batch_cap = x.val.get(txn, node).batch_capacity();
        let x_dim = x.val.get(txn, node).dim();
        DeviceBatchArray3d::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for MapOp<DeviceBatchArray1d<f32>, RectMapKernel> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      let x_dim = self.x.val.get(txn, node).dim();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_fwd_f32(
          x_dim * batch_sz,
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
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_bwd_f32(
          x_dim * batch_sz,
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
      // TODO: batch size.
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
      // TODO: batch size.
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
      // TODO: batch size.
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

impl AutodiffOp for MapOp<DeviceBatchArray3d<f32>, RectMapKernel> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      let x_dim = self.x.val.get(txn, node).dim();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().flatten().wait(&conn);
      self.y.val.get_excl(txn, node).as_view().flatten().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_fwd_f32(
          x_dim.flat_len() * batch_sz,
          self.x.val.get(txn, node).as_view().flatten().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().flatten_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().flatten().post(&conn);
      self.y.val.get_excl(txn, node).as_view().flatten().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().flatten_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().flatten().wait(&conn);
      self.y.grad.get(txn, node).as_view().flatten().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_rect_bwd_f32(
          x_dim.flat_len() * batch_sz,
          self.x.val.get(txn, node).as_view().flatten().as_ptr(),
          self.y.grad.get(txn, node).as_view().flatten().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().flatten().post(&conn);
      self.y.grad.get(txn, node).as_view().flatten().post(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut().post(&conn);
    }
  }

  // TODO: higher order.
}

impl<Op> CastExt<DeviceBatchArray1d<u8>, DeviceBatchArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<u8>> {
  fn cast(&self) -> Rc<TransformOp<DeviceBatchArray1d<u8>, DeviceBatchArray1d<f32>, CastTransform>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), CastTransform, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray1d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray1d<f32>> for TransformOp<DeviceBatchArray1d<u8>, DeviceBatchArray1d<f32>, CastTransform> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray1d<f32>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchArray1d<u8>, DeviceBatchArray1d<f32>, CastTransform> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
            x_dim * batch_sz,
            self.x.val.get(txn, node).as_view().as_ptr(),
            self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
            conn.raw_stream().as_ptr(),
        ) };
        self.x.val.get(txn, node).as_view().post(&conn);
        self.y.val.get_excl(txn, node).as_view().post(&conn);
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
    let clk_horizon = self._data().horizon();
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut()
        .add(1.0, self.y.grad.get(txn, node).as_view().flatten(), conn);
    }
  }
}

impl<Op> ReshapeExt<usize, DeviceBatchIoMem<u8>, DeviceBatchArray1d<u8>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchIoMem<u8>> {
  fn reshape(&self, dim: usize) -> Rc<TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray1d<u8>, ReshapeTransform<usize>>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), ReshapeTransform{dim: dim}, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        // TODO: DeviceBatchIoMem has no present capacity, only a current size.
        let batch_cap = x.val.get(txn, node).batch_capacity().unwrap();
        //println!("DEBUG: reshape: batch cap: {}", batch_cap);
        DeviceBatchArray1d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray1d<u8>> for TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray1d<u8>, ReshapeTransform<usize>> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray1d<u8>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray1d<u8>, ReshapeTransform<usize>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
          .view_mut((0, idx), (y_dim, idx + 1))
          .flatten_mut()
          .copy(self.x.val.get(txn, node)[idx].as_ref().flatten(), conn);
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
  }
}

impl<Op> ReshapeExt<(usize, usize, usize), DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchIoMem<u8>> {
  fn reshape(&self, dim: (usize, usize, usize)) -> Rc<TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>, ReshapeTransform<(usize, usize, usize)>>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), ReshapeTransform{dim: dim}, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        // TODO: DeviceBatchIoMem has no present capacity, only a current size.
        let batch_cap = x.val.get(txn, node).batch_capacity().unwrap();
        //println!("DEBUG: reshape: batch cap: {}", batch_cap);
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray3d<u8>> for TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>, ReshapeTransform<(usize, usize, usize)>> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray3d<u8>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchIoMem<u8>, DeviceBatchArray3d<u8>, ReshapeTransform<(usize, usize, usize)>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
    // TODO
  }
}

impl<Op> ReshapeExt<(), DeviceBatchArray1d<f32>, DeviceIoBatch<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  fn reshape(&self, dim: ()) -> Rc<TransformOp<DeviceBatchArray1d<f32>, DeviceIoBatch<f32>, ReshapeTransform<()>>> {
    let x = self.data();
    let clk_horizon = x.horizon();
    TransformOp::new(self.clone(), ReshapeTransform{dim: dim}, clk_horizon, {
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        assert_eq!(1, x_dim);
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceIoBatch::zeros(batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for TransformOp<DeviceBatchArray1d<f32>, DeviceIoBatch<f32>, ReshapeTransform<()>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    assert_eq!(1, x_dim);
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      self.y.val.get_excl(txn, node).as_mut().flatten_mut()
        .copy(self.x.val.get(txn, node).as_view().flatten(), DeviceStream::implicit().conn());
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    assert_eq!(1, x_dim);
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut()
        .add(1.0, self.y.grad.get(txn, node).as_ref().flatten(), DeviceStream::implicit().conn());
    }
  }
}

impl<Op> ZeroPadExt<DeviceBatchArray3d<f32>, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn zero_pad(&self, axis: usize, dim: usize) -> Rc<TransformOp<DeviceBatchArray3d<f32>, DeviceBatchArray3d<f32>, ZeroPadTransform>> {
    let clk_horizon = self._data().horizon();
    TransformOp::new(self.clone(), ZeroPadTransform{axis: axis, dim: dim}, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        assert!(dim >= x_dim.2);
        let batch_cap = x.val.get(txn, node).batch_capacity();
        match axis {
          2 => {
            let y_dim = (x_dim.0, x_dim.1, dim);
            DeviceBatchArray3d::zeros(y_dim, batch_cap, DeviceStream::implicit().conn())
          }
          _ => unimplemented!(),
        }
      })
    })
  }
}

impl ArrayOp<DeviceBatchArray3d<f32>> for TransformOp<DeviceBatchArray3d<f32>, DeviceBatchArray3d<f32>, ZeroPadTransform> {
  fn _data(&self) -> &ArrayData<DeviceBatchArray3d<f32>> {
    &self.y
  }
}

impl AutodiffOp for TransformOp<DeviceBatchArray3d<f32>, DeviceBatchArray3d<f32>, ZeroPadTransform> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      match self.kernel.axis {
        2 => {
          let y_dim = (x_dim.0, x_dim.1, self.kernel.dim);
          assert_eq!(y_dim, self.y.val.get_excl(txn, node).dim());
          for idx in 0 .. batch_sz {
            self.y.val.get_excl(txn, node).as_view_mut()
              .view_mut((0, 0, 0, idx), (x_dim.0, x_dim.1, x_dim.2, idx + 1)).flatten_mut()
              .copy(self.x.val.get(txn, node).as_view().view((0, 0, 0, idx), (x_dim.0, x_dim.1, x_dim.2, idx + 1)).flatten(), DeviceStream::implicit().conn());
            // FIXME(20170409)
            /*self.y.val.get_excl(txn, node).as_view_mut()
              .view_mut((0, 0, x_dim.2, idx), (y_dim.0, y_dim.1, y_dim.2, idx + 1)).flatten_mut()
              .set_constant(0.0, DeviceStream::implicit().conn());*/
          }
        }
        _ => unimplemented!(),
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      match self.kernel.axis {
        2 => {
          let y_dim = (x_dim.0, x_dim.1, self.kernel.dim);
          assert_eq!(y_dim, self.y.val.get_excl(txn, node).dim());
          for idx in 0 .. batch_sz {
            self.x.grad.get_mut(txn, node).as_view_mut()
              .view_mut((0, 0, 0, idx), (x_dim.0, x_dim.1, x_dim.2, idx + 1)).flatten_mut()
              .add(1.0, self.y.grad.get(txn, node).as_view().view((0, 0, 0, idx), (x_dim.0, x_dim.1, x_dim.2, idx + 1)).flatten(), DeviceStream::implicit().conn());
          }
        }
        _ => unimplemented!(),
      }
    }
  }
}

/*impl<Op> AddExt<DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn add<RhsOp>(&self, x_: Rc<RhsOp>) -> Rc<JoinOp<DeviceBatchArray3d<f32>, SumJoinKernel>> where RhsOp: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
    <Rc<JoinOp<DeviceBatchArray3d<f32>, SumJoinKernel>> as SumExt<DeviceBatchArray3d<f32>>>::sum(vec![ArrayOp::from(self.clone()), ArrayOp::from(x_)])
  }
}*/

impl SumExt<DeviceBatchArray3d<f32>> for Rc<JoinOp<DeviceBatchArray3d<f32>, SumJoinKernel>> {
  fn sum(xs_: Vec<Rc<ArrayOp<DeviceBatchArray3d<f32>>>>) -> Rc<JoinOp<DeviceBatchArray3d<f32>, SumJoinKernel>> {
    let mut clk_horizon0 = None;
    let mut xs = Vec::with_capacity(xs_.len());
    for x_ in xs_.iter() {
      let x = x_.data();
      let clk_horizon = x.horizon();
      match clk_horizon0 {
        None      => clk_horizon0 = Some(clk_horizon),
        Some(hzn) => assert_eq!(hzn, clk_horizon),
      }
      xs.push(x);
    }
    let clk_horizon = clk_horizon0.unwrap();
    JoinOp::new(xs_, SumJoinKernel, clk_horizon, {
      Rc::new(move |txn, node| {
        let mut batch_cap0 = None;
        let mut x_dim0 = None;
        for x in xs.iter() {
          let batch_cap = x.val.get(txn, node).batch_capacity();
          match batch_cap0 {
            None      => batch_cap0 = Some(batch_cap),
            Some(cap) => assert_eq!(cap, batch_cap),
          }
          let x_dim = x.val.get(txn, node).dim();
          match x_dim0 {
            None      => x_dim0 = Some(x_dim),
            Some(dim) => assert_eq!(dim, x_dim),
          }
        }
        let batch_cap = batch_cap0.unwrap();
        let x_dim = x_dim0.unwrap();
        DeviceBatchArray3d::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for JoinOp<DeviceBatchArray3d<f32>, SumJoinKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      for x_ in self.xs_.iter() {
        x_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      for x_ in self.xs_.iter().rev() {
        x_._pop(epoch, apply);
      }
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz0 = self.xs[0].val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz0).as_view_mut()
        .copy(self.xs[0].val.get(txn, node).as_view(), DeviceStream::implicit().conn());
      for x in self.xs.iter().skip(1) {
        let batch_sz = x.val.get(txn, node).batch_size();
        assert_eq!(batch_sz0, batch_sz);
        self.y.val.get_excl(txn, node).as_view_mut().flatten_mut()
          .add(1.0, x.val.get(txn, node).as_view().flatten(), DeviceStream::implicit().conn());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    for x in self.xs.iter() {
      let batch_sz = x.val.get(txn, node).batch_size();
      if x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
        x.grad.get_mut(txn, node).as_view_mut().flatten_mut()
          .add(1.0, self.y.grad.get(txn, node).as_view().flatten(), DeviceStream::implicit().conn());
      }
    }
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

impl<Op> SymmClipExt<DeviceBatchArray1d<f32>, DeviceMem<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  fn symm_unit_clip(&self, c_: Rc<ArrayOp<DeviceMem<f32>>>) -> Rc<ClipOp<DeviceBatchArray1d<f32>, DeviceMem<f32>, SymmUnitClipKernel>> {
    let x = self.data();
    let clk_horizon = x.horizon();
    ClipOp::new(self.clone(), c_, SymmUnitClipKernel, clk_horizon, {
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray1d::<f32>::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl<Op> SymmClipExt<DeviceBatchArray3d<f32>, DeviceMem<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn symm_unit_clip(&self, c_: Rc<ArrayOp<DeviceMem<f32>>>) -> Rc<ClipOp<DeviceBatchArray3d<f32>, DeviceMem<f32>, SymmUnitClipKernel>> {
    let x = self.data();
    let clk_horizon = x.horizon();
    ClipOp::new(self.clone(), c_, SymmUnitClipKernel, clk_horizon, {
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::<f32>::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for ClipOp<DeviceBatchArray1d<f32>, DeviceMem<f32>, SymmUnitClipKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.c_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.c_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      // TODO: wait/post.
      unsafe { arraydiff_cuda_kernel_symm_unit_clip_fwd_f32(
          x_dim.flat_len() * batch_sz,
          self.c.val.get(txn, node).as_ref().as_ptr(),
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.c.grad.accumulate(txn, node, |grad| grad.as_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      // TODO: wait/post.
      unsafe { arraydiff_cuda_kernel_symm_unit_clip_param_bwd_nondeterministic_f32(
          x_dim.flat_len() * batch_sz,
          self.c.val.get(txn, node).as_ref().as_ptr(),
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.c.grad.get_mut(txn, node).as_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      // TODO: wait/post.
      unsafe { arraydiff_cuda_kernel_symm_unit_clip_input_bwd_f32(
          x_dim.flat_len() * batch_sz,
          self.c.val.get(txn, node).as_ref().as_ptr(),
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
  }
}

impl AutodiffOp for ClipOp<DeviceBatchArray3d<f32>, DeviceMem<f32>, SymmUnitClipKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.c_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.c_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let conn = DeviceStream::implicit().conn();
      // TODO: wait/post.
      unsafe { arraydiff_cuda_kernel_symm_unit_clip_fwd_f32(
          x_dim.flat_len() * batch_sz,
          self.c.val.get(txn, node).as_ref().as_ptr(),
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.c.grad.accumulate(txn, node, |grad| grad.as_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      // TODO: wait/post.
      unsafe { arraydiff_cuda_kernel_symm_unit_clip_param_bwd_nondeterministic_f32(
          x_dim.flat_len() * batch_sz,
          self.c.val.get(txn, node).as_ref().as_ptr(),
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.c.grad.get_mut(txn, node).as_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      // TODO: wait/post.
      unsafe { arraydiff_cuda_kernel_symm_unit_clip_input_bwd_f32(
          x_dim.flat_len() * batch_sz,
          self.c.val.get(txn, node).as_ref().as_ptr(),
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
  }
}

impl<Op> BroadcastAddExt<DeviceBatchArray1d<f32>, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  fn broadcast_add(&self, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<BroadcastAddOp<DeviceBatchArray1d<f32>, DeviceBatchArray3d<f32>>> {
    let clk_horizon = x_.data().horizon();
    BroadcastAddOp::new(self.clone(), x_.clone(), clk_horizon, {
      let a = self.data();
      let x = x_.data();
      Rc::new(move |txn, node| {
        let a_dim = a.val.get(txn, node).dim();
        let x_dim = x.val.get(txn, node).dim();
        assert_eq!(a_dim, x_dim.2);
        let batch_cap = a.val.get(txn, node).batch_capacity();
        assert_eq!(batch_cap, x.val.get(txn, node).batch_capacity());
        DeviceBatchArray3d::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for BroadcastAddOp<DeviceBatchArray1d<f32>, DeviceBatchArray3d<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let x_dim = self.x.val.get(txn, node).dim();
      let spatial_dim = x_dim.0 * x_dim.1;
      let chan_dim = x_dim.2;
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.a.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_conv_bcast_add_fwd_f32(
          spatial_dim,
          chan_dim,
          batch_sz,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.a.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.x.val.get(txn, node).as_view().post(&conn);
      self.a.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view_mut().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let spatial_dim = x_dim.0 * x_dim.1;
      let chan_dim = x_dim.2;
      let conn = DeviceStream::implicit().conn();
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.a.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      //if GLOBAL_CONFIG.deterministic {
      unsafe { arraydiff_cuda_kernel_conv_bcast_add_param_bwd_nonatomic_f32(
          spatial_dim,
          chan_dim,
          batch_sz,
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.a.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.a.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let spatial_dim = x_dim.0 * x_dim.1;
      let chan_dim = x_dim.2;
      let conn = DeviceStream::implicit().conn();
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_conv_bcast_add_input_bwd_f32(
          spatial_dim,
          chan_dim,
          batch_sz,
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
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
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
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
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
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
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
    if let Some(ref b) = self.b {
      unimplemented!();
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
  }
}

impl<Op> ElemMultExt<f32, DeviceBatchArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<f32> {
  fn elem_mult(&self, x_: Rc<ArrayOp<DeviceBatchArray1d<f32>>>) -> Rc<ElemLinearOp<f32, DeviceBatchArray1d<f32>, BroadcastMultAddKernel>> {
    let clk_horizon = x_.data().horizon();
    ElemLinearOp::new(self.clone(), x_.clone(), None, BroadcastMultAddKernel, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray1d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }

  fn elem_mult_add(&self, x_: Rc<ArrayOp<DeviceBatchArray1d<f32>>>, b_: Rc<ArrayOp<f32>>) -> Rc<ElemLinearOp<f32, DeviceBatchArray1d<f32>, BroadcastMultAddKernel>> {
    unimplemented!();
  }
}

impl AutodiffOp for ElemLinearOp<f32, DeviceBatchArray1d<f32>, BroadcastMultAddKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
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
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      self.y.val.get_excl(txn, node).as_view_mut()
        .flatten_mut()
        .copy(self.x.val.get(txn, node).as_view().flatten(), DeviceStream::implicit().conn());
      self.y.val.get_excl(txn, node).as_view_mut()
        .flatten_mut()
        .scale(*self.a.val.get(txn, node), DeviceStream::implicit().conn());
      if let Some(ref b) = self.b {
        // TODO
        unimplemented!();
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
    //unimplemented!();
  }
}

impl<Op> ElemMultExt<f32, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<f32> {
  fn elem_mult(&self, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<ElemLinearOp<f32, DeviceBatchArray3d<f32>, BroadcastMultAddKernel>> {
    let clk_horizon = x_.data().horizon();
    ElemLinearOp::new(self.clone(), x_.clone(), None, BroadcastMultAddKernel, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }

  fn elem_mult_add(&self, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>, b_: Rc<ArrayOp<f32>>) -> Rc<ElemLinearOp<f32, DeviceBatchArray3d<f32>, BroadcastMultAddKernel>> {
    unimplemented!();
  }
}

impl AutodiffOp for ElemLinearOp<f32, DeviceBatchArray3d<f32>, BroadcastMultAddKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
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
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      //println!("DEBUG: ElemMultOp: forward: {:?}", txn);
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      //println!("DEBUG: ElemMultOp: get x");
      self.y.val.get_excl(txn, node).as_view_mut()
        .flatten_mut()
        .copy(self.x.val.get(txn, node).as_view().flatten(), DeviceStream::implicit().conn());
      //println!("DEBUG: ElemMultOp: get a");
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

impl<Op> ElemMultExt<DeviceArray1d<f32>, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceArray1d<f32>> {
  fn elem_mult(&self, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<ElemLinearOp<DeviceArray1d<f32>, DeviceBatchArray3d<f32>, BroadcastMultAddKernel>> {
    unimplemented!();
  }

  fn elem_mult_add(&self, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>, b_: Rc<ArrayOp<DeviceArray1d<f32>>>) -> Rc<ElemLinearOp<DeviceArray1d<f32>, DeviceBatchArray3d<f32>, BroadcastMultAddKernel>> {
    let clk_horizon = self._data().horizon();
    ElemLinearOp::new(/*axes,*/ self.clone(), x_.clone(), Some(b_), BroadcastMultAddKernel, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::zeros(x_dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for ElemLinearOp<DeviceArray1d<f32>, DeviceBatchArray3d<f32>, BroadcastMultAddKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
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
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      if let Some(ref b) = self.b {
        let x_dim = self.x.val.get(txn, node).dim();
        let spatial_dim = x_dim.0 * x_dim.1;
        let chan_dim = x_dim.2;
        let conn = DeviceStream::implicit().conn();
        self.x.val.get(txn, node).as_view().wait(&conn);
        self.a.val.get(txn, node).as_view().wait(&conn);
        b.val.get(txn, node).as_view().wait(&conn);
        self.y.val.get_excl(txn, node).as_view_mut().wait(&conn);
        unsafe { arraydiff_cuda_kernel_conv_bcast_mult_add_fwd_f32(
            spatial_dim,
            chan_dim,
            batch_sz,
            self.x.val.get(txn, node).as_view().as_ptr(),
            self.a.val.get(txn, node).as_view().as_ptr(),
            b.val.get(txn, node).as_view().as_ptr(),
            self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
            conn.raw_stream().as_ptr(),
        ) };
        self.x.val.get(txn, node).as_view().post(&conn);
        self.a.val.get(txn, node).as_view().post(&conn);
        b.val.get(txn, node).as_view().post(&conn);
        self.y.val.get_excl(txn, node).as_view_mut().post(&conn);
      } else {
        unimplemented!();
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if let Some(ref b) = self.b {
      if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
        assert!(b.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())));
        let x_dim = self.x.val.get(txn, node).dim();
        let spatial_dim = x_dim.0 * x_dim.1;
        let chan_dim = x_dim.2;
        let conn = DeviceStream::implicit().conn();
        self.x.val.get(txn, node).as_view().wait(&conn);
        self.a.val.get(txn, node).as_view().wait(&conn);
        b.val.get(txn, node).as_view().wait(&conn);
        self.y.grad.get(txn, node).as_view().wait(&conn);
        self.a.grad.get_mut(txn, node).as_view_mut().wait(&conn);
        b.grad.get_mut(txn, node).as_view_mut().wait(&conn);
        if GLOBAL_CONFIG.deterministic {
          unsafe { arraydiff_cuda_kernel_conv_bcast_mult_add_param_bwd_nonatomic_f32(
              spatial_dim,
              chan_dim,
              batch_sz,
              self.x.val.get(txn, node).as_view().as_ptr(),
              self.a.val.get(txn, node).as_view().as_ptr(),
              b.val.get(txn, node).as_view().as_ptr(),
              self.y.grad.get(txn, node).as_view().as_ptr(),
              self.a.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              b.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
        } else {
          unsafe { arraydiff_cuda_kernel_conv_bcast_mult_add_param_bwd_atomic_f32(
              spatial_dim,
              chan_dim,
              batch_sz,
              self.x.val.get(txn, node).as_view().as_ptr(),
              self.a.val.get(txn, node).as_view().as_ptr(),
              b.val.get(txn, node).as_view().as_ptr(),
              self.y.grad.get(txn, node).as_view().as_ptr(),
              self.a.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              b.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
        }
        self.x.val.get(txn, node).as_view().post(&conn);
        self.a.val.get(txn, node).as_view().post(&conn);
        b.val.get(txn, node).as_view().post(&conn);
        self.y.grad.get(txn, node).as_view().post(&conn);
        self.a.grad.get_mut(txn, node).as_view_mut().post(&conn);
        b.grad.get_mut(txn, node).as_view_mut().post(&conn);
      }
    } else {
      if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
        unimplemented!();
      }
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let spatial_dim = x_dim.0 * x_dim.1;
      let chan_dim = x_dim.2;
      let conn = DeviceStream::implicit().conn();
      self.a.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      unsafe { arraydiff_cuda_kernel_conv_bcast_mult_add_input_bwd_f32(
          spatial_dim,
          chan_dim,
          batch_sz,
          self.a.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
      self.a.val.get(txn, node).as_view().post(&conn);
      self.y.grad.get(txn, node).as_view().post(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
    }
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

impl<Op> ElemNormalizeExt<(usize, usize), DeviceArray1d<f32>, DeviceBatchArray3d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn elem_normalize(&self, axes: Axes<(usize, usize)>, epsilon: f64, mean_: Rc<ArrayOp<DeviceArray1d<f32>>>, var_: Rc<ArrayOp<DeviceArray1d<f32>>>) -> Rc<ElemNormalizeOp<(usize, usize), DeviceArray1d<f32>, DeviceBatchArray3d<f32>>> {
    let clk_horizon = self._data().horizon();
    ElemNormalizeOp::new(axes, epsilon, self.clone(), mean_, var_, clk_horizon, {
      let x = self.data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceBatchArray3d::zeros(dim, batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for ElemNormalizeOp<(usize, usize), DeviceArray1d<f32>, DeviceBatchArray3d<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.mean_._push(epoch, apply);
      self.var_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.var_._pop(epoch, apply);
      self.mean_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      match self.axes {
        Axes((0, 1)) => {
          let x_dim = self.x.val.get(txn, node).dim();
          let spatial_dim = x_dim.0 * x_dim.1;
          let chan_dim = x_dim.2;
          let conn = DeviceStream::implicit().conn();
          self.x.val.get(txn, node).as_view().wait(&conn);
          self.mean.val.get(txn, node).as_view().wait(&conn);
          self.var.val.get(txn, node).as_view().wait(&conn);
          self.y.val.get_excl(txn, node).as_view().wait(&conn);
          unsafe { arraydiff_cuda_kernel_conv_normalize_fwd_f32(
              spatial_dim,
              chan_dim,
              batch_sz,
              self.x.val.get(txn, node).as_view().as_ptr(),
              self.mean.val.get(txn, node).as_view().as_ptr(),
              self.var.val.get(txn, node).as_view().as_ptr(),
              self.epsilon as f32,
              self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
          self.x.val.get(txn, node).as_view().post(&conn);
          self.mean.val.get(txn, node).as_view().post(&conn);
          self.var.val.get(txn, node).as_view().post(&conn);
          self.y.val.get_excl(txn, node).as_view().post(&conn);
        }
        _ => unimplemented!(),
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    match self.axes {
      Axes((0, 1)) => {
        let batch_sz = self.x.val.get(txn, node).batch_size();
        let x_dim = self.x.val.get(txn, node).dim();
        let spatial_dim = x_dim.0 * x_dim.1;
        let chan_dim = x_dim.2;
        if self.var.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          let conn = DeviceStream::implicit().conn();
          self.x.val.get(txn, node).as_view().wait(&conn);
          self.mean.val.get(txn, node).as_view().wait(&conn);
          self.var.val.get(txn, node).as_view().wait(&conn);
          self.y.grad.get(txn, node).as_view().wait(&conn);
          self.var.grad.get_mut(txn, node).as_view().wait(&conn);
          if GLOBAL_CONFIG.deterministic {
            unsafe { arraydiff_cuda_kernel_conv_normalize_var_bwd_nonatomic_f32(
                spatial_dim,
                chan_dim,
                batch_sz,
                self.x.val.get(txn, node).as_view().as_ptr(),
                self.mean.val.get(txn, node).as_view().as_ptr(),
                self.var.val.get(txn, node).as_view().as_ptr(),
                self.y.grad.get(txn, node).as_view().as_ptr(),
                self.epsilon as f32,
                self.var.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
                conn.raw_stream().as_ptr(),
            ) };
          } else {
            unsafe { arraydiff_cuda_kernel_conv_normalize_var_bwd_atomic_f32(
                spatial_dim,
                chan_dim,
                batch_sz,
                self.x.val.get(txn, node).as_view().as_ptr(),
                self.mean.val.get(txn, node).as_view().as_ptr(),
                self.var.val.get(txn, node).as_view().as_ptr(),
                self.y.grad.get(txn, node).as_view().as_ptr(),
                self.epsilon as f32,
                self.var.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
                conn.raw_stream().as_ptr(),
            ) };
          }
          self.x.val.get(txn, node).as_view().post(&conn);
          self.mean.val.get(txn, node).as_view().post(&conn);
          self.var.val.get(txn, node).as_view().post(&conn);
          self.y.grad.get(txn, node).as_view().post(&conn);
          self.var.grad.get_mut(txn, node).as_view().post(&conn);
        }
        if self.mean.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          let conn = DeviceStream::implicit().conn();
          self.x.val.get(txn, node).as_view().wait(&conn);
          self.mean.val.get(txn, node).as_view().wait(&conn);
          self.var.val.get(txn, node).as_view().wait(&conn);
          self.var.grad.get_mut(txn, node).as_view().wait(&conn);
          self.y.grad.get(txn, node).as_view().wait(&conn);
          self.mean.grad.get_mut(txn, node).as_view().wait(&conn);
          if GLOBAL_CONFIG.deterministic {
            unsafe { arraydiff_cuda_kernel_conv_normalize_mean_bwd_nonatomic_f32(
                spatial_dim,
                chan_dim,
                batch_sz,
                self.x.val.get(txn, node).as_view().as_ptr(),
                self.mean.val.get(txn, node).as_view().as_ptr(),
                self.var.val.get(txn, node).as_view().as_ptr(),
                self.var.grad.get_mut(txn, node).as_view().as_ptr(),
                self.y.grad.get(txn, node).as_view().as_ptr(),
                self.epsilon as f32,
                self.mean.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
                conn.raw_stream().as_ptr(),
            ) };
          } else {
          unsafe { arraydiff_cuda_kernel_conv_normalize_mean_bwd_atomic_f32(
              spatial_dim,
              chan_dim,
              batch_sz,
              self.x.val.get(txn, node).as_view().as_ptr(),
              self.mean.val.get(txn, node).as_view().as_ptr(),
              self.var.val.get(txn, node).as_view().as_ptr(),
              self.var.grad.get_mut(txn, node).as_view().as_ptr(),
              self.y.grad.get(txn, node).as_view().as_ptr(),
              self.epsilon as f32,
              self.mean.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
          }
          self.x.val.get(txn, node).as_view().post(&conn);
          self.mean.val.get(txn, node).as_view().post(&conn);
          self.var.val.get(txn, node).as_view().post(&conn);
          self.var.grad.get_mut(txn, node).as_view().post(&conn);
          self.y.grad.get(txn, node).as_view().post(&conn);
          self.mean.grad.get_mut(txn, node).as_view().post(&conn);
        }
        if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          let conn = DeviceStream::implicit().conn();
          self.var.val.get(txn, node).as_view().wait(&conn);
          self.y.grad.get(txn, node).as_view().wait(&conn);
          self.x.grad.get_mut(txn, node).as_view().wait(&conn);
          unsafe { arraydiff_cuda_kernel_conv_normalize_input_bwd_f32(
              spatial_dim,
              chan_dim,
              batch_sz,
              self.var.val.get(txn, node).as_view().as_ptr(),
              self.y.grad.get(txn, node).as_view().as_ptr(),
              self.epsilon as f32,
              self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
          self.var.val.get(txn, node).as_view().post(&conn);
          self.y.grad.get(txn, node).as_view().post(&conn);
          self.x.grad.get_mut(txn, node).as_view().post(&conn);
        }
      }
      _ => unimplemented!(),
    }
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
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        //println!("DEBUG: conv: input dim: {:?}", x_dim);
        //println!("DEBUG: conv: output dim: {:?}", shape.conv2d_output_dim(x_dim));
        DeviceBatchArray3d::zeros(shape.conv2d_output_dim(x_dim), batch_cap, DeviceStream::implicit().conn())
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
      self.x_._push(epoch, apply);
      self.a_._push(epoch, apply);
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
      self.a_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
        let (pad_w, pad_h) = self.shape.conv2d_pad_dim(x_dim);
        //let conn = DeviceStream::implicit().conn();
        let fwd = CudnnConvFwdOp::create_fastest(
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
            CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
            CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
            CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
            &*DeviceStream::implicit().conn().cudnn(),
        ).unwrap();
        workspace_size = max(workspace_size, fwd.work_size);
        let bwd_w = if GLOBAL_CONFIG.deterministic {
          CudnnConvBwdFilterOp::create_deterministic(
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
              CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
              CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
              &*DeviceStream::implicit().conn().cudnn(),
          ).unwrap()
        } else {
          CudnnConvBwdFilterOp::create_fastest(
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
              CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
              CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, 1, 1, out_chan, 1).unwrap(),
              &*DeviceStream::implicit().conn().cudnn(),
          ).unwrap()
        };
        workspace_size = max(workspace_size, bwd_w.work_size);
        let bwd_d = if GLOBAL_CONFIG.deterministic {
          CudnnConvBwdDataOp::create_deterministic(
              CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
              CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
              &*DeviceStream::implicit().conn().cudnn(),
          ).unwrap()
        } else {
          CudnnConvBwdDataOp::create_fastest(
              CudnnFilterDesc::<f32>::create_4d(kernel_w, kernel_h, in_chan, out_chan).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, out_w, out_h, out_chan, batch_sz).unwrap(),
              CudnnConvDesc::create_2d(stride_w, stride_h, pad_w, pad_h).unwrap(),
              CudnnTensorDesc::<f32>::create_4d(CudnnTensorLayout::NCHW, in_w, in_h, in_chan, batch_sz).unwrap(),
              &*DeviceStream::implicit().conn().cudnn(),
          ).unwrap()
        };
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
        // FIXME FIXME FIXME(20170321): resizing the global scratch can be really bad!
        if workspace_size > self.backend.scratch_sz.get() {
          DeviceStream::implicit().sync_debug();
          self.backend.scratch_sz.set(workspace_size);
          *self.backend.scratch.borrow_mut() = DeviceMem::zeros(workspace_size, DeviceStream::implicit().conn());
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
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.a.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      self.backend.scratch.borrow_mut().as_mut().wait(&conn);
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
      self.backend.scratch.borrow_mut().as_mut().post(&conn);
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
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let conn = DeviceStream::implicit().conn();
      self.a.val.get(txn, node).as_view().wait(&conn);
      self.y.grad.get(txn, node).as_view().wait(&conn);
      self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
      self.backend.scratch.borrow_mut().as_mut().wait(&conn);
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
      self.backend.scratch.borrow_mut().as_mut().post(&conn);
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }

  fn _backward2(&self, txn: TxnId) {
    unimplemented!();
  }
}

#[derive(Clone, Copy)]
pub enum CaffePoolKind {
  Avg,
  Max,
}

pub trait CaffePoolKernel {
  fn kind() -> CaffePoolKind;
}

impl CaffePoolKernel for AvgPool {
  fn kind() -> CaffePoolKind {
    CaffePoolKind::Avg
  }
}

impl CaffePoolKernel for MaxPool {
  fn kind() -> CaffePoolKind {
    CaffePoolKind::Max
  }
}

pub struct CaffePoolGPUBackend {
  //mask: ArrayData<DeviceBatchArray3d<i32>>,
  //mask: RefCell<DeviceBatchArray3d<i32>>,
  mask: TxnVar<DeviceBatchArray3d<i32>>,
}

impl<Op> PoolExt<(usize, usize), DeviceBatchArray3d<f32>, CaffePoolGPUBackend> for Op where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  //fn avg_pool(shape: PoolShape<(usize, usize)>, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<PoolOp<(usize, usize), DeviceBatchArray3d<f32>, AvgPool, CaffePoolGPUBackend>> {
  fn avg_pool(shape: PoolShape<(usize, usize)>, x_: Rc<Op>) -> Rc<PoolOp<(usize, usize), DeviceBatchArray3d<f32>, AvgPool, CaffePoolGPUBackend>> {
    let clk_horizon = x_._data().horizon();
    let backend = CaffePoolGPUBackend{
      mask: TxnVar::new(Symbol::new(), Val, clk_horizon, {
        let x = x_.data();
        Rc::new(move |txn, node| {
          let x_dim = x.val.get(txn, node).dim();
          let batch_cap = x.val.get(txn, node).batch_capacity();
          //println!("DEBUG: allocating avgpool mask: shape: {:?}", shape.pool2d_output_dim(x_dim));
          DeviceBatchArray3d::zeros(shape.pool2d_output_dim(x_dim), batch_cap, DeviceStream::implicit().conn())
        })
      }),
    };
    PoolOp::new(shape, x_.clone(), AvgPool, backend, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        //println!("DEBUG: avgpool input: shape: {:?}", x_dim);
        //println!("DEBUG: allocating avgpool output: shape: {:?}", shape.pool2d_output_dim(x_dim));
        DeviceBatchArray3d::zeros(shape.pool2d_output_dim(x_dim), batch_cap, DeviceStream::implicit().conn())
      })
    })
  }

  //fn max_pool(shape: PoolShape<(usize, usize)>, x_: Rc<ArrayOp<DeviceBatchArray3d<f32>>>) -> Rc<PoolOp<(usize, usize), DeviceBatchArray3d<f32>, MaxPool, CaffePoolGPUBackend>> {
  fn max_pool(shape: PoolShape<(usize, usize)>, x_: Rc<Op>) -> Rc<PoolOp<(usize, usize), DeviceBatchArray3d<f32>, MaxPool, CaffePoolGPUBackend>> {
    let clk_horizon = x_._data().horizon();
    let backend = CaffePoolGPUBackend{
      mask: TxnVar::new(Symbol::new(), Val, clk_horizon, {
        let x = x_.data();
        Rc::new(move |txn, node| {
          let x_dim = x.val.get(txn, node).dim();
          let batch_cap = x.val.get(txn, node).batch_capacity();
          //println!("DEBUG: allocating maxpool mask: shape: {:?}", shape.pool2d_output_dim(x_dim));
          DeviceBatchArray3d::zeros(shape.pool2d_output_dim(x_dim), batch_cap, DeviceStream::implicit().conn())
        })
      }),
    };
    PoolOp::new(shape, x_.clone(), MaxPool, backend, clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        let x_dim = x.val.get(txn, node).dim();
        let batch_cap = x.val.get(txn, node).batch_capacity();
        //println!("DEBUG: maxpool input: shape: {:?}", x_dim);
        //println!("DEBUG: allocating maxpool output: shape: {:?}", shape.pool2d_output_dim(x_dim));
        DeviceBatchArray3d::zeros(shape.pool2d_output_dim(x_dim), batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl<Kernel> AutodiffOp for PoolOp<(usize, usize), DeviceBatchArray3d<f32>, Kernel, CaffePoolGPUBackend> where Kernel: CaffePoolKernel {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      let pad_dim = self.shape.pool2d_pad_dim(x_dim);
      let y_dim = self.shape.pool2d_output_dim(x_dim);
      let conn = DeviceStream::implicit().conn();
      self.x.val.get(txn, node).as_view().wait(&conn);
      self.y.val.get_excl(txn, node).as_view_mut().wait(&conn);
      match Kernel::kind() {
        CaffePoolKind::Avg => {
          unsafe { arraydiff_cuda_kernel_avg_pool_fwd_f32(
              x_dim.0, x_dim.1, x_dim.2, batch_sz,
              y_dim.0, y_dim.1,
              self.shape.kernel.0, self.shape.kernel.1,
              self.shape.stride.0, self.shape.stride.1,
              pad_dim.0, pad_dim.1,
              self.x.val.get(txn, node).as_view().as_ptr(),
              self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
        }
        CaffePoolKind::Max => {
          unsafe { arraydiff_cuda_kernel_max_pool_fwd_f32(
              x_dim.0, x_dim.1, x_dim.2, batch_sz,
              y_dim.0, y_dim.1,
              self.shape.kernel.0, self.shape.kernel.1,
              self.shape.stride.0, self.shape.stride.1,
              pad_dim.0, pad_dim.1,
              self.x.val.get(txn, node).as_view().as_ptr(),
              self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
              null_mut(),
              conn.raw_stream().as_ptr(),
          ) };
        }
      }
      self.x.val.get(txn, node).as_view().post(&conn);
      self.y.val.get_excl(txn, node).as_view_mut().post(&conn);
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      let x_dim = self.x.val.get(txn, node).dim();
      let pad_dim = self.shape.pool2d_pad_dim(x_dim);
      let y_dim = self.shape.pool2d_output_dim(x_dim);
      match Kernel::kind() {
        CaffePoolKind::Avg => {
          let conn = DeviceStream::implicit().conn();
          self.y.grad.get(txn, node).as_view().wait(&conn);
          self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
          unsafe { arraydiff_cuda_kernel_avg_pool_bwd_f32(
              x_dim.0, x_dim.1, x_dim.2, batch_sz,
              y_dim.0, y_dim.1,
              self.shape.kernel.0, self.shape.kernel.1,
              self.shape.stride.0, self.shape.stride.1,
              pad_dim.0, pad_dim.1,
              self.y.grad.get(txn, node).as_view().as_ptr(),
              self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
          self.y.grad.get(txn, node).as_view().post(&conn);
          self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
        }
        CaffePoolKind::Max => {
          assert!(self.backend.mask.overwrite(txn, node));
          let conn = DeviceStream::implicit().conn();
          self.y.grad.get(txn, node).as_view().wait(&conn);
          self.x.grad.get_mut(txn, node).as_view_mut().wait(&conn);
          self.backend.mask.get_excl(txn, node).as_view().wait(&conn);
          unsafe { arraydiff_cuda_kernel_max_pool_fwd_f32(
              x_dim.0, x_dim.1, x_dim.2, batch_sz,
              y_dim.0, y_dim.1,
              self.shape.kernel.0, self.shape.kernel.1,
              self.shape.stride.0, self.shape.stride.1,
              pad_dim.0, pad_dim.1,
              self.x.val.get(txn, node).as_view().as_ptr(),
              null_mut(),
              self.backend.mask.get_excl(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
          unsafe { arraydiff_cuda_kernel_max_pool_bwd_f32(
              x_dim.0, x_dim.1, x_dim.2, batch_sz,
              y_dim.0, y_dim.1,
              self.shape.kernel.0, self.shape.kernel.1,
              self.shape.stride.0, self.shape.stride.1,
              pad_dim.0, pad_dim.1,
              self.y.grad.get(txn, node).as_view().as_ptr(),
              self.backend.mask.get_excl(txn, node).as_view().as_ptr(),
              self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
              conn.raw_stream().as_ptr(),
          ) };
          self.backend.mask.get_excl(txn, node).as_view().post(&conn);
          self.y.grad.get(txn, node).as_view().post(&conn);
          self.x.grad.get_mut(txn, node).as_view_mut().post(&conn);
        }
      }
    }
  }
}

/*pub struct CudnnPoolBackendSize {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
        //let (pad_w, pad_h) = self.shape.zero_pad;
        let (pad_w, pad_h) = self.shape.conv2d_pad_dim(x_dim);
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
        //let (pad_w, pad_h) = self.shape.zero_pad;
        let (pad_w, pad_h) = self.shape.conv2d_pad_dim(x_dim);
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
}*/

impl<Op> BatchStatsExt<(usize, usize), DeviceBatchArray3d<f32>, DeviceArray1d<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray3d<f32>> {
  fn batch_stats(reduce_axes: Axes<(usize, usize)>, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Rc<Op>) -> BatchStatsOutput<DeviceArray1d<f32>> {
    let clk_horizon = x_._data().horizon();
    BatchStatsOp::<(usize, usize), DeviceBatchArray3d<f32>, DeviceArray1d<f32>>::new(reduce_axes, cfg, ctrl, x_.clone(), clk_horizon, {
      let x = x_.data();
      Rc::new(move |txn, node| {
        // FIXME(20170323): the following `rollover` is a little bit of a hack.
        x.val.rollover(txn, &mut x.vars());
        let x_dim = x.val.get(txn, node).dim();
        match reduce_axes {
          Axes((0, 1)) => DeviceArray1d::zeros(x_dim.2, DeviceStream::implicit().conn()),
          _ => unimplemented!(),
        }
      })
    })
  }
}

impl BatchStatsOpExt for BatchStatsOp<(usize, usize), DeviceBatchArray3d<f32>, DeviceArray1d<f32>> {
  fn _configure(&self, reconf: &Fn(&mut BatchStatsConfig)) {
    // FIXME(20170214): only safe to mutate state at the beginning of a txn.
    let mut state = self.state.borrow_mut();
    reconf(&mut state.cfg);
  }

  fn _reset_accumulators(&self, txn: TxnId) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    //self.mean_acc.val.accumulate(txn, node, |val| val.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn()));
    //self.var_acc.val.accumulate(txn, node, |val| val.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn()));
    state.batch_ct = 0;
  }

  fn _accumulate(&self, txn: TxnId) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    // FIXME: does not account for non-uniform batch sizes.
    //let batch_sz = self.x.val.get(txn, node).batch_size();
    let n = (state.batch_ct + 1) as f32;
    if state.batch_ct > 0 {
      self.mean_acc.val.rollover(txn, &mut self.mean_acc.val.var().singleton());
    }
    if self.mean_acc.val.accumulate(txn, node, |val| val.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.mean_acc.val.get_mut(txn, node).as_view_mut().average(1.0 / n, self.mean.val.get_excl(txn, node).as_view(), DeviceStream::implicit().conn());
    }
    if state.batch_ct > 0 {
      self.var_acc.val.rollover(txn, &mut self.var_acc.val.var().singleton());
    }
    if self.var_acc.val.accumulate(txn, node, |val| val.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.var_acc.val.get_mut(txn, node).as_view_mut().average(1.0 / n, self.var.val.get_excl(txn, node).as_view(), DeviceStream::implicit().conn());
    }
    state.batch_ct += 1;
  }

  fn _update_stats(&self, prev_txn: TxnId, next_txn: TxnId) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    assert!(state.batch_ct >= 1);
    let rate = state.cfg.average.rate(state.update_ct) as f32;
    if self.mean_run.val.accumulate(next_txn, node, |val| val.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.mean_run.val.get_mut(next_txn, node).as_view_mut().average(rate, self.mean_acc.val.get(prev_txn, node).as_view(), DeviceStream::implicit().conn());
      assert!(self.mean_acc.val.overwrite(next_txn, node));
      self.mean_acc.val.get_excl(next_txn, node).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn());
    }
    if self.var_run.val.accumulate(next_txn, node, |val| val.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      self.var_run.val.get_mut(next_txn, node).as_view_mut().average(rate, self.var_acc.val.get(prev_txn, node).as_view(), DeviceStream::implicit().conn());
      assert!(self.var_acc.val.overwrite(next_txn, node));
      self.var_acc.val.get_excl(next_txn, node).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn());
    }
    self.mean_acc.val.rollover(next_txn, &mut self.mean_acc.val.var().singleton());
    self.mean_run.val.rollover(next_txn, &mut self.mean_run.val.var().singleton());
    self.var_acc.val.rollover(next_txn, &mut self.var_acc.val.var().singleton());
    self.var_run.val.rollover(next_txn, &mut self.var_run.val.var().singleton());
    state.batch_ct = 0;
    state.update_ct += 1;
  }
}

impl AutodiffOp for BatchStatsOp<(usize, usize), DeviceBatchArray3d<f32>, DeviceArray1d<f32>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.mean.rollover_all(txn, vars);
    self.var.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    state.mode.var.persist(txn);
    match state.mode.var.get(txn) {
      false => {
        let batch_sz = self.x.val.get(txn, node).batch_size();
        if self.mean.val.overwrite(txn, node) {
          self.mean.val.get_excl(txn, node).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn());
          match state.reduce_axes {
            Axes((0, 1)) => {
              let x_dim = self.x.val.get(txn, node).dim();
              let spatial_dim = x_dim.0 * x_dim.1;
              let chan_dim = x_dim.2;
              let conn = DeviceStream::implicit().conn();
              self.x.val.get(txn, node).as_view().wait(&conn);
              self.mean.val.get_excl(txn, node).as_view().wait(&conn);
              if GLOBAL_CONFIG.deterministic {
                unsafe { arraydiff_cuda_kernel_conv_batch_stats_mean_fwd_nonatomic_f32(
                    spatial_dim,
                    chan_dim,
                    batch_sz,
                    self.x.val.get(txn, node).as_view().as_ptr(),
                    self.mean.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
                    conn.raw_stream().as_ptr(),
                ) };
              } else {
                unsafe { arraydiff_cuda_kernel_conv_batch_stats_mean_fwd_atomic_f32(
                    spatial_dim,
                    chan_dim,
                    batch_sz,
                    self.x.val.get(txn, node).as_view().as_ptr(),
                    self.mean.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
                    conn.raw_stream().as_ptr(),
                ) };
              }
              self.x.val.get(txn, node).as_view().post(&conn);
              self.mean.val.get_excl(txn, node).as_view().post(&conn);
            }
            _ => unimplemented!(),
          }
        }
        if self.var.val.overwrite(txn, node) {
          self.var.val.get_excl(txn, node).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn());
          match state.reduce_axes {
            Axes((0, 1)) => {
              let x_dim = self.x.val.get(txn, node).dim();
              let spatial_dim = x_dim.0 * x_dim.1;
              let chan_dim = x_dim.2;
              let conn = DeviceStream::implicit().conn();
              self.x.val.get(txn, node).as_view().wait(&conn);
              self.mean.val.get_excl(txn, node).as_view().wait(&conn);
              self.var.val.get_excl(txn, node).as_view().wait(&conn);
              if GLOBAL_CONFIG.deterministic {
                unsafe { arraydiff_cuda_kernel_conv_batch_stats_var_fwd_nonatomic_f32(
                    spatial_dim,
                    chan_dim,
                    batch_sz,
                    self.x.val.get(txn, node).as_view().as_ptr(),
                    self.mean.val.get_excl(txn, node).as_view().as_ptr(),
                    self.var.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
                    conn.raw_stream().as_ptr(),
                ) };
              } else {
                unsafe { arraydiff_cuda_kernel_conv_batch_stats_var_fwd_atomic_f32(
                    spatial_dim,
                    chan_dim,
                    batch_sz,
                    self.x.val.get(txn, node).as_view().as_ptr(),
                    self.mean.val.get_excl(txn, node).as_view().as_ptr(),
                    self.var.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
                    conn.raw_stream().as_ptr(),
                ) };
              }
              self.x.val.get(txn, node).as_view().post(&conn);
              self.mean.val.get_excl(txn, node).as_view().post(&conn);
              self.var.val.get_excl(txn, node).as_view().post(&conn);
            }
            _ => unimplemented!(),
          }
        }
      }
      true => {
        // Do nothing.
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    state.mode.var.persist(txn);
    match state.mode.var.get(txn) {
      false => {
        let batch_sz = self.x.val.get(txn, node).batch_size();
        if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
          match state.reduce_axes {
            Axes((0, 1)) => {
              let x_dim = self.x.val.get(txn, node).dim();
              let spatial_dim = x_dim.0 * x_dim.1;
              let chan_dim = x_dim.2;
              let conn = DeviceStream::implicit().conn();
              self.x.val.get(txn, node).as_view().wait(&conn);
              self.mean.val.get_excl(txn, node).as_view().wait(&conn);
              self.mean.grad.get(txn, node).as_view().wait(&conn);
              self.var.grad.get(txn, node).as_view().wait(&conn);
              self.x.grad.get_mut(txn, node).as_view().wait(&conn);
              unsafe { arraydiff_cuda_kernel_conv_batch_stats_bwd_f32(
                  spatial_dim,
                  chan_dim,
                  batch_sz,
                  self.x.val.get(txn, node).as_view().as_ptr(),
                  self.mean.val.get_excl(txn, node).as_view().as_ptr(),
                  self.mean.grad.get(txn, node).as_view().as_ptr(),
                  self.var.grad.get(txn, node).as_view().as_ptr(),
                  self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
                  conn.raw_stream().as_ptr(),
              ) };
              self.x.val.get(txn, node).as_view().post(&conn);
              self.mean.val.get_excl(txn, node).as_view().post(&conn);
              self.mean.grad.get(txn, node).as_view().post(&conn);
              self.var.grad.get(txn, node).as_view().post(&conn);
              self.x.grad.get_mut(txn, node).as_view().post(&conn);
            }
            _ => unimplemented!(),
          }
        }
      }
      true => {
        // Do nothing.
      }
    }
  }
}

//impl<Op, IdxOp> IndexExt<Op, IdxOp> for IndexOp<DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>>, IdxOp: 'static + ArrayOp<DeviceIoBatch<u32>> {
impl<Op, IdxOp> IndexExt<IdxOp, DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>>, IdxOp: 'static + ArrayOp<DeviceIoBatch<u32>> {
  fn index(&self, index_: Rc<IdxOp>) -> Rc<IndexOp<DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>>> {
    let x = self.data();
    let clk_horizon = x.horizon();
    IndexOp::new(self.clone(), index_.clone(), clk_horizon, {
      Rc::new(move |txn, node| {
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceIoBatch::zeros(batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for IndexOp<DeviceBatchArray1d<f32>, DeviceIoBatch<u32>, DeviceIoBatch<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.index_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.index_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_excl(txn, node).set_batch_size(batch_sz);
      // TODO: wait/post.
      let conn = DeviceStream::implicit().conn();
      unsafe { arraydiff_cuda_kernel_reduce_index_fwd_f32(
          x_dim,
          batch_sz,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.index.val.get(txn, node).as_ref().as_ptr(),
          self.y.val.get_excl(txn, node).as_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x_dim = self.x.val.get(txn, node).dim();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      // TODO: wait/post.
      let conn = DeviceStream::implicit().conn();
      unsafe { arraydiff_cuda_kernel_reduce_index_bwd_f32(
          x_dim,
          batch_sz,
          self.y.grad.get(txn, node).as_ref().as_ptr(),
          self.index.val.get(txn, node).as_ref().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
          conn.raw_stream().as_ptr(),
      ) };
    }
  }
}

//impl<Op> AutodiffSink<Op> for ArraySink<Op, DeviceMem<f32>> where Op: ArrayOp<DeviceMem<f32>> {
impl<Op> AutodiffSink for ArraySink<Op, DeviceMem<f32>> where Op: ArrayOp<DeviceMem<f32>> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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

impl<Op, Target> LstSqLossExt<Op, Target> for LstSqLoss<DeviceIoBatch<f32>, DeviceIoBatch<f32>> where Op: 'static + ArrayOp<DeviceIoBatch<f32>>, Target: 'static + ArrayOp<DeviceIoBatch<f32>> {
  fn lst_sq_loss(huber_clip: bool, x_: Rc<Op>, target_: Rc<Target>) -> Rc<Self> {
    let x = x_.data();
    let clk_horizon = x.horizon();
    LstSqLoss::new(huber_clip, x_.clone(), target_.clone(), clk_horizon, {
      Rc::new(move |txn, node| {
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceIoBatch::zeros(batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for LstSqLoss<DeviceIoBatch<f32>, DeviceIoBatch<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.target_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.target_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.loss.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.loss.val.overwrite(txn, node) {
      // TODO: wait/post.
      let conn = DeviceStream::implicit().conn();
      unsafe { arraydiff_cuda_kernel_lst_sq1_fwd_f32(
          batch_sz,
          self.x.val.get(txn, node).as_ref().as_ptr(),
          self.target.val.get(txn, node).as_ref().as_ptr(),
          self.loss.val.get_excl(txn, node).as_mut().as_mut_ptr(),
          match self.clip {
            false => 0,
            true  => 1,
          },
          conn.raw_stream().as_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      // TODO: wait/post.
      let conn = DeviceStream::implicit().conn();
      unsafe { arraydiff_cuda_kernel_lst_sq_bwd_f32(
          1,
          batch_sz,
          self.x.val.get(txn, node).as_ref().as_ptr(),
          self.target.val.get(txn, node).as_ref().as_ptr(),
          self.loss.grad.get(txn, node).as_ref().as_ptr(),
          self.x.grad.get_mut(txn, node).as_mut().as_mut_ptr(),
          match self.clip {
            false => 0,
            true  => 1,
          },
          conn.raw_stream().as_ptr(),
      ) };
    }
  }
}

impl<Op, Target> LstSqLossExt<Op, Target> for LstSqLoss<DeviceBatchArray1d<f32>, DeviceIoBatch<f32>> where Op: 'static + ArrayOp<DeviceBatchArray1d<f32>>, Target: 'static + ArrayOp<DeviceBatchArray1d<f32>> {
  fn lst_sq_loss(huber_clip: bool, x_: Rc<Op>, target_: Rc<Target>) -> Rc<Self> {
    let x = x_.data();
    let clk_horizon = x.horizon();
    LstSqLoss::new(huber_clip, x_.clone(), target_.clone(), clk_horizon, {
      Rc::new(move |txn, node| {
        let batch_cap = x.val.get(txn, node).batch_capacity();
        DeviceIoBatch::zeros(batch_cap, DeviceStream::implicit().conn())
      })
    })
  }
}

impl AutodiffOp for LstSqLoss<DeviceBatchArray1d<f32>, DeviceIoBatch<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      self.target_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.target_._pop(epoch, apply);
      self.x_._pop(epoch, apply);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.loss.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.loss.val.overwrite(txn, node) {
      // TODO
      unimplemented!();
    }
  }

  fn _backward(&self, txn: TxnId, gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.set_batch_size(batch_sz).as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
      // TODO
      unimplemented!();
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
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
      {
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
