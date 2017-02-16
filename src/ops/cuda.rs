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
  fn data(&self) -> ArrayData<DeviceArray1d<f32>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<DeviceArray1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
        if reader.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
          let mut val = self.data.val.get_mut(txn, node);
          let val_len = val.dim();
          let reader = reader.downcast_mut::<CursorBuf<Vec<f32>>>().unwrap();
          val.as_view_mut().load_sync(reader.read_buf(val_len).flatten(), DeviceStream::implicit().conn());
        } else if reader.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
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

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if writer.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorBuf<Vec<f32>>>().unwrap();
        val.as_view().store_sync(writer.write_buf(val_len).flatten_mut(), DeviceStream::implicit().conn());
      } else if writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view(), DeviceStream::implicit().conn());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.grad.var()) {
      if writer.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
        let mut grad = self.data.grad.get(txn, node);
        let grad_len = grad.dim();
        let writer = writer.downcast_mut::<CursorBuf<Vec<f32>>>().unwrap();
        grad.as_view().store_sync(writer.write_buf(grad_len).flatten_mut(), DeviceStream::implicit().conn());
      } else if writer.downcast_mut::<CursorBuf<DeviceMem<f32>>>().is_some() {
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

impl ArrayOp<DeviceArray2d<f32>> for ArraySrc<DeviceArray2d<f32>> {
  fn data(&self) -> ArrayData<DeviceArray2d<f32>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<DeviceArray2d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
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

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
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

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.grad.var()) {
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

impl ArrayOp<DeviceArray4d<f32>> for ArraySrc<DeviceArray4d<f32>> {
  fn data(&self) -> ArrayData<DeviceArray4d<f32>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<DeviceArray4d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
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

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
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

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.grad.var()) {
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

impl<F> AutodiffOp for InitializeOp<DeviceArray1d<f32>, Rc<F>> where F: Fn(Rc<RefCell<ChaChaRng>>, &mut DeviceArray1d<f32>) {
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
    if self.data.val.overwrite(txn, node) {
      (self.kernel)(seed_rng, &mut *self.data.val.get_excl(txn, node));
    }
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

impl<Op> MultiplyExt<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> for Rc<Op> where Op: 'static + ArrayOp<DeviceArray1d<f32>> {
  fn mult(&self, x_: Rc<ArrayOp<DeviceArray1d<f32>>>) -> Rc<LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, None, clk_horizon, Rc::new(|_, _| unsafe { DeviceMem::<f32>::alloc(1, DeviceStream::implicit().conn()) }))
  }

  fn mult_add(&self, x_: Rc<ArrayOp<DeviceArray1d<f32>>>, b_: Rc<ArrayOp<DeviceMem<f32>>>) -> Rc<LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, Some(b_), clk_horizon, Rc::new(|_, _| unsafe { DeviceMem::<f32>::alloc(1, DeviceStream::implicit().conn()) }))
  }
}

impl ArrayOp<DeviceMem<f32>> for LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> {
  fn data(&self) -> ArrayData<DeviceMem<f32>> {
    self.y.clone()
  }
}

impl AutodiffObjective for LinearOp<DeviceArray1d<f32>, DeviceArray1d<f32>, DeviceMem<f32>, DeviceMem<f32>> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if self.y.grad.accumulate(txn, node, |grad| grad.as_mut().set_constant(1.0, DeviceStream::implicit().conn())) {
    } else {
      // TODO
      //assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}

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
      self.a.grad.get_mut(txn, node).as_view_mut().add(1.0, self.x.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
      //self.a.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), self.x.val.get(txn, node).as_view(), DeviceStream::implicit().conn());
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())) {
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
