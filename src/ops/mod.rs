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

use prelude::*;
use ffi::*;

use densearray::prelude::*;
use rng::xorshift::*;

use rand::{Rng, SeedableRng};
use rand::chacha::{ChaChaRng};
use rand::distributions::{IndependentSample};
use rand::distributions::normal::{Normal};
use rand::distributions::range::{Range};
use std::any::{Any, /*TypeId*/};
use std::cell::{Cell, RefCell};
use std::cmp::{max};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

#[cfg(feature = "cuda")] pub mod cuda;

//const VEC_F32_TYPEID: TypeId = TypeId::of::<Vec<f32>>();

pub trait IoBuf {
  fn load(dst: &mut Self, offset: usize, reader: &mut Any) -> usize;
  fn store(src: &Self, offset: usize, writer: &mut Any) -> usize;
}

impl IoBuf for Array1d<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.dim();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_view_mut().copy(reader[offset .. offset + buf_len].flatten());
      offset += buf_len;
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.dim();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      writer[offset .. offset + buf_len].flatten_mut().copy(src.as_view());
      offset += buf_len;
    }
    offset
  }
}

impl IoBuf for Array2d<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.dim().flat_len();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_view_mut().flatten_mut().copy(reader[offset .. offset + buf_len].flatten());
      offset += buf_len;
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.dim().flat_len();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      writer[offset .. offset + buf_len].flatten_mut().copy(src.as_view().flatten());
      offset += buf_len;
    }
    offset
  }
}

impl IoBuf for Array4d<f32> {
  fn load(dst: &mut Self, mut offset: usize, reader: &mut Any) -> usize {
    let buf_len = dst.dim().flat_len();
    if reader.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if reader.downcast_mut::<Vec<f32>>().is_some() {
      let reader = reader.downcast_mut::<Vec<f32>>().unwrap();
      dst.as_view_mut().flatten_mut().copy(reader[offset .. offset + buf_len].flatten());
      offset += buf_len;
    }
    offset
  }

  fn store(src: &Self, mut offset: usize, writer: &mut Any) -> usize {
    let buf_len = src.dim().flat_len();
    if writer.downcast_mut::<NullIo>().is_some() {
      offset += buf_len;
    } else if writer.downcast_mut::<Vec<f32>>().is_some() {
      let writer = writer.downcast_mut::<Vec<f32>>().unwrap();
      writer[offset .. offset + buf_len].flatten_mut().copy(src.as_view().flatten());
      offset += buf_len;
    }
    offset
  }
}

pub fn src<A, F>(cons: F) -> Rc<ArraySrc<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  ArraySrc::new(1, false, Rc::new(cons))
}

pub fn sequential_src<A, F>(horizon: usize, cons: F) -> Rc<ArraySrc<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  ArraySrc::new(horizon, true, Rc::new(cons))
}

/*pub fn test_var() {
  let x: Rc<ArraySrc<Array1d<f32>>> = var(|_, _| Array1d::zeros(10));
}*/

pub struct CopyConstant<A> where A: Copy {
  /*node_id:  NodeId,
  stack:    OperatorStack,*/
  pub var:  TxnCopyVar<A>,
}

pub struct ArraySrc<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  data:     ArrayData<A>,
  clock:    bool,
}

impl<A> ArraySrc<A> {
  pub fn new(horizon: usize, clock: bool, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<Self> {
    let node = NodeId::new();
    Rc::new(ArraySrc{
      node_id:  node,
      stack:    OperatorStack::new(node, 0),
      data:     ArrayData::new(horizon, alloc),
      clock:    clock,
    })
  }
}

impl<A> ArrayOp<A> for ArraySrc<A> where ArraySrc<A>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}

impl AutodiffOp for ArraySrc<f32> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
        /*if reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
          let mut val = self.data.val.get_excl(txn, node);
          let val_len = val.dim();
          let reader = reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
          val.as_view_mut().copy(reader.read_buf(val_len).flatten());
        } else {
          unimplemented!();
        }*/
      }
      unimplemented!();
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view());
      } else {
        unimplemented!();
      }*/
      unimplemented!();
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let mut grad = self.data.grad.get(txn, node);
        let grad_len = grad.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view());
      } else {
        unimplemented!();
      }*/
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

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      *self.data.r_val.get_excl(txn, node) = 0.0;
    }
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

impl AutodiffOp for ArraySrc<Batch<u32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, reader: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
        /*if reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
          let mut val = self.data.val.get_excl(txn, node);
          let val_len = val.dim();
          let reader = reader.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
          val.as_view_mut().copy(reader.read_buf(val_len).flatten());
        } else {
          unimplemented!();
        }*/
      }
      unimplemented!();
    }
    offset
  }

  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.val.var()) {
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let mut val = self.data.val.get(txn, node);
        let val_len = val.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        writer.write_buf(val_len).flatten_mut().copy(val.as_view());
      } else {
        unimplemented!();
      }*/
      unimplemented!();
    }
    offset
  }

  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    let node = self._id();
    if vars.mask(self.data.grad.var()) {
      /*if writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().is_some() {
        let mut grad = self.data.grad.get(txn, node);
        let grad_len = grad.dim();
        let writer = writer.downcast_mut::<CursorIoBuf<Vec<f32>>>().unwrap();
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view());
      } else {
        unimplemented!();
      }*/
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

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      // TODO
      //*self.data.r_val.get_excl(txn, node) = 0.0;
    }
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

impl AutodiffOp for ArraySrc<Array1d<f32>> {
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

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      self.data.r_val.get_excl(txn, node).as_view_mut().set_constant(0.0);
    }
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

/*impl ArrayOp<Array2d<f32>> for ArraySrc<Array2d<f32>> {
  fn data(&self) -> ArrayData<Array2d<f32>> {
    self.data.clone()
  }
}*/

impl AutodiffOp for ArraySrc<Array2d<f32>> {
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

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      self.data.r_val.get_excl(txn, node).as_view_mut().flatten_mut().set_constant(0.0);
    }
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

/*impl ArrayOp<Array4d<f32>> for ArraySrc<Array4d<f32>> {
  fn data(&self) -> ArrayData<Array4d<f32>> {
    self.data.clone()
  }
}*/

impl AutodiffOp for ArraySrc<Array4d<f32>> {
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

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.data.r_val.overwrite(txn, node) {
      self.data.r_val.get_excl(txn, node).as_view_mut().flatten_mut().set_constant(0.0);
    }
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

pub fn pass<A, Op>(x_: Rc<Op>) -> Rc<PassOp<A>> where Op: 'static + ArrayOp<A> {
  let data = x_.data();
  PassOp::new(Some(AutodiffOp::from(x_)), data)
}

pub struct PassOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       RefCell<Option<Rc<AutodiffOp>>>,
  data:     ArrayData<A>,
}

impl<A> PassOp<A> {
  pub fn new(x_: Option<Rc<AutodiffOp>>, data: ArrayData<A>) -> Rc<Self> {
    let node = NodeId::new();
    Rc::new(PassOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       RefCell::new(x_),
      data:     data,
    })
  }
}

impl<A> ArrayOp<A> for PassOp<A> {
  default fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}

impl<A> AutodiffOp for PassOp<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._pop(epoch, apply);
    }
  }

  default fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    // Do nothing, `data` belongs to `x`.
  }

  default fn _forward(&self, txn: TxnId) {
  }

  default fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

pub fn no_pass<A, Op>(x_: Rc<Op>) -> Rc<NoPassOp<A>> where Op: 'static + ArrayOp<A> {
  let data = x_.data();
  NoPassOp::new(Some(AutodiffOp::from(x_)), data)
}

pub struct NoPassOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       RefCell<Option<Rc<AutodiffOp>>>,
  data:     ArrayData<A>,
}

impl<A> NoPassOp<A> {
  pub fn new(x_: Option<Rc<AutodiffOp>>, data: ArrayData<A>) -> Rc<Self>{
    let node = NodeId::new();
    Rc::new(NoPassOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       RefCell::new(x_),
      data:     data,
    })
  }
}

impl<A> ArrayOp<A> for NoPassOp<A> {
  default fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}

impl<A> AutodiffOp for NoPassOp<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      // Forward pass stops here.
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      // Backward pass stops here.
      apply(self);
    }
  }

  default fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    // Do nothing, `data` belongs to `x`.
  }

  default fn _forward(&self, txn: TxnId) {
  }

  default fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

pub struct IoOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  //x_:       Rc<ArrayOp<A>>,
  x_:       RefCell<Option<Rc<AutodiffOp>>>,
  data:     ArrayData<A>,
}

impl<A> IoOp<A> {
  pub fn new(x_: Option<Rc<AutodiffOp>>, data: ArrayData<A>) -> Rc<Self>{
    let node = NodeId::new();
    Rc::new(IoOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       RefCell::new(x_),
      data:     data,
    })
  }
}

impl<A> ArrayOp<A> for IoOp<A> {
  default fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}

impl<A> AutodiffOp for IoOp<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._pop(epoch, apply);
    }
  }

  default fn _load_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, reader: &mut Any) -> usize {
    unimplemented!();
  }

  default fn _store_val(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    unimplemented!();
  }

  default fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, offset: usize, writer: &mut Any) -> usize {
    unimplemented!();
  }

  default fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    // TODO: `persist` semantics?
  }

  default fn _forward(&self, txn: TxnId) {
  }

  default fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

pub struct InitializeOp<A, Init> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<ArrayOp<A>>,
  data:     ArrayData<A>,
  kernel:   Init,
}

pub fn init_val<R, A, F>(f: F) -> impl Fn(TxnId, NodeId, Rc<RefCell<R>>, ArrayData<A>) where R: Rng, F: Fn(Rc<RefCell<R>>, &mut A) {
  let init_f = Rc::new(f);
  move |txn: TxnId, node: NodeId, rng: Rc<RefCell<R>>, data: ArrayData<A>| {
    if data.val.overwrite(txn, node) {
      (init_f)(rng, &mut *data.val.get_excl(txn, node));
    }
  }
}

pub fn xavier_linear_init<R>() -> impl Fn(Rc<RefCell<R>>, &mut Array2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let half_range = (6.0 / (a.dim().0 + a.dim().1) as f64).sqrt();
    let dist = Range::new(-half_range, half_range);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub fn xavier_conv2d_init<R>(axes: Axes<(usize, usize)>) -> impl Fn(Rc<RefCell<R>>, &mut Array4d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array4d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let half_range = match axes {
      Axes((0, 1)) => (6.0 / (a.dim().0 * a.dim().1 * a.dim().2 + a.dim().3) as f64).sqrt(),
      _ => unimplemented!(),
    };
    let dist = Range::new(-half_range, half_range);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub fn kaiming_linear_init<R>() -> impl Fn(Rc<RefCell<R>>, &mut Array2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let std = (2.0 / a.dim().0 as f64).sqrt();
    let dist = Normal::new(0.0, std);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub fn kaiming_conv2d_init<R>(axes: Axes<(usize, usize)>) -> impl Fn(Rc<RefCell<R>>, &mut Array4d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array4d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed(&mut *seed_rng);
    let std = match axes {
      Axes((0, 1)) => (2.0 / (a.dim().0 * a.dim().1 * a.dim().2) as f64).sqrt(),
      _ => unimplemented!(),
    };
    let dist = Normal::new(0.0, std);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub trait InitializeExt<A, F, Init> {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, Init>>;
}

/*impl<Op, A, F> InitializeExt<A, F, Rc<F>> for Rc<Op> where Op: 'static + ArrayOp<A>, F: Fn(Rc<RefCell<ChaChaRng>>, &mut A) {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, Rc<F>>> {
    let node = NodeId::new();
    let stack = OperatorStack::new(node, 1);
    Rc::new(InitializeOp{
      node_id:  node,
      stack:    stack,
      x_:   self.clone(),
      data: self.data(),
      kernel:   Rc::new(f),
    })
  }
}*/

//impl<Op, A, F> InitializeExt<A, F, Rc<F>> for Rc<Op> where Op: 'static + ArrayOp<A>, F: Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<A>) {
impl<Op, A, F> InitializeExt<A, F, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<A>)>> for Rc<Op> where Op: 'static + ArrayOp<A>, F: 'static + Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<A>) {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<A>)>>> {
    let node = NodeId::new();
    let stack = OperatorStack::new(node, 1);
    let init: Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<A>)> = Rc::new(f);
    Rc::new(InitializeOp{
      node_id:  node,
      stack:    stack,
      x_:   self.clone(),
      data: self.data(),
      kernel:   init,
    })
  }
}

impl<A, Init> ArrayOp<A> for InitializeOp<A, Init> where InitializeOp<A, Init>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}

/*impl ArrayOp<f32> for InitializeOp<f32, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<f32>)>> {
  fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}*/

//impl<F> AutodiffOp for InitializeOp<f32, Rc<F>> where F: Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<f32>) {
impl AutodiffOp for InitializeOp<f32, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<f32>)>> {
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

//impl<S, F> AutodiffOp for InitializeOp<Array1d<f32, S>, Rc<F>> where S: DerefMut<Target=[f32]>, F: Fn(Rc<RefCell<ChaChaRng>>, &mut Array1d<f32, S>) {
//impl<S, F> AutodiffOp for InitializeOp<Array1d<f32, S>, Rc<F>> where S: DerefMut<Target=[f32]>, F: Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<Array1d<f32, S>>) {
impl<S> AutodiffOp for InitializeOp<Array1d<f32, S>, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, ArrayData<Array1d<f32, S>>)>> where S: DerefMut<Target=[f32]> {
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

pub struct BranchOp<Cond, On, Off, Data> {
  node_id:  NodeId,
  stack:    OperatorStack,
  cond:     Cond,
  on_:      On,
  off_:     Off,
  on:       Data,
  off:      Data,
  output:   Data,
}

impl<Cond, On, Off, Data> OutputOp for BranchOp<Cond, On, Off, Data> where BranchOp<Cond, On, Off, Data>: AutodiffOp, Data: OutputData {
  type Data = Data;

  fn _own_data(&self) -> &Data {
    &self.output
  }
}

pub struct ExpMapKernel;
pub struct RectMapKernel;
pub struct LeakRectMapKernel<T>{c: T}
pub struct LogisticMapKernel;
pub struct TanhMapKernel;

pub trait SpecialMapKernel {}

impl SpecialMapKernel for ExpMapKernel {}
impl SpecialMapKernel for RectMapKernel {}
impl<T> SpecialMapKernel for LeakRectMapKernel<T> {}
impl SpecialMapKernel for TanhMapKernel {}
impl SpecialMapKernel for LogisticMapKernel {}

pub trait SpecialMapExt</*T,*/ A> {
  //fn exp(&self) -> Rc<MapOp<A, ExpMapKernel>>;
  fn rect(&self) -> Rc<MapOp<A, RectMapKernel>>;
  //fn leak_rect(&self, c: T) -> Rc<MapOp<A, LeakRectMapKernel<T>>>;
  //fn logistic(&self) -> Rc<MapOp<A, LogisticMapKernel>>;
  //fn tanh(&self) -> Rc<MapOp<A, TanhMapKernel>>;
}

impl<Op, S> SpecialMapExt</*f32,*/ Array1d<f32, S>> for Rc<Op> where Op: 'static + ArrayOp<Array1d<f32, S>>, S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn rect(&self) -> Rc<MapOp<Array1d<f32, S>, RectMapKernel>> {
    let clk_horizon = self.data().horizon();
    MapOp::new(RectMapKernel, self.clone(), clk_horizon, {
      let x = self.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }
}

pub struct MapOp<A, MapF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<ArrayOp<A>>,
  x:    ArrayData<A>,
  y:    ArrayData<A>,
  kernel:   MapF,
}

impl<A, MapF> MapOp<A, MapF> {
  pub fn new<F>(kernel: MapF, x_: Rc<ArrayOp<A>>, clk_horizon: usize, alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> A {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(MapOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc),
      kernel:   kernel,
    })
  }
}

impl<A, MapF> ArrayOp<A> for MapOp<A, MapF> where MapOp<A, MapF>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<A> {
    &self.y
  }
}

impl<S, MapF> AutodiffOp for MapOp<Array1d<f32, S>, MapF> where S: DerefMut<Target=[f32]>, MapF: SpecialMapKernel {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x_._pop(epoch, apply);
    }
  }

  default fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  default fn _forward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOp for MapOp<Array1d<f32, S>, RectMapKernel> where S: DerefMut<Target=[f32]> {
  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_fwd_f32(
          x_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      let x_dim = self.x.r_val.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_bwd_f32(
          x_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.x.r_val.get(txn, node).as_view().as_ptr(),
          self.y.r_val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.r_grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.r_grad.get(txn, node).as_view().as_ptr(),
          self.x.r_grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward2(&self, txn: TxnId) {
    let node = self._id();
    if self.x.grad2.accumulate(txn, node, |grad2| grad2.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad2.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad2.get(txn, node).as_view().as_ptr(),
          self.x.grad2.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }
}

impl<S> AutodiffOp for MapOp<Array1d<f32, S>, LogisticMapKernel> where S: DerefMut<Target=[f32]> {
  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      unsafe { arraydiff_kernel_logistic_fwd_f32(
          x_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_logistic_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOp for MapOp<Array1d<f32, S>, TanhMapKernel> where S: DerefMut<Target=[f32]> {
  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let x_dim = self.x.val.get(txn, node).dim();
      unsafe { arraydiff_kernel_tanh_fwd_f32(
          x_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_excl(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_tanh_bwd_f32(
          y_dim,
          self.x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          self.x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

pub struct TransformOp<A, B, Transform> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<ArrayOp<A>>,
  x:    ArrayData<A>,
  y:    ArrayData<B>,
  kernel:   Transform,
}

pub struct CastTransform;
pub struct FlattenTransform;
pub struct ReifyTransform<Idx> {
  pub dim:  Idx,
}

pub trait CastExt<A, B> {
  fn cast(&self) -> Rc<TransformOp<A, B, CastTransform>>;
}

pub trait FlattenExt<A, B> {
  fn flatten(&self) -> Rc<TransformOp<A, B, FlattenTransform>>;
}

pub trait ReifyExt<Idx, A, B> {
  fn reify(&self, dim: Idx) -> Rc<TransformOp<A, B, ReifyTransform<Idx>>>;
}

impl<A, B, Transform> TransformOp<A, B, Transform> {
  pub fn new<F>(x_: Rc<ArrayOp<A>>, transform: Transform, clk_horizon: usize, alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> B {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(TransformOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc),
      kernel:   transform,
    })
  }
}

impl<S> FlattenExt<Array3d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array3d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn flatten(&self) -> Rc<TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), FlattenTransform, clk_horizon, {
      let x = self.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim.flat_len());
        Array1d::from_storage(dim.flat_len(), buf)
      })
    })
  }
}

impl<S> AutodiffOp for TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform> where S: DerefMut<Target=[f32]> {
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
      self.y.val.get_excl(txn, node).as_view_mut().copy(self.x.val.get(txn, node).as_view().flatten());
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.y.grad.get(txn, node).as_view());
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
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
  }
}

pub struct JoinOp<A, JoinF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  xs_:  Vec<Rc<ArrayOp<A>>>,
  xs:   Vec<ArrayData<A>>,
  y:    ArrayData<A>,
  kernel:   JoinF,
}

impl<A, JoinF> JoinOp<A, JoinF> {
  pub fn new<F>(xs_: Vec<Rc<ArrayOp<A>>>, kernel: JoinF, clk_horizon: usize, alloc: Rc<F>) -> Rc<JoinOp<A, JoinF>> where F: 'static + Fn(TxnId, NodeId) -> A {
    let node = NodeId::new();
    let in_degree = xs_.len();
    let mut xs = Vec::with_capacity(in_degree);
    for x_ in xs_.iter() {
      let x = x_.data();
      xs.push(x);
    }
    assert_eq!(in_degree, xs_.len());
    assert_eq!(in_degree, xs.len());
    Rc::new(JoinOp{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      xs_:      xs_,
      xs:       xs,
      y:        ArrayData::new(clk_horizon, alloc),
      kernel:   kernel,
    })
  }
}

impl<A, JoinF> ArrayOp<A> for JoinOp<A, JoinF> where JoinOp<A, JoinF>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<A> {
    &self.y
  }
}

pub struct AxisJoinKernel;
pub struct SumJoinKernel;

pub trait AxisJoinExt<Op, A> where Op: ArrayOp<A> {
  fn axis_join(xs: Vec<Rc<Op>>) -> Rc<JoinOp<A, AxisJoinKernel>>;
}

pub fn axis_join<Op, A>(xs: Vec<Rc<Op>>) -> Rc<JoinOp<A, AxisJoinKernel>> where Rc<Op>: AxisJoinExt<Op, A>, Op: ArrayOp<A> {
  <Rc<Op> as AxisJoinExt<Op, A>>::axis_join(xs)
}

pub trait AddExt<A> {
  fn add<RhsOp>(&self, x_: Rc<RhsOp>) -> Rc<JoinOp<A, SumJoinKernel>> where RhsOp: 'static + ArrayOp<A>;
}

impl<Op, A> AddExt<A> for Rc<Op> where Rc<JoinOp<A, SumJoinKernel>>: SumExt<A>, Op: 'static + ArrayOp<A> {
  default fn add<RhsOp>(&self, x_: Rc<RhsOp>) -> Rc<JoinOp<A, SumJoinKernel>> where RhsOp: 'static + ArrayOp<A> {
    <Rc<JoinOp<A, SumJoinKernel>> as SumExt<A>>::sum(vec![ArrayOp::from(self.clone()), ArrayOp::from(x_)])
  }
}

pub trait SumExt<A> {
  fn sum(xs_: Vec<Rc<ArrayOp<A>>>) -> Rc<JoinOp<A, SumJoinKernel>>;
}

pub fn sum<A>(xs_: Vec<Rc<ArrayOp<A>>>) -> Rc<JoinOp<A, SumJoinKernel>> where Rc<JoinOp<A, SumJoinKernel>>: SumExt<A> {
  <Rc<JoinOp<A, SumJoinKernel>> as SumExt<A>>::sum(xs_)
}

impl<Op, S> AxisJoinExt<Op, Array1d<f32, S>> for Rc<Op> where Op: ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn axis_join(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<Array1d<f32, S>, AxisJoinKernel>> {
    unimplemented!();
  }
}

/*impl<Op, A> SumExt<A> for Rc<Op> where Op: ArrayOp<A> {
  default fn sum(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<A, SumJoinKernel>> {
    unimplemented!();
  }

  default fn add(&self, x_: Rc<Op>) -> Rc<JoinOp<A, SumJoinKernel>> {
    Self::sum(vec![self.clone(), x_])
  }
}*/

/*impl<Op, S> SumExt<Array1d<f32, S>> for Rc<Op> where Op: ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn sum(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> where S: DerefMut<Target=[f32]> {
    unimplemented!();
  }

  fn add(&self, x_: Rc<Op>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> {
    Self::sum(vec![self.clone(), x_])
  }
}*/

impl<S> AutodiffOp for JoinOp<Array1d<f32, S>, SumJoinKernel> where S: DerefMut<Target=[f32]> {
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
      self.y.val.get_excl(txn, node).as_view_mut().copy(self.xs[0].val.get(txn, node).as_view());
      for x in self.xs.iter().skip(1) {
        self.y.val.get_excl(txn, node).as_view_mut().add(1.0, x.val.get(txn, node).as_view());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    for x in self.xs.iter() {
      if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
        x.grad.get_mut(txn, node).as_view_mut().add(1.0, self.y.grad.get(txn, node).as_view());
      }
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      self.y.r_val.get_excl(txn, node).as_view_mut().copy(self.xs[0].r_val.get(txn, node).as_view());
      for x in self.xs.iter().skip(1) {
        self.y.r_val.get_excl(txn, node).as_view_mut().add(1.0, x.r_val.get(txn, node).as_view());
      }
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    for x in self.xs.iter() {
      if x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
        x.r_grad.get_mut(txn, node).as_view_mut().add(1.0, self.y.r_grad.get(txn, node).as_view());
      }
    }
  }
}

pub struct SplitOp<A, SplitF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<ArrayOp<A>>,
  y:    ArrayData<A>,
  kernel:   SplitF,
}

pub struct AxisSplitKernel {
  op_idx:   usize,
  axis:     usize,
  offset:   usize,
  length:   usize,
}

pub trait AxisSplitExt<A> {
  fn axis_split(&self, axis: usize, parts: Vec<usize>) -> Vec<Rc<SplitOp<A, AxisSplitKernel>>>;
}

pub trait DummyExt<A> {
}

impl<S> DummyExt<Array1d<f32, S>> for Rc<ArrayOp<Array1d<f32, S>>> where S: DerefMut<Target=[f32]> {
}

pub trait MultExt<A, V, W, B> {
  fn mult(&self, x: Rc<ArrayOp<V>>) -> Rc<LinearOp<A, V, W, B>>;
  fn mult_add(&self, x: Rc<ArrayOp<V>>, b: Rc<ArrayOp<B>>) -> Rc<LinearOp<A, V, W, B>>;
}

pub struct LinearOp<A, V, W, B> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a_:   Rc<ArrayOp<A>>,
  x_:   Rc<ArrayOp<V>>,
  b_:   Option<Rc<ArrayOp<B>>>,
  a:    ArrayData<A>,
  x:    ArrayData<V>,
  b:    Option<ArrayData<B>>,
  y:    ArrayData<W>,
  tmp:  ArrayData<W>,
}

impl<A, V, W, B> LinearOp<A, V, W, B> {
  pub fn new<F>(a_: Rc<ArrayOp<A>>, x_: Rc<ArrayOp<V>>, b_: Option<Rc<ArrayOp<B>>>, clk_horizon: usize, alloc: Rc<F>) -> Rc<LinearOp<A, V, W, B>> where F: 'static + Fn(TxnId, NodeId) -> W {
    let node = NodeId::new();
    let in_degree = match b_ {
      None    => 2,
      Some(_) => 3,
    };
    let a = a_.data();
    let x = x_.data();
    let b = b_.as_ref().map(|b_| b_.data());
    Rc::new(LinearOp{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      a_:   a_,
      x_:   x_,
      b_:   b_,
      a:    a,
      x:    x,
      b:    b,
      y:    ArrayData::new(clk_horizon, alloc.clone()),
      tmp:  ArrayData::new(1, alloc),
    })
  }
}

impl<A, V, W, B> ArrayOp<W> for LinearOp<A, V, W, B> where LinearOp<A, V, W, B>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<W> {
    &self.y
  }
}

impl<Op, S> MultExt<Array1d<f32, S>, Array1d<f32, S>, f32, f32> for Rc<Op> where Op: 'static + ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn mult(&self, x_: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, None, clk_horizon, Rc::new(|_, _| 0.0_f32))
  }

  fn mult_add(&self, x_: Rc<ArrayOp<Array1d<f32, S>>>, b_: Rc<ArrayOp<f32>>) -> Rc<LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, Some(b_), clk_horizon, Rc::new(|_, _| 0.0_f32))
  }
}

/*impl<S> AutodiffObjective for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
    } else {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}*/

impl<S> AutodiffOp for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      *self.y.val.get_excl(txn, node) = self.a.val.get(txn, node).as_view().inner_prod(1.0, self.x.val.get(txn, node).as_view());
      if let Some(ref b) = self.b {
        *self.y.val.get_excl(txn, node) += *b.val.get(txn, node);
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      //println!("DEBUG: LinearOp: backward: a.grad: before: {:?}", &self.a.grad.get_mut(txn, node).as_slice()[ .. 5]);
      self.a.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), self.x.val.get(txn, node).as_view());
      //println!("DEBUG: LinearOp: backward: a.grad: after:  {:?}", &self.a.grad.get_mut(txn, node).as_slice()[ .. 5]);
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      //println!("DEBUG: LinearOp: backward: x.grad: before: {:?}", &self.x.grad.get_mut(txn, node).as_slice()[ .. 5]);
      self.x.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), self.a.val.get(txn, node).as_view());
      //println!("DEBUG: LinearOp: backward: x.grad: after:  {:?}", &self.x.grad.get_mut(txn, node).as_slice()[ .. 5]);
    }
    if let Some(ref b) = self.b {
      if b.grad.accumulate(txn, node, |g| *g = 0.0) {
        *b.grad.get_mut(txn, node) += *self.y.grad.get(txn, node);
      }
    }
  }
}

impl<S> MultExt<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array2d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn mult(&self, x_: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), None, clk_horizon, {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }

  fn mult_add(&self, x_: Rc<ArrayOp<Array1d<f32, S>>>, b_: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), Some(b_), clk_horizon, {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }
}

/*impl<S> ArrayOp<Array1d<f32, S>> for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}*/

impl<S> AutodiffOp for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      self.y.val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          self.a.val.get(txn, node).as_view(), Transpose::N,
          self.x.val.get(txn, node).as_view(),
          0.0,
      );
      if let Some(ref b) = self.b {
        self.y.val.get_mut(txn, node).as_view_mut().add(1.0, b.val.get(txn, node).as_view());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let x_dim = self.x.val.get(txn, node).dim();
      let y_dim = self.y.val.get(txn, node).dim();
      self.a.grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn, node).as_view().reshape((y_dim, 1)), Transpose::N,
          self.x.val.get(txn, node).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      self.x.grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          self.a.val.get(txn, node).as_view(), Transpose::T,
          self.y.grad.get(txn, node).as_view(),
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      if b.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
        b.grad.get_mut(txn, node).as_view_mut().copy(self.y.grad.get(txn, node).as_view());
      }
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.overwrite(txn, node) {
      self.y.r_val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          self.a.r_val.get(txn, node).as_view(), Transpose::N,
          self.x.val.get(txn, node).as_view(),
          0.0,
      );
      self.y.r_val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          self.a.val.get(txn, node).as_view(), Transpose::N,
          self.x.r_val.get(txn, node).as_view(),
          1.0,
      );
      if let Some(ref b) = self.b {
        self.y.r_val.get_mut(txn, node).as_view_mut().add(1.0, b.r_val.get(txn, node).as_view());
      }
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    if self.a.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      let x_dim = self.x.val.get(txn, node).dim();
      let y_dim = self.y.grad.get(txn, node).dim();
      self.a.r_grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.r_grad.get(txn, node).as_view().reshape((y_dim, 1)), Transpose::N,
          self.x.val.get(txn, node).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
      self.a.r_grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn, node).as_view().reshape((y_dim, 1)), Transpose::N,
          self.x.r_val.get(txn, node).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
    }
    if self.x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      self.x.r_grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          self.a.r_val.get(txn, node).as_view(), Transpose::T,
          self.y.grad.get(txn, node).as_view(),
          1.0,
      );
      self.x.r_grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          self.a.val.get(txn, node).as_view(), Transpose::T,
          self.y.r_grad.get(txn, node).as_view(),
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      if b.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
        b.r_grad.get_mut(txn, node).as_view_mut().add(1.0, self.y.r_grad.get(txn, node).as_view());
      }
    }
  }
}

impl<S> MultExt<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array2d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + BatchArrayStorage<usize> {
  fn mult(&self, x_: Rc<ArrayOp<BatchArray1d<f32, S>>>) -> Rc<LinearOp<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), None, clk_horizon, {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_sz = x.val.get(txn, node).batch_size();
        let buf = <S as BatchArrayStorage<usize>>::alloc(dim, batch_sz);
        BatchArray1d::from_storage(dim, batch_sz, buf)
      })
    })
  }

  fn mult_add(&self, x_: Rc<ArrayOp<BatchArray1d<f32, S>>>, b_: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), Some(b_), clk_horizon, {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_sz = x.val.get(txn, node).batch_size();
        let buf = <S as BatchArrayStorage<usize>>::alloc(dim, batch_sz);
        BatchArray1d::from_storage(dim, batch_sz, buf)
      })
    })
  }
}

/*impl<S> ArrayOp<BatchArray1d<f32, S>> for LinearOp<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _data(&self) -> &ArrayData<A> {
    &self.data
  }
}*/

impl<S> AutodiffOp for LinearOp<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      self.y.val.get_mut(txn, node).set_batch_size(batch_sz);
      self.y.val.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.a.val.get(txn, node).as_view(), Transpose::N,
          self.x.val.get(txn, node).as_view(), Transpose::N,
          0.0,
      );
      if let Some(ref b) = self.b {
        unimplemented!();
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      self.a.grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn, node).as_view(), Transpose::N,
          self.x.val.get(txn, node).as_view(), Transpose::T,
          1.0,
      );
    }
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let batch_sz = self.y.grad.get(txn, node).batch_size();
      self.x.grad.get_mut(txn, node).set_batch_size(batch_sz);
      self.x.grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.a.val.get(txn, node).as_view(), Transpose::T,
          self.y.grad.get(txn, node).as_view(), Transpose::N,
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      unimplemented!();
    }
  }

  fn _backward2(&self, txn: TxnId) {
    let node = self._id();
    if self.a.grad2.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let tmp_txn = TxnId::new();
      assert!(self.tmp.val.overwrite(tmp_txn, node));
      self.tmp.val.get_excl(tmp_txn, node).as_view_mut().flatten_mut().copy(self.a.grad.get(txn, node).as_view().flatten());
      self.tmp.val.get_excl(tmp_txn, node).as_view_mut().flatten_mut().square();
      self.tmp.val.get_excl(tmp_txn, node).as_view_mut().flatten_mut().elem_mult(self.x.grad2.get(txn, node).as_view().flatten());
      self.a.grad2.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.tmp.val.get(tmp_txn, node).as_view().flatten());
    }
    if self.x.grad2.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let tmp_txn = TxnId::new();
      assert!(self.tmp.val.overwrite(tmp_txn, node));
      self.tmp.val.get_excl(tmp_txn, node).as_view_mut().flatten_mut().copy(self.a.val.get(txn, node).as_view().flatten());
      self.tmp.val.get_excl(tmp_txn, node).as_view_mut().flatten_mut().square();
      let batch_sz = self.y.grad2.get(txn, node).batch_size();
      self.x.grad2.get_mut(txn, node).set_batch_size(batch_sz);
      self.x.grad2.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.tmp.val.get(tmp_txn, node).as_view(), Transpose::T,
          self.y.grad2.get(txn, node).as_view(), Transpose::N,
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      unimplemented!();
    }
  }
}

pub struct ElemLinearOp<A, V, K> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a_:   Rc<ArrayOp<A>>,
  x_:   Rc<ArrayOp<V>>,
  b_:   Option<Rc<ArrayOp<A>>>,
  a:    ArrayData<A>,
  x:    ArrayData<V>,
  b:    Option<ArrayData<A>>,
  y:    ArrayData<V>,
  //tmp:  ArrayData<V>,
  kernel:   K,
}

impl<A, V, Kernel> ElemLinearOp<A, V, Kernel> {
  pub fn new(a_: Rc<ArrayOp<A>>, x_: Rc<ArrayOp<V>>, b_: Option<Rc<ArrayOp<A>>>, kernel: Kernel, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
    let node = NodeId::new();
    let in_degree = match b_ {
      None    => 2,
      Some(_) => 3,
    };
    let a = a_.data();
    let x = x_.data();
    let b = b_.as_ref().map(|b_| b_.data());
    Rc::new(ElemLinearOp{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      a_:   a_,
      x_:   x_,
      b_:   b_,
      a:    a,
      x:    x,
      b:    b,
      y:    ArrayData::new(clk_horizon, alloc.clone()),
      //tmp:  ArrayData::new(clk_horizon, alloc.clone()),
      kernel:   kernel,
    })
  }
}

pub struct BroadcastMultAddKernel;
pub struct ElemNormalizeKernel{pub epsilon: f64}

pub trait ElemMultExt<A, V> {
  fn elem_mult(&self, x_: Rc<ArrayOp<V>>) -> Rc<ElemLinearOp<A, V, BroadcastMultAddKernel>>;
  fn elem_mult_add(&self, x_: Rc<ArrayOp<V>>, b_: Rc<ArrayOp<A>>) -> Rc<ElemLinearOp<A, V, BroadcastMultAddKernel>>;
}

/*pub trait ElemNormalizeExt<A, V> {
  fn elem_normalize(&self, mean_: Rc<ArrayOp<A>>, var_: Rc<ArrayOp<A>>) -> Rc<ElemLinearOp<A, V, ElemNormalizeKernel>>;
}*/

impl<A, V, Kernel> ArrayOp<V> for ElemLinearOp<A, V, Kernel> where ElemLinearOp<A, V, Kernel>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<V> {
    &self.y
  }
}

impl<S> AutodiffOp for ElemLinearOp<f32, BatchArray3d<f32, S>, BroadcastMultAddKernel> where S: DerefMut<Target=[f32]> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    // TODO
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      if let Some(ref b) = self.b {
      }
    }
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
    unimplemented!();
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

impl<S> AutodiffOp for ElemLinearOp<Array1d<f32, S>, BatchArray3d<f32, S>, ElemNormalizeKernel> where S: DerefMut<Target=[f32]> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    // TODO
    let node = self._id();
    if self.y.val.overwrite(txn, node) {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      if let Some(ref b) = self.b {
      }
    }
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    // TODO
    unimplemented!();
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

pub struct ElemNormalizeOp<Idx, A, V> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  axes:     Idx::Axes,
  epsilon:  f64,
  x_:       Rc<ArrayOp<V>>,
  mean_:    Rc<ArrayOp<A>>,
  var_:     Rc<ArrayOp<A>>,
  x:        ArrayData<V>,
  mean:     ArrayData<A>,
  var:      ArrayData<A>,
  y:        ArrayData<V>,
}

impl<Idx, A, V> ElemNormalizeOp<Idx, A, V> where Idx: ArrayIndex {
  pub fn new(axes: Idx::Axes, epsilon: f64, x_: Rc<ArrayOp<V>>, mean_: Rc<ArrayOp<A>>, var_: Rc<ArrayOp<A>>, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
    let node = NodeId::new();
    let x = x_.data();
    let mean = mean_.data();
    let var = var_.data();
    Rc::new(ElemNormalizeOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 3),
      axes:     axes,
      epsilon:  epsilon,
      x_:       x_,
      mean_:    mean_,
      var_:     var_,
      x:        x,
      mean:     mean,
      var:      var,
      y:        ArrayData::new(clk_horizon, alloc.clone()),
    })
  }
}

pub trait ElemNormalizeExt<Idx, A, V> where Idx: ArrayIndex {
  fn elem_normalize(&self, axes: Idx::Axes, epsilon: f64, mean_: Rc<ArrayOp<A>>, var_: Rc<ArrayOp<A>>) -> Rc<ElemNormalizeOp<Idx, A, V>>;
}

impl<Idx, A, V> ArrayOp<V> for ElemNormalizeOp<Idx, A, V> where ElemNormalizeOp<Idx, A, V>: AutodiffOp, Idx: ArrayIndex {
  default fn _data(&self) -> &ArrayData<V> {
    &self.y
  }
}

#[derive(Clone, Copy)]
pub struct ConvShape<Idx> where Idx: ArrayIndex {
  pub axes:     Idx::Axes,
  pub kernel:   Idx,
  pub stride:   Idx,
  //pub zero_pad: Idx,
  pub zero_pad: bool,
  pub filters:  Option<usize>,
}

impl ConvShape<(usize, usize)> {
  pub fn conv2d_pad_dim(&self, in_dim: (usize, usize, usize)) -> (usize, usize) {
    match self.zero_pad {
      false => (0, 0),
      true  => {
        match self.axes {
          // FIXME(20170309): this calculation ignores input and stride dimensions.
          Axes((0, 1)) => (self.kernel.0 / 2, self.kernel.1 / 2),
          _ => unimplemented!(),
        }
      }
    }
  }

  pub fn conv2d_kernel_dim(&self, in_dim: (usize, usize, usize)) -> (usize, usize, usize, usize) {
    match self.axes {
      Axes((0, 1)) => (self.kernel.0, self.kernel.1, in_dim.2, self.filters.unwrap_or(in_dim.2)),
      _ => unimplemented!(),
    }
  }

  pub fn conv2d_output_dim(&self, in_dim: (usize, usize, usize)) -> (usize, usize, usize) {
    match self.axes {
      Axes((0, 1)) => {
        let (in_w, in_h, in_chan) = in_dim;
        let (kernel_w, kernel_h) = self.kernel;
        let (stride_w, stride_h) = self.stride;
        let (pad_w, pad_h) = self.conv2d_pad_dim(in_dim);
        let out_w = max(0, (in_w + 2 * pad_w - kernel_w + stride_w) as isize) as usize / stride_w;
        let out_h = max(0, (in_h + 2 * pad_h - kernel_h + stride_h) as isize) as usize / stride_h;
        assert!(out_w > 0);
        assert!(out_h > 0);
        (out_w, out_h, self.filters.unwrap_or(in_dim.2))
      }
      _ => unimplemented!(),
    }
  }
}

pub trait ConvExt<Idx, A, B, V, Backend> where Idx: ArrayIndex {
  fn conv(&self, shape: ConvShape<Idx>, x_: Rc<ArrayOp<V>>) -> Rc<ConvOp<Idx, A, B, V, Backend>>;
  fn conv_add(&self, shape: ConvShape<Idx>, x_: Rc<ArrayOp<V>>, b_: Rc<ArrayOp<B>>) -> Rc<ConvOp<Idx, A, B, V, Backend>>;
}

pub struct ConvOp<Idx, A, B, V, Backend> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    ConvShape<Idx>,
  a_:   Rc<ArrayOp<A>>,
  x_:   Rc<ArrayOp<V>>,
  b_:   Option<Rc<ArrayOp<B>>>,
  a:    ArrayData<A>,
  x:    ArrayData<V>,
  b:    Option<ArrayData<B>>,
  y:    ArrayData<V>,
  backend:  Backend,
}

impl<Idx, A, B, V, Backend> ConvOp<Idx, A, B, V, Backend> where Idx: ArrayIndex {
  pub fn new(shape: ConvShape<Idx>, a_: Rc<ArrayOp<A>>, x_: Rc<ArrayOp<V>>, b_: Option<Rc<ArrayOp<B>>>, backend: Backend, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
    let node = NodeId::new();
    let in_degree = match b_ {
      None    => 2,
      Some(_) => 3,
    };
    let a = a_.data();
    let x = x_.data();
    let b = b_.as_ref().map(|b_| b_.data());
    Rc::new(ConvOp{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      shape:    shape,
      a_:   a_,
      x_:   x_,
      b_:   b_,
      a:    a,
      x:    x,
      b:    b,
      y:    ArrayData::new(clk_horizon, alloc.clone()),
      //tmp:  ArrayData::new(1, alloc),
      backend:  backend,
    })
  }
}

impl<S> AutodiffOp for ConvOp<(usize, usize), Array4d<f32, S>, Array1d<f32, S>, Array3d<f32, S>, ()> where S: DerefMut<Target=[f32]> {
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

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.y.rollover_all(txn, vars);
  }

  fn _forward(&self, txn: TxnId) {
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
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
pub struct PoolShape<Idx> where Idx: ArrayIndex {
  pub axes:     Idx::Axes,
  pub kernel:   Idx,
  pub stride:   Idx,
  pub zero_pad: bool,
}

impl PoolShape<(usize, usize)> {
  pub fn pool2d_pad_dim(&self, in_dim: (usize, usize, usize)) -> (usize, usize) {
    match self.zero_pad {
      false => (0, 0),
      true  => {
        match self.axes {
          // FIXME(20170309): this calculation ignores input and stride dimensions.
          Axes((0, 1)) => (self.kernel.0 / 2, self.kernel.1 / 2),
          _ => unimplemented!(),
        }
      }
    }
  }

  pub fn pool2d_output_dim(&self, in_dim: (usize, usize, usize)) -> (usize, usize, usize) {
    match self.axes {
      Axes((0, 1)) => {
        let (in_w, in_h, _) = in_dim;
        let (kernel_w, kernel_h) = self.kernel;
        let (stride_w, stride_h) = self.stride;
        let (pad_w, pad_h) = self.pool2d_pad_dim(in_dim);
        let out_w = max(0, (in_w + 2 * pad_w - kernel_w + stride_w) as isize) as usize / stride_w;
        let out_h = max(0, (in_h + 2 * pad_h - kernel_h + stride_h) as isize) as usize / stride_h;
        assert!(out_w > 0);
        assert!(out_h > 0);
        (out_w, out_h, in_dim.2)
      }
      _ => unimplemented!(),
    }
  }
}

pub trait PoolKernel {}

pub struct AvgPool;
pub struct MaxPool;

impl PoolKernel for AvgPool {}
impl PoolKernel for MaxPool {}

pub trait PoolExt<Idx, V, Backend>: ArrayOp<V> where Idx: ArrayIndex {
  fn avg_pool(shape: PoolShape<Idx>, x_: Rc<Self>) -> Rc<PoolOp<Idx, V, AvgPool, Backend>>;
  fn max_pool(shape: PoolShape<Idx>, x_: Rc<Self>) -> Rc<PoolOp<Idx, V, MaxPool, Backend>>;
}

pub fn avg_pool<Op, Idx, V, Backend>(shape: PoolShape<Idx>, x_: Rc<Op>) -> Rc<PoolOp<Idx, V, AvgPool, Backend>> where Op: ArrayOp<V> + PoolExt<Idx, V, Backend>, Idx: ArrayIndex {
  PoolExt::avg_pool(shape, x_)
}

pub fn max_pool<Op, Idx, V, Backend>(shape: PoolShape<Idx>, x_: Rc<Op>) -> Rc<PoolOp<Idx, V, MaxPool, Backend>> where Op: ArrayOp<V> + PoolExt<Idx, V, Backend>, Idx: ArrayIndex {
  PoolExt::max_pool(shape, x_)
}

pub struct PoolOp<Idx, V, Kernel, Backend> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    PoolShape<Idx>,
  x_:   Rc<ArrayOp<V>>,
  x:    ArrayData<V>,
  y:    ArrayData<V>,
  kernel:   Kernel,
  backend:  Backend,
}

impl<Idx, V, Kernel, Backend> PoolOp<Idx, V, Kernel, Backend> where Idx: ArrayIndex {
  pub fn new(shape: PoolShape<Idx>, x_: Rc<ArrayOp<V>>, kernel: Kernel, backend: Backend, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(PoolOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      shape:    shape,
      x_:   x_,
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc.clone()),
      //tmp:  ArrayData::new(1, alloc),
      kernel:   kernel,
      backend:  backend,
    })
  }
}

impl<Idx, V, Kernel, Backend> ArrayOp<V> for PoolOp<Idx, V, Kernel, Backend> where PoolOp<Idx, V, Kernel, Backend>: AutodiffOp, Idx: ArrayIndex {
  fn _data(&self) -> &ArrayData<V> {
    &self.y
  }
}

pub struct GenPoolOp<Idx, V, Kernel, Backend> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    PoolShape<Idx>,
  x_:   Rc<ArrayOp<V>>,
  sel_: Rc<ArrayOp<V>>,
  x:    ArrayData<V>,
  sel:  ArrayData<V>,
  y:    ArrayData<V>,
  kernel:   Kernel,
  backend:  Backend,
}

#[derive(Clone, Copy)]
pub enum BatchStatsAverage {
  Geometric(f64),
  Arithmetic,
  ArithmeticCountCutoff(usize),
  ArithmeticRateCutoff(f64),
}

impl BatchStatsAverage {
  pub fn rate(self, update_ct: usize) -> f64 {
    let n = (update_ct + 1) as f64;
    match self {
      BatchStatsAverage::Geometric(rate) => rate,
      BatchStatsAverage::Arithmetic => 1.0 / n,
      BatchStatsAverage::ArithmeticCountCutoff(max_ct) => {
        if update_ct >= max_ct { 0.0 }
        else { 1.0 / n }
      }
      BatchStatsAverage::ArithmeticRateCutoff(max_rate) => {
        let rate = 1.0 / n;
        if rate >= max_rate { 0.0 }
        else { rate }
      }
    }
  }
}

#[derive(Clone, Copy)]
pub enum BatchStatsUnbias {
  Normalize,
}

#[derive(Clone, Copy)]
pub struct BatchStatsConfig {
  pub average:  BatchStatsAverage,
  //pub unbias:   BatchStatsUnbias,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BatchStatsMode {
  PassThrough,
  UseFixedRunningStats,
}

pub struct BatchStatsState<Idx> where Idx: ArrayIndex {
  pub reduce_axes:  Idx::Axes,
  pub cfg:          BatchStatsConfig,
  pub curr_txn:     Option<TxnId>,
  pub inner_mode:   BatchStatsMode,
  pub batch_ct:     usize,
  pub update_ct:    usize,
}

impl<Idx> BatchStatsState<Idx> where Idx: ArrayIndex {
  pub fn new(reduce_axes: Idx::Axes, cfg: BatchStatsConfig) -> Self {
    BatchStatsState{
      reduce_axes:  reduce_axes,
      cfg:          cfg,
      curr_txn:     None,
      inner_mode:   BatchStatsMode::PassThrough,
      batch_ct:     0,
      update_ct:    0,
    }
  }

  pub fn get_mode(&mut self, txn: TxnId) -> BatchStatsMode {
    match self.curr_txn {
      None => {
        self.curr_txn = Some(txn);
      }
      Some(prev_txn) => {
        if prev_txn != txn {
          self.curr_txn = Some(txn);
        }
      }
    }
    self.inner_mode
  }

  pub fn set_mode(&mut self, txn: TxnId, mode: BatchStatsMode) {
    match self.curr_txn {
      None => {
        self.curr_txn = Some(txn);
        self.inner_mode = mode;
      }
      Some(prev_txn) => {
        if prev_txn != txn {
          self.curr_txn = Some(txn);
          self.inner_mode = mode;
        } else {
          assert_eq!(self.inner_mode, mode);
        }
      }
    }
  }
}

pub struct BatchStatsControl {
  ops:  Vec<Rc<BatchStatsOpExt>>,
}

impl BatchStatsControl {
  pub fn new() -> Self {
    BatchStatsControl{ops: vec![]}
  }

  pub fn configure(&self, f: &Fn(&mut BatchStatsConfig)) {
    for op in self.ops.iter() {
      op._configure(f);
    }
  }

  pub fn set_mode(&self, txn: TxnId, mode: BatchStatsMode) {
    for op in self.ops.iter() {
      op._set_mode(txn, mode);
    }
  }

  pub fn accumulate(&self, txn: TxnId) {
    for op in self.ops.iter() {
      op._accumulate(txn);
    }
  }

  /*pub fn clear_accumulators(&self, txn: TxnId) {
    for op in self.ops.iter() {
      op._clear_accumulators(txn);
    }
  }*/

  pub fn update_stats(&self, prev_txn: TxnId, next_txn: TxnId) {
    for op in self.ops.iter() {
      op._update_stats(prev_txn, next_txn);
    }
  }
}

pub trait BatchStatsOpExt {
  fn _configure(&self, f: &Fn(&mut BatchStatsConfig));
  fn _set_mode(&self, txn: TxnId, mode: BatchStatsMode);
  fn _accumulate(&self, txn: TxnId);
  //fn _clear_accumulators(&self, txn: TxnId);
  fn _update_stats(&self, prev_txn: TxnId, next_txn: TxnId);
}

#[derive(Clone)]
pub struct BatchStatsOutput<M> {
  pub mean:     Rc<ArrayOp<M>>,
  pub var:      Rc<ArrayOp<M>>,
  pub mean_run: Rc<ArrayOp<M>>,
  pub var_run:  Rc<ArrayOp<M>>,
}

#[derive(Clone)]
pub struct BatchStatsOutputNew<M> {
  pub mean: Rc<BranchOp<CopyConstant<bool>, Rc<ArrayOp<M>>, Rc<ArrayOp<M>>, M>>,
  pub var:  Rc<BranchOp<CopyConstant<bool>, Rc<ArrayOp<M>>, Rc<ArrayOp<M>>, M>>,
}

pub trait BatchStatsExt<Idx, A, M> where Idx: ArrayIndex {
  fn batch_stats(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Self) -> BatchStatsOutput<M> where Self: Sized;
}

pub fn batch_stats<Op, Idx, A, M>(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Rc<Op>) -> BatchStatsOutput<M> where Rc<Op>: BatchStatsExt<Idx, A, M>, Op: 'static + ArrayOp<A>, Idx: ArrayIndex {
  <Rc<Op> as BatchStatsExt<Idx, A, M>>::batch_stats(reduce_axes, cfg, ctrl, x_)
}

pub struct BatchStatsOp<Idx, A, M> where Idx: ArrayIndex {
  node_id:      NodeId,
  stack:        OperatorStack,
  state:        RefCell<BatchStatsState<Idx>>,
  x_:           Rc<ArrayOp<A>>,
  mean_:        Weak<PassOp<M>>,
  mean_acc_:    Rc<ArraySrc<M>>,
  mean_run_:    Rc<ArraySrc<M>>,
  var_:         Weak<PassOp<M>>,
  var_acc_:     Rc<ArraySrc<M>>,
  var_run_:     Rc<ArraySrc<M>>,
  x:            ArrayData<A>,
  mean:         ArrayData<M>,
  mean_acc:     ArrayData<M>,
  mean_run:     ArrayData<M>,
  var:          ArrayData<M>,
  var_acc:      ArrayData<M>,
  var_run:      ArrayData<M>,
}

impl<Idx, A, M> BatchStatsOp<Idx, A, M> where BatchStatsOp<Idx, A, M>: AutodiffOp + BatchStatsOpExt, ArraySrc<M>: AutodiffOp, Idx: 'static + ArrayIndex, A: 'static, M: 'static {
  pub fn new<Op>(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Rc<Op>, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> M>) -> BatchStatsOutput<M> where Op: 'static + ArrayOp<A> {
    let node = NodeId::new();
    let x = x_.data();
    let mean = ArrayData::new(clk_horizon, alloc.clone());
    let var = ArrayData::new(clk_horizon, alloc.clone());
    let mean_ = PassOp::new(None, mean.clone());
    let var_ = PassOp::new(None, var.clone());
    // FIXME: some hackiness around clocks.
    let mean_acc_ = ArraySrc::new(clk_horizon, clk_horizon > 1, alloc.clone()); //src(alloc.clone());
    let mean_run_ = ArraySrc::new(clk_horizon, clk_horizon > 1, alloc.clone()); //src(alloc.clone());
    let var_acc_ = ArraySrc::new(clk_horizon, clk_horizon > 1, alloc.clone()); //src(alloc.clone());
    let var_run_ = ArraySrc::new(clk_horizon, clk_horizon > 1, alloc.clone()); //src(alloc.clone());
    let mean_acc = mean_acc_.data();
    let mean_run = mean_run_.data();
    let var_acc = var_acc_.data();
    let var_run = var_run_.data();
    let op = Rc::new(BatchStatsOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      state:    RefCell::new(BatchStatsState::new(reduce_axes, cfg)),
      x_:           x_,
      mean_:        Rc::downgrade(&mean_),
      mean_acc_:    mean_acc_,
      mean_run_:    mean_run_.clone(),
      var_:         Rc::downgrade(&var_),
      var_acc_:     var_acc_,
      var_run_:     var_run_.clone(),
      x:            x,
      mean:         mean,
      mean_acc:     mean_acc,
      mean_run:     mean_run,
      var:          var,
      var_acc:      var_acc,
      var_run:      var_run,
    });
    *mean_.x_.borrow_mut() = Some(AutodiffOp::from(op.clone()));
    *var_.x_.borrow_mut() = Some(AutodiffOp::from(op.clone()));
    ctrl.ops.push(op);
    BatchStatsOutput{
      mean:     mean_,
      var:      var_,
      mean_run: mean_run_,
      var_run:  var_run_,
    }
  }

  /*pub fn mean(&self) -> Rc<PassOp<M>> {
    match Weak::upgrade(&self.mean_op) {
      None => panic!(),
      Some(op) => op,
    }
  }

  pub fn var(&self) -> Rc<PassOp<M>> {
    match Weak::upgrade(&self.var_op) {
      None => panic!(),
      Some(op) => op,
    }
  }*/
}

//impl<Idx, A, M> BatchStatsOpExt for BatchStatsOp<Idx, A, M> where Idx: ArrayIndex {
impl<S> BatchStatsOpExt for BatchStatsOp<(usize, usize), BatchArray3d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _configure(&self, reconf: &Fn(&mut BatchStatsConfig)) {
    // FIXME(20170214): only safe to mutate state at the beginning of a txn.
    let mut state = self.state.borrow_mut();
    reconf(&mut state.cfg);
  }

  fn _set_mode(&self, txn: TxnId, mode: BatchStatsMode) {
    let mut state = self.state.borrow_mut();
    state.set_mode(txn, mode);
  }

  fn _accumulate(&self, txn: TxnId) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    // FIXME: does not account for non-uniform batch sizes.
    let n = (state.batch_ct + 1) as f32;
    //self.mean_acc.val.rollover(txn, self.mean_acc.val.var()); // FIXME
    if self.mean_acc.val.accumulate(txn, node, |val| val.as_view_mut().set_constant(0.0)) {
      assert!(!self.mean.val.overwrite(txn, node));
      self.mean_acc.val.get_mut(txn, node).as_view_mut().average(1.0 / n, self.mean.val.get_excl(txn, node).as_view());
    }
    //self.var_acc.val.rollover(txn, self.var_acc.val.var()); // FIXME
    if self.var_acc.val.accumulate(txn, node, |val| val.as_view_mut().set_constant(0.0)) {
      assert!(!self.var.val.overwrite(txn, node));
      self.var_acc.val.get_mut(txn, node).as_view_mut().average(1.0 / n, self.var.val.get_excl(txn, node).as_view());
    }
    state.batch_ct += 1;
  }

  /*fn _clear_accumulators(&self, prev_txn: TxnId, next_txn: TxnId) {
    unimplemented!();
  }*/

  fn _update_stats(&self, prev_txn: TxnId, next_txn: TxnId) {
    let node = self._id();
    let mut state = self.state.borrow_mut();
    assert!(state.batch_ct >= 1);
    // FIXME: rather than directly average with `rate`, should use a
    // normalized rate for bias correction.
    let rate = state.cfg.average.rate(state.update_ct) as f32;
    //self.mean_run.val.rollover(next_txn, self.mean_run.val.var()); // FIXME
    if self.mean_run.val.accumulate(next_txn, node, |val| val.as_view_mut().set_constant(0.0)) {
      if rate != 0.0 {
        self.mean_run.val.get_mut(next_txn, node).as_view_mut().average(rate, self.mean_acc.val.get(prev_txn, node).as_view());
      }
      if self.mean_acc.val.overwrite(next_txn, node) {
        self.mean_acc.val.get_excl(next_txn, node).as_view_mut().set_constant(0.0);
      }
    }
    //self.var_run.val.rollover(next_txn, self.var_run.val.var()); // FIXME
    if self.var_run.val.accumulate(next_txn, node, |val| val.as_view_mut().set_constant(0.0)) {
      if rate != 0.0 {
        self.var_run.val.get_mut(next_txn, node).as_view_mut().average(rate, self.var_acc.val.get(prev_txn, node).as_view());
      }
      if self.var_acc.val.overwrite(next_txn, node) {
        self.var_acc.val.get_excl(next_txn, node).as_view_mut().set_constant(0.0);
      }
    }
    state.batch_ct = 0;
    state.update_ct += 1;
  }
}

impl<S> AutodiffOp for BatchStatsOp<(usize, usize), BatchArray3d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
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
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }
}

pub struct BatchJoinOp<A, B, Join> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<ArrayOp<A>>,
  x:    ArrayData<A>,
  y:    ArrayData<B>,
  kernel:   Join,
}

impl<A, B, Join> BatchJoinOp<A, B, Join> {
  pub fn new<Op>(x_: Rc<Op>, kernel: Join, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> B>) -> Rc<BatchJoinOp<A, B, Join>> where Op: 'static + ArrayOp<A> {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(BatchJoinOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc),
      kernel:   kernel,
    })
  }
}

pub trait BatchSumExt<Op, A, B> where Op: ArrayOp<A> {
  fn batch_sum(x_: Rc<Op>) -> Rc<BatchJoinOp<A, B, SumJoinKernel>>;
}

pub fn batch_sum<Op, A, B>(x_: Rc<Op>) -> Rc<BatchJoinOp<A, B, SumJoinKernel>> where Rc<Op>: BatchSumExt<Op, A, B>, Op: ArrayOp<A> {
  <Rc<Op> as BatchSumExt<Op, A, B>>::batch_sum(x_)
}

/*impl BatchSumExt<Batch<f32>, f32> for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn batch_sum(x_: Rc<ArrayOp<Batch<f32>>>) -> Rc<Self> {
    unimplemented!();
  }
}*/

impl<A, B, Join> ArrayOp<B> for BatchJoinOp<A, B, Join> where BatchJoinOp<A, B, Join>: AutodiffOp {
  default fn _data(&self) -> &ArrayData<B> {
    &self.y
  }
}

/*impl AutodiffObjective for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if !self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}*/

impl AutodiffOp for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
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
      let x_val = self.x.val.get(txn, node);
      let mut y_val = self.y.val.get_excl(txn, node);
      *y_val = 0.0;
      let batch_sz = x_val.batch_size();
      for i in 0 .. batch_sz {
        *y_val += x_val[i];
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let batch_sz = self.x.val.get(txn, node).batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
      let mut x_grad = self.x.grad.get_mut(txn, node);
      let y_grad = self.y.grad.get(txn, node);
      for i in 0 .. batch_sz {
        x_grad[i] += *y_grad;
      }
    }
  }
}

pub struct SequentialJoinOp<A, B, JoinF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<ArrayOp<A>>,
  x:    ArrayData<A>,
  y:    ArrayData<B>,
  curr_clk: Cell<usize>,
  kernel:   JoinF,
}

impl AutodiffOp for SequentialJoinOp<Batch<f32>, Batch<f32>, SumJoinKernel> {
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
    let clk = self.curr_clk.get();
    let x_val = self.x.val.get_clk(clk, txn, node);
    let batch_sz = x_val.batch_size();
    if self.y.val.accumulate(txn, node, |val| val.reshape_mut(batch_sz).set_constant(0.0)) {
      self.y.val.get_mut(txn, node).reshape_mut(batch_sz)
        .add(1.0, x_val.reshape(batch_sz));
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let clk = self.curr_clk.get();
    let y_grad = self.y.grad.get_clk(clk, txn, node);
    let batch_sz = y_grad.batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
      self.x.grad.get_mut(txn, node).reshape_mut(batch_sz)
        .add(1.0, y_grad.reshape(batch_sz));
    }
  }

  fn _reset_clock(&self) {
    self.curr_clk.set(0);
  }

  fn _set_clock(&self, new_clk: usize) {
    self.curr_clk.set(new_clk);
  }
}

pub struct OneHotOp<A, Idx, Out> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<ArrayOp<A>>,
  index_:   Rc<ArrayOp<Idx>>,
  x:        ArrayData<A>,
  index:    ArrayData<Idx>,
  output:   ArrayData<Out>,
}

pub fn sink<Op, A>(x_: Rc<Op>) -> Rc<ArraySink<Op, A>> where Op: ArrayOp<A> {
  let x = x_.data();
  Rc::new(ArraySink{
    node:   NodeId::new(),
    x_:     x_,
    x:      x,
    _m:     PhantomData,
  })
}

pub struct ArraySink<Op, A> where Op: ArrayOp<A> {
  node: NodeId,
  x_:   Rc<Op>,
  x:    ArrayData<A>,
  _m:   PhantomData<A>,
}

/*(impl<Op, A> Deref for ArraySink<Op, A> where Op: ArrayOp<A> {
  type Target = Op;

  fn deref(&self) -> &Op {
    &*self.x_
  }
}*/

//impl<Op> AutodiffSink<Op> for ArraySink<Op, f32> where Op: ArrayOp<f32> {
impl<Op> AutodiffSink for ArraySink<Op, f32> where Op: ArrayOp<f32> {
  fn _op(&self) -> &AutodiffOp {
    &*self.x_
  }

  fn _set_source(&self, txn: TxnId) {
    let node = self.node;
    if self.x.grad.overwrite(txn, node) {
      *self.x.grad.get_excl(txn, node) = 1.0;
    }
  }
}

pub fn lst_sq_loss<A, Op>(x_: Rc<Op>, t_: Rc<Op>) -> Rc<LstSqLoss<A>> where Op: 'static + ArrayOp<A> {
  unimplemented!();
}

pub struct LstSqLoss<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<ArrayOp<A>>,
  target_:  Rc<ArrayOp<A>>,
  x:        ArrayData<A>,
  target:   ArrayData<A>,
  loss:     ArrayData<A>,
}

impl AutodiffOp for LstSqLoss<Batch<f32>> {
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
      let mut loss_val = self.loss.val.get_mut(txn, node);
      loss_val.set_batch_size(batch_sz, 0.0);
      loss_val.reshape_mut(batch_sz).copy(self.x.val.get(txn, node).reshape(batch_sz));
      loss_val.reshape_mut(batch_sz).add(-1.0, self.target.val.get(txn, node).reshape(batch_sz));
      loss_val.reshape_mut(batch_sz).square();
      loss_val.reshape_mut(batch_sz).scale(0.5);
    }
  }

  fn _backward(&self, txn: TxnId, gauss_newton: bool) {
    let node = self._id();
    if gauss_newton {
      unimplemented!();
    } else {
      let batch_sz = self.x.val.get(txn, node).batch_size();
      if self.x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
        self.x.grad.get_mut(txn, node).set_batch_size(batch_sz, 0.0);
        // FIXME(20170209): mix with `loss.grad`; requires a tmp variable.
        self.x.grad.get_mut(txn, node).reshape_mut(batch_sz).add(1.0, self.x.val.get(txn, node).reshape(batch_sz));
        self.x.grad.get_mut(txn, node).reshape_mut(batch_sz).add(-1.0, self.target.val.get(txn, node).reshape(batch_sz));
      }
    }
  }
}

pub struct KL1LossLink;
pub struct KL2LossLink;
pub struct LRLossLink;
pub struct NLLLossLink;

pub trait LikelihoodLossLink {}

impl LikelihoodLossLink for KL1LossLink {}
impl LikelihoodLossLink for KL2LossLink {}
impl LikelihoodLossLink for LRLossLink {}
impl LikelihoodLossLink for NLLLossLink {}

pub struct SoftmaxLoss<A, Target, Loss, Link> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<ArrayOp<A>>,
  target_:  Option<Rc<ArrayOp<Target>>>,
  prob_:    Weak<PassOp<A>>,
  loss_:    Weak<PassOp<Loss>>,
  x:        ArrayData<A>,
  target:   Option<ArrayData<Target>>,
  prob:     ArrayData<A>,
  loss:     ArrayData<Loss>,
  /*logit:        ArrayData<A>,
  max_logit:    ArrayData<Loss>,
  factor:       ArrayData<A>,
  max_factor:   ArrayData<Loss>,*/
  link:     Link,
}

impl<A, Target, Loss, Link> SoftmaxLoss<A, Target, Loss, Link> where SoftmaxLoss<A, Target, Loss, Link>: AutodiffOp, A: 'static, Target: 'static, Loss: 'static, Link: 'static {
  //pub fn new(x_: Rc<ArrayOp<A>>, target_: Option<Rc<ArrayOp<Target>>>, link: Link, clk_horizon: usize, prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> Rc<Self> {
  pub fn new<Op>(x_: Rc<Op>, target_: Option<Rc<ArrayOp<Target>>>, link: Link, clk_horizon: usize, prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> (Rc<Self>, Rc<PassOp<A>>, Rc<PassOp<Loss>>) where Op: 'static + ArrayOp<A> {
    let node = NodeId::new();
    let in_degree = match target_ {
      None      => 1,
      Some(_)   => 2,
    };
    //let x__: Rc<AutodiffOp> = ArrayOp::downgrade(x_.clone());
    let x = x_.data();
    let target = target_.as_ref().map(|t_| t_.data());
    let prob = ArrayData::new(clk_horizon, prob_alloc.clone());
    let loss = ArrayData::new(clk_horizon, loss_alloc.clone());
    let prob_ = PassOp::new(None, prob.clone());
    let loss_ = PassOp::new(None, loss.clone());
    let softmax = Rc::new(SoftmaxLoss{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      x_:       x_,
      target_:  target_,
      prob_:    Rc::downgrade(&prob_),
      loss_:    Rc::downgrade(&loss_),
      x:        x,
      target:   target,
      prob:     prob,
      loss:     loss, //ArrayData::new(clk_horizon, alloc.clone()),
      //logit:    ArrayData::new(1, alloc.clone()),
      link:     link,
    });
    *prob_.x_.borrow_mut() = Some(AutodiffOp::from(softmax.clone()));
    *loss_.x_.borrow_mut() = Some(AutodiffOp::from(softmax.clone()));
    (softmax, prob_, loss_)
  }

  /*pub fn prob(&self) -> Rc<PassOp<A>> {
    Weak::upgrade(&self.prob_).unwrap()
  }

  pub fn loss(&self) -> Rc<PassOp<Loss>> {
    Weak::upgrade(&self.loss_).unwrap()
  }*/
}

pub trait SoftmaxKLLossExt<A, L> {
  fn softmax_kl2_loss(x_: Rc<ArrayOp<A>>, target_: Rc<ArrayOp<A>>) -> Rc<SoftmaxLoss<A, A, L, KL2LossLink>>;
}

pub trait SoftmaxNLLLossExt<Op, A, Target, L> where Op: ArrayOp<A> {
  //fn softmax_nll_loss(x_: Rc<Op>, target_: Rc<ArrayOp<Target>>) -> (Rc<SoftmaxLoss<A, Target, L, NLLLossLink>>, Rc<PassOp<A>>, Rc<PassOp<L>>);
  fn softmax_nll_loss(x_: Rc<Op>, target_: Rc<ArrayOp<Target>>) -> (Rc<PassOp<A>>, Rc<PassOp<L>>);
}

//pub fn softmax_nll_loss<Op, A, Target, L>(x_: Rc<Op>, target_: Rc<ArrayOp<Target>>) -> (Rc<SoftmaxLoss<A, Target, L, NLLLossLink>>, Rc<PassOp<A>>, Rc<PassOp<L>>) where Rc<Op>: SoftmaxNLLLossExt<Op, A, Target, L>, Op: ArrayOp<A> {
pub fn softmax_nll_loss<Op, A, Target, L>(x_: Rc<Op>, target_: Rc<ArrayOp<Target>>) -> (Rc<PassOp<A>>, Rc<PassOp<L>>) where Rc<Op>: SoftmaxNLLLossExt<Op, A, Target, L>, Op: ArrayOp<A> {
  <Rc<Op> as SoftmaxNLLLossExt<Op, A, Target, L>>::softmax_nll_loss(x_, target_)
}

/*impl<S, Target, Loss, Link> AutodiffOp for SoftmaxLoss<BatchArray1d<f32, S>, Target, Loss, Link> where S: DerefMut<Target=[f32]>, Link: LikelihoodLossLink {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      if let Some(ref target_) = self.target_ {
        target_._push(epoch, apply);
      }
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref target_) = self.target_ {
        target_._pop(epoch, apply);
      }
      self.x_._pop(epoch, apply);
    }
  }

  default fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.loss.rollover_all(txn, vars);
  }

  default fn _forward(&self, txn: TxnId) {
    unimplemented!();
  }
}*/

impl<S> AutodiffOp for SoftmaxLoss<BatchArray1d<f32, S>, BatchArray1d<f32, S>, Batch<f32>, KL2LossLink> where S: DerefMut<Target=[f32]> {
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
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOp for SoftmaxLoss<BatchArray1d<f32, S>, Batch<(u32, f32)>, Batch<f32>, LRLossLink> where S: DerefMut<Target=[f32]> {
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
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOp for SoftmaxLoss<BatchArray1d<f32, S>, Batch<u32>, Batch<f32>, NLLLossLink> where S: DerefMut<Target=[f32]> {
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
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    unimplemented!();
  }

  fn _r_backward(&self, txn: TxnId) {
    unimplemented!();
  }
}
