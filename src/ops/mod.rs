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

pub fn src<A, F>(cons: F) -> Rc<SrcOp<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  SrcOp::new(Rc::new(cons))
}

/*pub fn sequential_src<A, F>(horizon: usize, cons: F) -> Rc<SrcOp<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  SrcOp::new(horizon, true, Rc::new(cons))
}*/

/*pub fn test_var() {
  let x: Rc<SrcOp<Array1d<f32>>> = var(|_, _| Array1d::zeros(10));
}*/

pub struct CopyConstant<A> where A: Copy {
  /*node_id:  NodeId,
  stack:    OperatorStack,*/
  pub var:  TxnCopyVar<A>,
}

pub struct SrcOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  /*horizon:  usize,
  clock:    bool,*/
  alloc:    Rc<Fn(TxnId, NodeId) -> A>,
  data:     AData<A>,
  adj:      RefCell<Option<Rc<AVar<AData<A>>>>>,
}

impl<A> SrcOp<A> {
  pub fn new(/*horizon: usize, clock: bool,*/ alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<Self> {
    let node = NodeId::new();
    Rc::new(SrcOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 0),
      /*horizon:  horizon,
      clock:    clock,*/
      alloc:    alloc.clone(),
      data:     AData::new(/*horizon,*/ alloc),
      adj:      RefCell::new(None),
    })
  }
}

/*impl<A> AVar<AData<A>> for SrcOp<A> where A: 'static, SrcOp<A>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
    &self.data
  }

  default fn adjoint(&self) -> Rc<AVar<AData<A>>> {
    let adj_src_ = SrcOp::new(self.horizon, self.clock, self.alloc.clone());
    adj_src_
  }
}*/

impl<A> AVar<AData<A>> for SrcOp<A> where A: 'static, SrcOp<A>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
    &self.data
  }

  default fn _make_adjoint(&self) -> Rc<AVar<AData<A>>> {
    let adj_src_ = SrcOp::new(/*self.horizon, self.clock,*/ self.alloc.clone());
    adj_src_
  }

  default fn adjoint(&self) -> Rc<AVar<AData<A>>> {
    if self.adj.borrow().is_none() {
      *self.adj.borrow_mut() = Some(self._make_adjoint());
    }
    self.adj.borrow().as_ref().unwrap().clone()
  }
}

impl AOp for SrcOp<f32> {
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

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    //println!("DEBUG: trace: persist: {:?}", txn);
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId) {
  }

  /*fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }*/
}

impl AOp for SrcOp<Batch<u32>> {
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

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId) {
  }

  /*fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }*/
}

impl AOp for SrcOp<Array1d<f32>> {
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

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId) {
  }

  /*fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }*/
}

impl AOp for SrcOp<Array2d<f32>> {
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

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId) {
  }

  /*fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }*/
}

/*impl AVar<AData<Array4d<f32>>> for SrcOp<Array4d<f32>> {
  fn data(&self) -> AData<Array4d<f32>> {
    self.data.clone()
  }
}*/

impl AOp for SrcOp<Array4d<f32>> {
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

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self.data.rollover_all(txn, vars);
  }

  fn _forward(&self, _txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId) {
  }

  /*fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }*/
}

pub fn pass<A, Op>(x_: Rc<Op>) -> Rc<PassOp<A>> where Op: 'static + AVar<AData<A>> {
  let data = x_.data();
  PassOp::new(Some(AOp::from(x_)), data)
}

pub struct PassOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       RefCell<Option<Rc<AOp>>>,
  data:     AData<A>,
}

impl<A> PassOp<A> {
  pub fn new(x_: Option<Rc<AOp>>, data: AData<A>) -> Rc<Self> {
    let node = NodeId::new();
    Rc::new(PassOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       RefCell::new(x_),
      data:     data,
    })
  }
}

impl<A> AVar<AData<A>> for PassOp<A> {
  default fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}

impl<A> AOp for PassOp<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  default fn _backward(&self, _txn: TxnId) {
  }
}

pub fn no_pass<A, Op>(x_: Rc<Op>) -> Rc<NoPassOp<A>> where Op: 'static + AVar<AData<A>> {
  let data = x_.data();
  NoPassOp::new(Some(AOp::from(x_)), data)
}

pub struct NoPassOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       RefCell<Option<Rc<AOp>>>,
  data:     AData<A>,
}

impl<A> NoPassOp<A> {
  pub fn new(x_: Option<Rc<AOp>>, data: AData<A>) -> Rc<Self>{
    let node = NodeId::new();
    Rc::new(NoPassOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       RefCell::new(x_),
      data:     data,
    })
  }
}

impl<A> AVar<AData<A>> for NoPassOp<A> {
  default fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}

impl<A> AOp for NoPassOp<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      // Forward pass stops here.
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  default fn _backward(&self, _txn: TxnId) {
  }
}

pub struct IoOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  //x_:       Rc<AVar<AData<A>>>,
  x_:       RefCell<Option<Rc<AOp>>>,
  data:     AData<A>,
}

impl<A> IoOp<A> {
  pub fn new(x_: Option<Rc<AOp>>, data: AData<A>) -> Rc<Self>{
    let node = NodeId::new();
    Rc::new(IoOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       RefCell::new(x_),
      data:     data,
    })
  }
}

impl<A> AVar<AData<A>> for IoOp<A> {
  default fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}

impl<A> AOp for IoOp<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      let x_ = self.x_.borrow();
      x_.as_ref().unwrap()._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  default fn _backward(&self, _txn: TxnId) {
  }
}

pub struct InitializeOp<A, Init> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  data:     AData<A>,
  kernel:   Init,
}

pub fn init_val<R, A, F>(f: F) -> impl Fn(TxnId, NodeId, Rc<RefCell<R>>, AData<A>) where R: Rng, F: Fn(Rc<RefCell<R>>, &mut A) {
  let init_f = Rc::new(f);
  move |txn: TxnId, node: NodeId, rng: Rc<RefCell<R>>, data: AData<A>| {
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

/*impl<Op, A, F> InitializeExt<A, F, Rc<F>> for Rc<Op> where Op: 'static + AVar<AData<A>>, F: Fn(Rc<RefCell<ChaChaRng>>, &mut A) {
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

//impl<Op, A, F> InitializeExt<A, F, Rc<F>> for Rc<Op> where Op: 'static + AVar<AData<A>>, F: Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<A>) {
impl<Op, A, F> InitializeExt<A, F, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<A>)>> for Rc<Op> where Op: 'static + AVar<AData<A>>, F: 'static + Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<A>) {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<A>)>>> {
    let node = NodeId::new();
    let stack = OperatorStack::new(node, 1);
    let init: Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<A>)> = Rc::new(f);
    Rc::new(InitializeOp{
      node_id:  node,
      stack:    stack,
      x_:   self.clone(),
      data: self.data(),
      kernel:   init,
    })
  }
}

impl<A, Init> AVar<AData<A>> for InitializeOp<A, Init> where InitializeOp<A, Init>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}

/*impl AVar<AData<f32>> for InitializeOp<f32, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<f32>)>> {
  fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}*/

//impl<F> AOp for InitializeOp<f32, Rc<F>> where F: Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<f32>) {
impl AOp for InitializeOp<f32, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<f32>)>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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
    //println!("DEBUG: trace: init");
    (self.kernel)(txn, node, seed_rng, self.data.clone());
  }

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId) {
  }
}

//impl<S, F> AOp for InitializeOp<Array1d<f32, S>, Rc<F>> where S: DerefMut<Target=[f32]>, F: Fn(Rc<RefCell<ChaChaRng>>, &mut Array1d<f32, S>) {
//impl<S, F> AOp for InitializeOp<Array1d<f32, S>, Rc<F>> where S: DerefMut<Target=[f32]>, F: Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<Array1d<f32, S>>) {
impl<S> AOp for InitializeOp<Array1d<f32, S>, Rc<Fn(TxnId, NodeId, Rc<RefCell<ChaChaRng>>, AData<Array1d<f32, S>>)>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, _txn: TxnId) {
  }
}

pub struct BranchOp<Cond, Off, On, Data> {
  node_id:  NodeId,
  stack:    OperatorStack,
  cond:     Cond,
  off_:     Off,
  on_:      On,
  off:      Data,
  on:       Data,
  output:   Data,
}

impl<Cond, A> BranchOp<Cond, Rc<AVar<AData<A>>>, Rc<AVar<AData<A>>>, AData<A>> {
  //pub fn new<On, Off, F>(cond: Cond, on_: Rc<On>, off_: Rc<Off>, /*clk_horizon: usize,*/ alloc: Rc<F>) -> Rc<Self> where On: 'static + AVar<AData<A>>, Off: 'static + AVar<AData<A>>, F: 'static + Fn(TxnId, NodeId) -> A {
  pub fn new<Off, On>(cond: Cond, off_: Rc<Off>, on_: Rc<On>, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<Self> where On: 'static + AVar<AData<A>>, Off: 'static + AVar<AData<A>> {
  //pub fn new(cond: Cond, on_: Rc<AVar<AData<A>>>, off_: Rc<AVar<AData<A>>>, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<Self> {
    let node = NodeId::new();
    let off = off_.data();
    let on = on_.data();
    Rc::new(BranchOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      cond:     cond,
      /*off_:     ArrayOp::from(off_),
      on_:      ArrayOp::from(on_),*/
      off_:     AVar::from(off_),
      on_:      AVar::from(on_),
      off:      off,
      on:       on,
      output:   AData::new(/*clk_horizon,*/ alloc),
    })
  }
}

impl<Cond, On, Off, Data> AVar<Data> for BranchOp<Cond, On, Off, Data> where BranchOp<Cond, On, Off, Data>: AOp, Data: AVarOutput {
  default fn _owned_data(&self) -> &Data {
    &self.output
  }
}

impl<Cond, On, Off, A> AVar<AData<A>> for BranchOp<Cond, On, Off, AData<A>> where BranchOp<Cond, On, Off, AData<A>>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
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

impl<Op, S> SpecialMapExt</*f32,*/ Array1d<f32, S>> for Rc<Op> where Op: 'static + AVar<AData<Array1d<f32, S>>>, S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn rect(&self) -> Rc<MapOp<Array1d<f32, S>, RectMapKernel>> {
    //let clk_horizon = self.data().horizon();
    MapOp::new(RectMapKernel, self.clone(), /*clk_horizon,*/ {
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
  x_:   Rc<AVar<AData<A>>>,
  x:    AData<A>,
  y:    AData<A>,
  kernel:   MapF,
}

impl<A, MapF> MapOp<A, MapF> {
  pub fn new<F>(kernel: MapF, x_: Rc<AVar<AData<A>>>, /*clk_horizon: usize,*/ alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> A {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(MapOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    AData::new(/*clk_horizon,*/ alloc),
      kernel:   kernel,
    })
  }
}

impl<A, MapF> AVar<AData<A>> for MapOp<A, MapF> where MapOp<A, MapF>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
    &self.y
  }
}

impl<S, MapF> AOp for MapOp<Array1d<f32, S>, MapF> where S: DerefMut<Target=[f32]>, MapF: SpecialMapKernel {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

impl<S> AOp for MapOp<Array1d<f32, S>, RectMapKernel> where S: DerefMut<Target=[f32]> {
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

  fn _backward(&self, txn: TxnId) {
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

  /*fn _r_forward(&self, txn: TxnId) {
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
  }*/
}

impl<S> AOp for MapOp<Array1d<f32, S>, LogisticMapKernel> where S: DerefMut<Target=[f32]> {
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

  fn _backward(&self, txn: TxnId) {
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
}

impl<S> AOp for MapOp<Array1d<f32, S>, TanhMapKernel> where S: DerefMut<Target=[f32]> {
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

  fn _backward(&self, txn: TxnId) {
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
}

pub struct TransformOp<A, B, Transform> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<AVar<AData<A>>>,
  x:    AData<A>,
  y:    AData<B>,
  kernel:   Transform,
}

pub struct CastTransform;
pub struct FlattenTransform;
pub struct ReshapeTransform<Idx> {
  pub dim:  Idx,
}

pub struct ZeroPadTransform {
  pub axis: usize,
  pub dim:  usize,
}

pub trait CastExt<A, B> {
  fn cast(&self) -> Rc<TransformOp<A, B, CastTransform>>;
}

pub trait FlattenExt<A, B> {
  fn flatten(&self) -> Rc<TransformOp<A, B, FlattenTransform>>;
}

pub trait ReshapeExt<Idx, A, B> {
  fn reshape(&self, dim: Idx) -> Rc<TransformOp<A, B, ReshapeTransform<Idx>>>;

  fn reify(&self, dim: Idx) -> Rc<TransformOp<A, B, ReshapeTransform<Idx>>> {
    self.reshape(dim)
  }
}

pub trait ZeroPadExt<A, B> {
  fn zero_pad(&self, axis: usize, dim: usize) -> Rc<TransformOp<A, B, ZeroPadTransform>>;
}

impl<A, B, Transform> TransformOp<A, B, Transform> {
  pub fn new<F>(x_: Rc<AVar<AData<A>>>, transform: Transform, /*clk_horizon: usize,*/ alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> B {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(TransformOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    AData::new(/*clk_horizon,*/ alloc),
      kernel:   transform,
    })
  }
}

impl<A, B, Transform> AVar<AData<B>> for TransformOp<A, B, Transform> where TransformOp<A, B, Transform>: AOp {
  default fn _owned_data(&self) -> &AData<B> {
    &self.y
  }
}

impl<S> FlattenExt<Array3d<f32, S>, Array1d<f32, S>> for Rc<AVar<AData<Array3d<f32, S>>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn flatten(&self) -> Rc<TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform>> {
    //let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), FlattenTransform, /*clk_horizon,*/ {
      let x = self.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim.flat_len());
        Array1d::from_storage(dim.flat_len(), buf)
      })
    })
  }
}

impl<S> AOp for TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    let node = self._id();
    if self.x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      self.x.grad.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.y.grad.get(txn, node).as_view());
    }
  }

  /*fn _r_forward(&self, txn: TxnId) {
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

pub struct JoinOp<A, JoinF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  xs_:  Vec<Rc<AVar<AData<A>>>>,
  xs:   Vec<AData<A>>,
  y:    AData<A>,
  kernel:   JoinF,
}

impl<A, JoinF> JoinOp<A, JoinF> {
  pub fn new<F>(xs_: Vec<Rc<AVar<AData<A>>>>, kernel: JoinF, /*clk_horizon: usize,*/ alloc: Rc<F>) -> Rc<JoinOp<A, JoinF>> where F: 'static + Fn(TxnId, NodeId) -> A {
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
      y:        AData::new(/*clk_horizon,*/ alloc),
      kernel:   kernel,
    })
  }
}

impl<A, JoinF> AVar<AData<A>> for JoinOp<A, JoinF> where JoinOp<A, JoinF>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
    &self.y
  }
}

pub struct AxisJoinKernel;
pub struct SumJoinKernel;

pub trait AxisJoinExt<Op, A> where Op: AVar<AData<A>> {
  fn axis_join(xs: Vec<Rc<Op>>) -> Rc<JoinOp<A, AxisJoinKernel>>;
}

pub fn axis_join<Op, A>(xs: Vec<Rc<Op>>) -> Rc<JoinOp<A, AxisJoinKernel>> where Rc<Op>: AxisJoinExt<Op, A>, Op: AVar<AData<A>> {
  <Rc<Op> as AxisJoinExt<Op, A>>::axis_join(xs)
}

pub trait AddExt<A> {
  fn add<RhsOp>(&self, x_: Rc<RhsOp>) -> Rc<JoinOp<A, SumJoinKernel>> where RhsOp: 'static + AVar<AData<A>>;
}

impl<Op, A> AddExt<A> for Rc<Op> where Rc<JoinOp<A, SumJoinKernel>>: SumExt<A>, Op: 'static + AVar<AData<A>> {
  default fn add<RhsOp>(&self, x_: Rc<RhsOp>) -> Rc<JoinOp<A, SumJoinKernel>> where RhsOp: 'static + AVar<AData<A>> {
    //<Rc<JoinOp<A, SumJoinKernel>> as SumExt<A>>::sum(vec![ArrayOp::from(self.clone()), ArrayOp::from(x_)])
    <Rc<JoinOp<A, SumJoinKernel>> as SumExt<A>>::sum(vec![self.clone(), x_])
  }
}

pub trait SumExt<A> {
  fn sum(xs_: Vec<Rc<AVar<AData<A>>>>) -> Rc<JoinOp<A, SumJoinKernel>>;
}

pub fn sum<A>(xs_: Vec<Rc<AVar<AData<A>>>>) -> Rc<JoinOp<A, SumJoinKernel>> where Rc<JoinOp<A, SumJoinKernel>>: SumExt<A> {
  <Rc<JoinOp<A, SumJoinKernel>> as SumExt<A>>::sum(xs_)
}

impl<Op, S> AxisJoinExt<Op, Array1d<f32, S>> for Rc<Op> where Op: AVar<AData<Array1d<f32, S>>>, S: DerefMut<Target=[f32]> {
  fn axis_join(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<Array1d<f32, S>, AxisJoinKernel>> {
    unimplemented!();
  }
}

/*impl<Op, A> SumExt<A> for Rc<Op> where Op: AVar<AData<A>> {
  default fn sum(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<A, SumJoinKernel>> {
    unimplemented!();
  }

  default fn add(&self, x_: Rc<Op>) -> Rc<JoinOp<A, SumJoinKernel>> {
    Self::sum(vec![self.clone(), x_])
  }
}*/

/*impl<Op, S> SumExt<Array1d<f32, S>> for Rc<Op> where Op: AVar<AData<Array1d<f32, S>>>, S: DerefMut<Target=[f32]> {
  fn sum(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> where S: DerefMut<Target=[f32]> {
    unimplemented!();
  }

  fn add(&self, x_: Rc<Op>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> {
    Self::sum(vec![self.clone(), x_])
  }
}*/

impl<S> AOp for JoinOp<Array1d<f32, S>, SumJoinKernel> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      for x_ in self.xs_.iter() {
        x_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    let node = self._id();
    for x in self.xs.iter() {
      if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
        x.grad.get_mut(txn, node).as_view_mut().add(1.0, self.y.grad.get(txn, node).as_view());
      }
    }
  }

  /*fn _r_forward(&self, txn: TxnId) {
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
  }*/
}

pub struct SplitOp<A, SplitF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<AVar<AData<A>>>,
  y:    AData<A>,
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

impl<S> DummyExt<Array1d<f32, S>> for Rc<AVar<AData<Array1d<f32, S>>>> where S: DerefMut<Target=[f32]> {
}

pub struct SymmUnitClipKernel;

pub trait SymmClipExt<A, Clip> {
  fn symm_unit_clip(&self, c_: Rc<AVar<AData<Clip>>>) -> Rc<ClipOp<A, Clip, SymmUnitClipKernel>>;
}

pub struct ClipOp<A, Clip, Kernel> {
  node_id:  NodeId,
  stack:    OperatorStack,
  c_:   Rc<AVar<AData<Clip>>>,
  x_:   Rc<AVar<AData<A>>>,
  c:    AData<Clip>,
  x:    AData<A>,
  y:    AData<A>,
  kernel:   Kernel,
}

impl<A, Clip, Kernel> ClipOp<A, Clip, Kernel> {
  pub fn new(x_: Rc<AVar<AData<A>>>, c_: Rc<AVar<AData<Clip>>>, kernel: Kernel, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<ClipOp<A, Clip, Kernel>> {
    let node = NodeId::new();
    let x = x_.data();
    let c = c_.data();
    Rc::new(ClipOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      c_:       c_,
      x_:       x_,
      c:        c,
      x:        x,
      y:        AData::new(/*clk_horizon,*/ alloc),
      kernel:   kernel,
    })
  }
}

impl<A, Clip, Kernel> AVar<AData<A>> for ClipOp<A, Clip, Kernel> where ClipOp<A, Clip, Kernel>: AOp {
  default fn _owned_data(&self) -> &AData<A> {
    &self.y
  }
}

pub trait MultExt<A, B, V, W> {
  fn mult(&self, x: Rc<AVar<AData<V>>>) -> Rc<LinearOp<A, B, V, W>>;
  fn mult_add(&self, x: Rc<AVar<AData<V>>>, b: Rc<AVar<AData<B>>>) -> Rc<LinearOp<A, B, V, W>>;
}

pub struct LinearOp<A, B, V, W> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a_:   Rc<AVar<AData<A>>>,
  b_:   Option<Rc<AVar<AData<B>>>>,
  x_:   Rc<AVar<AData<V>>>,
  a:    AData<A>,
  b:    Option<AData<B>>,
  x:    AData<V>,
  y:    AData<W>,
  tmp:  AData<W>,
  adj:  RefCell<Option<Rc<AVar<AData<W>>>>>,
}

impl<A, B, V, W> LinearOp<A, B, V, W> {
  pub fn new<F>(a_: Rc<AVar<AData<A>>>, x_: Rc<AVar<AData<V>>>, b_: Option<Rc<AVar<AData<B>>>>, /*clk_horizon: usize,*/ alloc: Rc<F>) -> Rc<LinearOp<A, B, V, W>> where F: 'static + Fn(TxnId, NodeId) -> W {
    let node = NodeId::new();
    let in_degree = match b_ {
      None    => 2,
      Some(_) => 3,
    };
    let a = a_.data();
    let b = b_.as_ref().map(|b_| b_.data());
    let x = x_.data();
    Rc::new(LinearOp{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      a_:   a_,
      b_:   b_,
      x_:   x_,
      a:    a,
      b:    b,
      x:    x,
      y:    AData::new(/*clk_horizon,*/ alloc.clone()),
      tmp:  AData::new(/*1,*/ alloc),
      adj:  RefCell::new(None),
    })
  }
}

impl<A, B, V, W> AVar<AData<W>> for LinearOp<A, B, V, W> where LinearOp<A, B, V, W>: AOp {
  default fn _owned_data(&self) -> &AData<W> {
    &self.y
  }

  // FIXME(20170530): requires extra trait bounds on inputs.
  /*default fn _make_adjoint(&self) -> Rc<AVar<AData<W>>> {
    let adj_a_ = self.a_.adjoint();
    let adj_b_ = self.b_.as_ref().map(|b_| b_.adjoint());
    let adj_x_ = self.x_.adjoint();
    let adj_y_ = match adj_b_ {
      None          => self.a_.mult(adj_x_).add(adj_a_.mult(self.x_.clone())),
      Some(adj_b_)  => self.a_.mult(adj_x_).add(adj_a_.mult_add(self.x_.clone(), adj_b_)),
    };
    adj_y_
  }*/

  default fn adjoint(&self) -> Rc<AVar<AData<W>>> {
    if self.adj.borrow().is_none() {
      *self.adj.borrow_mut() = Some(self._make_adjoint());
    }
    self.adj.borrow().as_ref().unwrap().clone()
  }
}

impl<Op, S> MultExt<Array1d<f32, S>, f32, Array1d<f32, S>, f32> for Rc<Op> where Op: 'static + AVar<AData<Array1d<f32, S>>>, S: DerefMut<Target=[f32]> {
  fn mult(&self, x_: Rc<AVar<AData<Array1d<f32, S>>>>) -> Rc<LinearOp<Array1d<f32, S>, f32, Array1d<f32, S>, f32>> {
    //let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, None, /*clk_horizon,*/ Rc::new(|_, _| 0.0_f32))
  }

  fn mult_add(&self, x_: Rc<AVar<AData<Array1d<f32, S>>>>, b_: Rc<AVar<AData<f32>>>) -> Rc<LinearOp<Array1d<f32, S>, f32, Array1d<f32, S>, f32>> {
    //let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, Some(b_), /*clk_horizon,*/ Rc::new(|_, _| 0.0_f32))
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

impl<S> AOp for LinearOp<Array1d<f32, S>, f32, Array1d<f32, S>, f32> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
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

impl<S> MultExt<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> for Rc<AVar<AData<Array2d<f32, S>>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn mult(&self, x_: Rc<AVar<AData<Array1d<f32, S>>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>>> {
    //let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), None, /*clk_horizon,*/ {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }

  fn mult_add(&self, x_: Rc<AVar<AData<Array1d<f32, S>>>>, b_: Rc<AVar<AData<Array1d<f32, S>>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>>> {
    //let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), Some(b_), /*clk_horizon,*/ {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }
}

/*impl<S> AVar<AData<Array1d<f32, S>>> for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}*/

impl<S> AOp for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
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

  /*fn _r_forward(&self, txn: TxnId) {
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
  }*/
}

impl<S> MultExt<Array2d<f32, S>, Array1d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>> for Rc<AVar<AData<Array2d<f32, S>>>> where S: 'static + DerefMut<Target=[f32]> + BatchArrayStorage<usize> {
  fn mult(&self, x_: Rc<AVar<AData<BatchArray1d<f32, S>>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>>> {
    //let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), None, /*clk_horizon,*/ {
      let x = x_.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let batch_sz = x.val.get(txn, node).batch_size();
        let buf = <S as BatchArrayStorage<usize>>::alloc(dim, batch_sz);
        BatchArray1d::from_storage(dim, batch_sz, buf)
      })
    })
  }

  fn mult_add(&self, x_: Rc<AVar<AData<BatchArray1d<f32, S>>>>, b_: Rc<AVar<AData<Array1d<f32, S>>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>>> {
    //let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_.clone(), Some(b_), /*clk_horizon,*/ {
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

/*impl<S> AVar<AData<BatchArray1d<f32, S>>> for LinearOp<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _owned_data(&self) -> &AData<A> {
    &self.data
  }
}*/

impl<S> AOp for LinearOp<Array2d<f32, S>, Array1d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
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

  /*fn _backward2(&self, txn: TxnId) {
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
  }*/
}

pub struct BroadcastAddOp<A, V> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a_:   Rc<AVar<AData<A>>>,
  x_:   Rc<AVar<AData<V>>>,
  a:    AData<A>,
  x:    AData<V>,
  y:    AData<V>,
  //kernel:   K,
  axes: Vec<usize>,
}

impl<A, V> BroadcastAddOp<A, V> {
  pub fn new(axes: Vec<usize>, a_: Rc<AVar<AData<A>>>, x_: Rc<AVar<AData<V>>>, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
    let node = NodeId::new();
    let a = a_.data();
    let x = x_.data();
    Rc::new(BroadcastAddOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      a_:   a_,
      x_:   x_,
      a:    a,
      x:    x,
      y:    AData::new(/*clk_horizon,*/ alloc.clone()),
      axes: axes,
    })
  }
}

pub trait BroadcastAddExt<A, V> {
  fn broadcast_add(&self, axes: Vec<usize>, x_: Rc<AVar<AData<V>>>) -> Rc<BroadcastAddOp<A, V>>;
}

impl<A, V> AVar<AData<V>> for BroadcastAddOp<A, V> where BroadcastAddOp<A, V>: AOp {
  default fn _owned_data(&self) -> &AData<V> {
    &self.y
  }
}

pub struct ElemLinearOp<A, V, K> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a_:   Rc<AVar<AData<A>>>,
  x_:   Rc<AVar<AData<V>>>,
  b_:   Option<Rc<AVar<AData<A>>>>,
  a:    AData<A>,
  x:    AData<V>,
  b:    Option<AData<A>>,
  y:    AData<V>,
  //tmp:  AData<V>,
  kernel:   K,
}

impl<A, V, Kernel> ElemLinearOp<A, V, Kernel> {
  pub fn new(a_: Rc<AVar<AData<A>>>, x_: Rc<AVar<AData<V>>>, b_: Option<Rc<AVar<AData<A>>>>, kernel: Kernel, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
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
      y:    AData::new(/*clk_horizon,*/ alloc.clone()),
      //tmp:  AData::new(/*clk_horizon,*/ alloc.clone()),
      kernel:   kernel,
    })
  }
}

pub struct BroadcastMultAddKernel;
pub struct ElemNormalizeKernel{pub epsilon: f64}

pub trait ElemMultExt<A, V> {
  fn elem_mult(&self, x_: Rc<AVar<AData<V>>>) -> Rc<ElemLinearOp<A, V, BroadcastMultAddKernel>>;
  fn elem_mult_add(&self, x_: Rc<AVar<AData<V>>>, b_: Rc<AVar<AData<A>>>) -> Rc<ElemLinearOp<A, V, BroadcastMultAddKernel>>;
}

/*pub trait ElemNormalizeExt<A, V> {
  fn elem_normalize(&self, mean_: Rc<AVar<AData<A>>>, var_: Rc<AVar<AData<A>>>) -> Rc<ElemLinearOp<A, V, ElemNormalizeKernel>>;
}*/

impl<A, V, Kernel> AVar<AData<V>> for ElemLinearOp<A, V, Kernel> where ElemLinearOp<A, V, Kernel>: AOp {
  default fn _owned_data(&self) -> &AData<V> {
    &self.y
  }
}

impl<S> AOp for ElemLinearOp<f32, BatchArray3d<f32, S>, BroadcastMultAddKernel> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    // TODO
    unimplemented!();
  }
}

impl<S> AOp for ElemLinearOp<Array1d<f32, S>, BatchArray3d<f32, S>, ElemNormalizeKernel> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    // TODO
    unimplemented!();
  }
}

pub struct ElemNormalizeOp<Idx, A, V> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  axes:     Idx::Axes,
  epsilon:  f64,
  x_:       Rc<AVar<AData<V>>>,
  mean_:    Rc<AVar<AData<A>>>,
  var_:     Rc<AVar<AData<A>>>,
  x:        AData<V>,
  mean:     AData<A>,
  var:      AData<A>,
  y:        AData<V>,
}

impl<Idx, A, V> ElemNormalizeOp<Idx, A, V> where Idx: ArrayIndex {
  pub fn new(axes: Idx::Axes, epsilon: f64, x_: Rc<AVar<AData<V>>>, mean_: Rc<AVar<AData<A>>>, var_: Rc<AVar<AData<A>>>, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
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
      y:        AData::new(/*clk_horizon,*/ alloc.clone()),
    })
  }
}

pub trait ElemNormalizeExt<Idx, A, V> where Idx: ArrayIndex {
  fn elem_normalize(&self, axes: Idx::Axes, epsilon: f64, mean_: Rc<AVar<AData<A>>>, var_: Rc<AVar<AData<A>>>) -> Rc<ElemNormalizeOp<Idx, A, V>>;
}

impl<Idx, A, V> AVar<AData<V>> for ElemNormalizeOp<Idx, A, V> where ElemNormalizeOp<Idx, A, V>: AOp, Idx: ArrayIndex {
  default fn _owned_data(&self) -> &AData<V> {
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
  fn conv(&self, shape: ConvShape<Idx>, x_: Rc<AVar<AData<V>>>) -> Rc<ConvOp<Idx, A, B, V, Backend>>;
  fn conv_add(&self, shape: ConvShape<Idx>, x_: Rc<AVar<AData<V>>>, b_: Rc<AVar<AData<B>>>) -> Rc<ConvOp<Idx, A, B, V, Backend>>;
}

pub struct ConvOp<Idx, A, B, V, Backend> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    ConvShape<Idx>,
  a_:   Rc<AVar<AData<A>>>,
  x_:   Rc<AVar<AData<V>>>,
  b_:   Option<Rc<AVar<AData<B>>>>,
  a:    AData<A>,
  x:    AData<V>,
  b:    Option<AData<B>>,
  y:    AData<V>,
  backend:  Backend,
}

impl<Idx, A, B, V, Backend> ConvOp<Idx, A, B, V, Backend> where Idx: ArrayIndex {
  pub fn new(shape: ConvShape<Idx>, a_: Rc<AVar<AData<A>>>, x_: Rc<AVar<AData<V>>>, b_: Option<Rc<AVar<AData<B>>>>, backend: Backend, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
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
      y:    AData::new(/*clk_horizon,*/ alloc.clone()),
      //tmp:  AData::new(1, alloc),
      backend:  backend,
    })
  }
}

impl<S> AOp for ConvOp<(usize, usize), Array4d<f32, S>, Array1d<f32, S>, Array3d<f32, S>, ()> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.a_._push(epoch, apply);
      self.x_._push(epoch, apply);
      if let Some(ref b_) = self.b_ {
        b_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
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

pub trait PoolExt<Idx, V, Backend>: AVar<AData<V>> where Idx: ArrayIndex {
  fn avg_pool(shape: PoolShape<Idx>, x_: Rc<Self>) -> Rc<PoolOp<Idx, V, AvgPool, Backend>>;
  fn max_pool(shape: PoolShape<Idx>, x_: Rc<Self>) -> Rc<PoolOp<Idx, V, MaxPool, Backend>>;
}

pub fn avg_pool<Op, Idx, V, Backend>(shape: PoolShape<Idx>, x_: Rc<Op>) -> Rc<PoolOp<Idx, V, AvgPool, Backend>> where Op: AVar<AData<V>> + PoolExt<Idx, V, Backend>, Idx: ArrayIndex {
  PoolExt::avg_pool(shape, x_)
}

pub fn max_pool<Op, Idx, V, Backend>(shape: PoolShape<Idx>, x_: Rc<Op>) -> Rc<PoolOp<Idx, V, MaxPool, Backend>> where Op: AVar<AData<V>> + PoolExt<Idx, V, Backend>, Idx: ArrayIndex {
  PoolExt::max_pool(shape, x_)
}

pub struct PoolOp<Idx, V, Kernel, Backend> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    PoolShape<Idx>,
  x_:   Rc<AVar<AData<V>>>,
  x:    AData<V>,
  y:    AData<V>,
  kernel:   Kernel,
  backend:  Backend,
}

impl<Idx, V, Kernel, Backend> PoolOp<Idx, V, Kernel, Backend> where Idx: ArrayIndex {
  pub fn new(shape: PoolShape<Idx>, x_: Rc<AVar<AData<V>>>, kernel: Kernel, backend: Backend, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(PoolOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      shape:    shape,
      x_:   x_,
      x:    x,
      y:    AData::new(/*clk_horizon,*/ alloc.clone()),
      //tmp:  AData::new(1, alloc),
      kernel:   kernel,
      backend:  backend,
    })
  }
}

impl<Idx, V, Kernel, Backend> AVar<AData<V>> for PoolOp<Idx, V, Kernel, Backend> where PoolOp<Idx, V, Kernel, Backend>: AOp, Idx: ArrayIndex {
  fn _owned_data(&self) -> &AData<V> {
    &self.y
  }
}

pub struct GenPoolOp<Idx, V, Kernel, Backend> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    PoolShape<Idx>,
  x_:   Rc<AVar<AData<V>>>,
  sel_: Rc<AVar<AData<V>>>,
  x:    AData<V>,
  sel:  AData<V>,
  y:    AData<V>,
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
  pub mode:         Rc<CopyConstant<bool>>,
  pub batch_ct:     usize,
  pub update_ct:    usize,
}

impl<Idx> BatchStatsState<Idx> where Idx: ArrayIndex {
  pub fn new(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, mode: Rc<CopyConstant<bool>>) -> Self {
    BatchStatsState{
      reduce_axes:  reduce_axes,
      cfg:          cfg,
      curr_txn:     None,
      inner_mode:   BatchStatsMode::PassThrough,
      mode:         mode,
      batch_ct:     0,
      update_ct:    0,
    }
  }

  /*pub fn get_mode(&mut self, txn: TxnId) -> BatchStatsMode {
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
  }*/
}

pub struct BatchStatsControl {
  mode: Rc<CopyConstant<bool>>,
  ops:  Vec<Rc<BatchStatsOpExt>>,
  batch_ops:    Vec<Rc<AOp>>,
  batch_vars:   VarSet,
  acc_ops:      Vec<Rc<AOp>>,
  acc_vars:     VarSet,
  fixed_ops:    Vec<Rc<AOp>>,
  fixed_vars:   VarSet,
}

impl BatchStatsControl {
  pub fn new() -> Self {
    BatchStatsControl{
      mode: Rc::new(CopyConstant{var: TxnCopyVar::new()}),
      ops:  vec![],
      batch_ops:    vec![],
      batch_vars:   var_set(),
      acc_ops:      vec![],
      acc_vars:     var_set(),
      fixed_ops:    vec![],
      fixed_vars:   var_set(),
    }
  }

  pub fn mode(&self) -> Rc<CopyConstant<bool>> {
    self.mode.clone()
  }

  pub fn set_mode(&self, txn: TxnId, mode_value: bool) {
    self.mode.var.set(txn, mode_value);
  }

  pub fn get_mode(&self, txn: TxnId) -> bool {
    self.mode.var.persist(txn);
    self.mode.var.get(txn)
  }

  /*pub fn accumulator_vars(&self) -> VarSet {
    self.acc_vars.clone()
  }*/

  pub fn dim(&mut self, txn: TxnId) -> usize {
    let mut stats_dim = 0;
    self.batch_vars.unmask_all();
    for op in self.batch_ops.iter() {
      stats_dim += op.incremental_val_size(txn, &mut self.batch_vars);
    }
    self.batch_vars.unmask_all();
    stats_dim
  }

  /*pub fn fixed_vars(&self) -> VarSet {
    self.fixed_vars.clone()
  }*/

  pub fn configure(&self, f: &Fn(&mut BatchStatsConfig)) {
    for op in self.ops.iter() {
      op._configure(f);
    }
  }

  pub fn reset_accumulators(&self, txn: TxnId) {
    for op in self.ops.iter() {
      op._reset_accumulators(txn);
    }
  }

  pub fn accumulate(&self, txn: TxnId) {
    for op in self.ops.iter() {
      op._accumulate(txn);
    }
  }

  pub fn update_stats(&self, prev_txn: TxnId, next_txn: TxnId) {
    for op in self.ops.iter() {
      op._update_stats(prev_txn, next_txn);
    }
  }

  pub fn store_accumulators(&mut self, txn: TxnId, mut offset: usize, writer: &mut Any) -> usize {
    self.acc_vars.unmask_all();
    for op in self.acc_ops.iter() {
      offset = op._store_val(txn, &mut self.acc_vars, offset, writer);
    }
    self.acc_vars.unmask_all();
    offset
  }

  pub fn load_fixed_stats(&mut self, txn: TxnId, mut offset: usize, reader: &mut Any) -> usize {
    self.fixed_vars.unmask_all();
    for op in self.fixed_ops.iter() {
      offset = op._load_val(txn, &mut self.fixed_vars, offset, reader);
    }
    self.fixed_vars.unmask_all();
    offset
  }
}

pub trait BatchStatsOpExt {
  fn _configure(&self, f: &Fn(&mut BatchStatsConfig));
  fn _reset_accumulators(&self, txn: TxnId);
  fn _accumulate(&self, txn: TxnId);
  fn _update_stats(&self, prev_txn: TxnId, next_txn: TxnId);
}

#[derive(Clone)]
pub struct BatchStatsOutput<M> {
  pub mean_batch:   Rc<AVar<AData<M>>>,
  pub var_batch:    Rc<AVar<AData<M>>>,
  pub mean_fixed:   Rc<AVar<AData<M>>>,
  pub var_fixed:    Rc<AVar<AData<M>>>,
  pub mean_branch:  Rc<BranchOp<Rc<CopyConstant<bool>>, Rc<AVar<AData<M>>>, Rc<AVar<AData<M>>>, AData<M>>>,
  //pub mean_branch:  Rc<AVar<AData<M>>>,
  pub var_branch:   Rc<BranchOp<Rc<CopyConstant<bool>>, Rc<AVar<AData<M>>>, Rc<AVar<AData<M>>>, AData<M>>>,
  //pub var_branch:   Rc<AVar<AData<M>>>,
  // TODO: expose accumulators as well for block reductions.
}

#[derive(Clone)]
pub struct BatchStatsOutputNew<M> {
  //pub mean:     Rc<BranchOp<CopyConstant<bool>, Rc<AVar<AData<M>>>, Rc<AVar<AData<M>>>, M>>,
  //pub var:      Rc<BranchOp<CopyConstant<bool>, Rc<AVar<AData<M>>>, Rc<AVar<AData<M>>>, M>>,
  pub mean:     Rc<AVar<AData<M>>>,
  pub var:      Rc<AVar<AData<M>>>,
  pub batch_mean:   VarSet,
  pub batch_var:    VarSet,
  pub running_mean: VarSet,
  pub running_var:  VarSet,
}

pub trait BatchStatsExt<Idx, A, M> where Idx: ArrayIndex {
  fn batch_stats(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Self) -> BatchStatsOutput<M> where Self: Sized;
}

pub fn batch_stats<Op, Idx, A, M>(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Rc<Op>) -> BatchStatsOutput<M> where Rc<Op>: BatchStatsExt<Idx, A, M>, Op: 'static + AVar<AData<A>>, Idx: ArrayIndex {
  <Rc<Op> as BatchStatsExt<Idx, A, M>>::batch_stats(reduce_axes, cfg, ctrl, x_)
}

pub struct BatchStatsOp<Idx, A, M> where Idx: ArrayIndex {
  node_id:      NodeId,
  stack:        OperatorStack,
  state:        RefCell<BatchStatsState<Idx>>,
  x_:           Rc<AVar<AData<A>>>,
  mean_:        Weak<PassOp<M>>,
  mean_acc_:    Rc<SrcOp<M>>,
  mean_run_:    Rc<SrcOp<M>>,
  //mean_branch_: Rc<AVar<AData<M>>>,
  var_:         Weak<PassOp<M>>,
  var_acc_:     Rc<SrcOp<M>>,
  var_run_:     Rc<SrcOp<M>>,
  //var_branch_:  Rc<AVar<AData<M>>>,
  x:            AData<A>,
  mean:         AData<M>,
  mean_acc:     AData<M>,
  mean_run:     AData<M>,
  var:          AData<M>,
  var_acc:      AData<M>,
  var_run:      AData<M>,
}

impl<Idx, A, M> BatchStatsOp<Idx, A, M> where BatchStatsOp<Idx, A, M>: AOp + BatchStatsOpExt, SrcOp<M>: AOp, Idx: 'static + ArrayIndex, A: 'static, M: 'static {
  pub fn new<Op>(reduce_axes: Idx::Axes, cfg: BatchStatsConfig, ctrl: &mut BatchStatsControl, x_: Rc<Op>, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> M>) -> BatchStatsOutput<M> where Op: 'static + AVar<AData<A>> {
    let node = NodeId::new();
    let x = x_.data();
    let mean = AData::new(/*clk_horizon,*/ alloc.clone());
    let var = AData::new(/*clk_horizon,*/ alloc.clone());
    let mean_ = PassOp::new(None, mean.clone());
    let var_ = PassOp::new(None, var.clone());
    // FIXME: some hackiness around clocks.
    let mean_acc_ = SrcOp::new(/*clk_horizon,*/ /*clk_horizon > 1,*/ alloc.clone()); //src(alloc.clone());
    let var_acc_ = SrcOp::new(/*clk_horizon,*/ /*clk_horizon > 1,*/ alloc.clone()); //src(alloc.clone());
    let mean_run_ = SrcOp::new(/*clk_horizon,*/ /*clk_horizon > 1,*/ alloc.clone());
      //.initialize(init_val(|_, x: &mut DeviceArray1d<f32>| x.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())));
    let var_run_ = SrcOp::new(/*clk_horizon,*/ /*clk_horizon > 1,*/ alloc.clone());
      //.initialize(init_val(|_, x: &mut DeviceArray1d<f32>| x.as_view_mut().set_constant(0.0, DeviceStream::implicit().conn())));
    let mean_branch_ = BranchOp::new(ctrl.mode.clone(), mean_.clone(), mean_run_.clone(), /*clk_horizon,*/ alloc.clone());
    let var_branch_ = BranchOp::new(ctrl.mode.clone(), var_.clone(), var_run_.clone(), /*clk_horizon,*/ alloc.clone());
    ctrl.batch_ops.push(mean_.clone());
    ctrl.batch_ops.push(var_.clone());
    ctrl.batch_vars.insert_all(&mean_.vars());
    ctrl.batch_vars.insert_all(&var_.vars());
    ctrl.acc_ops.push(mean_acc_.clone());
    ctrl.acc_ops.push(var_acc_.clone());
    ctrl.acc_vars.insert_all(&mean_acc_.vars());
    ctrl.acc_vars.insert_all(&var_acc_.vars());
    ctrl.fixed_ops.push(mean_run_.clone());
    ctrl.fixed_ops.push(var_run_.clone());
    ctrl.fixed_vars.insert_all(&mean_run_.vars());
    ctrl.fixed_vars.insert_all(&var_run_.vars());
    let mean_acc = mean_acc_.data();
    let mean_run = mean_run_.data();
    let var_acc = var_acc_.data();
    let var_run = var_run_.data();
    let op = Rc::new(BatchStatsOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      state:    RefCell::new(BatchStatsState::new(reduce_axes, cfg, ctrl.mode.clone())),
      x_:           x_,
      mean_:        Rc::downgrade(&mean_),
      mean_acc_:    mean_acc_,
      mean_run_:    mean_run_.clone(),
      //mean_branch_: mean_branch_.clone(),
      var_:         Rc::downgrade(&var_),
      var_acc_:     var_acc_,
      var_run_:     var_run_.clone(),
      //var_branch_:  var_branch_.clone(),
      x:            x,
      mean:         mean,
      mean_acc:     mean_acc,
      mean_run:     mean_run,
      var:          var,
      var_acc:      var_acc,
      var_run:      var_run,
    });
    *mean_.x_.borrow_mut() = Some(AOp::from(op.clone()));
    *var_.x_.borrow_mut() = Some(AOp::from(op.clone()));
    ctrl.ops.push(op);
    BatchStatsOutput{
      mean_batch:   mean_,
      var_batch:    var_,
      mean_fixed:   mean_run_,
      var_fixed:    var_run_,
      mean_branch:  mean_branch_,
      var_branch:   var_branch_,
    }
  }
}

//impl<Idx, A, M> BatchStatsOpExt for BatchStatsOp<Idx, A, M> where Idx: ArrayIndex {
impl<S> BatchStatsOpExt for BatchStatsOp<(usize, usize), BatchArray3d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _configure(&self, reconf: &Fn(&mut BatchStatsConfig)) {
    // FIXME(20170214): only safe to mutate state at the beginning of a txn.
    let mut state = self.state.borrow_mut();
    reconf(&mut state.cfg);
  }

  fn _reset_accumulators(&self, txn: TxnId) {
    let mut state = self.state.borrow_mut();
    state.batch_ct = 0;
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

impl<S> AOp for BatchStatsOp<(usize, usize), BatchArray3d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

pub trait IndexExt<IdxOp, A, Idx, Out> {
  //fn one_hot(x_: Rc<Op>, index_: Rc<IdxOp>) -> Rc<Self>;
  fn index(&self, index_: Rc<IdxOp>) -> Rc<IndexOp<A, Idx, Out>>;
}

/*pub fn one_hot<Op, IdxOp, A, Idx, Out>(x_: Rc<Op>, index_: Rc<IdxOp>) -> Rc<IndexOp<A, Idx, Out>> where Op: 'static + AVar<AData<A>>, IdxOp: 'static + AVar<AData<Idx>>, IndexOp<A, Idx, Out>: IndexExt<Op, IdxOp> {
  <IndexOp<A, Idx, Out> as IndexExt<Op, IdxOp>>::one_hot(x_, index_)
}*/

pub struct IndexOp<A, Idx, Out> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  index_:   Rc<AVar<AData<Idx>>>,
  x:        AData<A>,
  index:    AData<Idx>,
  y:        AData<Out>,
}

impl<A, Idx, Out> AVar<AData<Out>> for IndexOp<A, Idx, Out> where IndexOp<A, Idx, Out>: AOp {
  fn _owned_data(&self) -> &AData<Out> {
    &self.y
  }
}

impl<A, Idx, Out> IndexOp<A, Idx, Out> {
  pub fn new(x_: Rc<AVar<AData<A>>>, index_: Rc<AVar<AData<Idx>>>, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> Out>) -> Rc<IndexOp<A, Idx, Out>> {
    let node = NodeId::new();
    let x = x_.data();
    let index = index_.data();
    Rc::new(IndexOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      x_:       x_,
      index_:   index_,
      x:        x,
      index:    index,
      y:        AData::new(/*clk_horizon,*/ alloc),
    })
  }
}

pub struct BatchJoinOp<A, B, Join> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<AVar<AData<A>>>,
  x:    AData<A>,
  y:    AData<B>,
  kernel:   Join,
}

impl<A, B, Join> BatchJoinOp<A, B, Join> {
  pub fn new<Op>(x_: Rc<Op>, kernel: Join, /*clk_horizon: usize,*/ alloc: Rc<Fn(TxnId, NodeId) -> B>) -> Rc<BatchJoinOp<A, B, Join>> where Op: 'static + AVar<AData<A>> {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(BatchJoinOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    AData::new(/*clk_horizon,*/ alloc),
      kernel:   kernel,
    })
  }
}

pub trait BatchSumExt<Op, A, B> where Op: AVar<AData<A>> {
  fn batch_sum(x_: Rc<Op>) -> Rc<BatchJoinOp<A, B, SumJoinKernel>>;
}

pub fn batch_sum<Op, A, B>(x_: Rc<Op>) -> Rc<BatchJoinOp<A, B, SumJoinKernel>> where Rc<Op>: BatchSumExt<Op, A, B>, Op: AVar<AData<A>> {
  <Rc<Op> as BatchSumExt<Op, A, B>>::batch_sum(x_)
}

/*impl BatchSumExt<Batch<f32>, f32> for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn batch_sum(x_: Rc<AVar<AData<Batch<f32>>>>) -> Rc<Self> {
    unimplemented!();
  }
}*/

impl<A, B, Join> AVar<AData<B>> for BatchJoinOp<A, B, Join> where BatchJoinOp<A, B, Join>: AOp {
  default fn _owned_data(&self) -> &AData<B> {
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

impl AOp for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
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
  x_:   Rc<AVar<AData<A>>>,
  x:    AData<A>,
  y:    AData<B>,
  //curr_clk: Cell<usize>,
  clock:    Rc<Clock>,
  kernel:   JoinF,
}

impl AOp for SequentialJoinOp<Batch<f32>, Batch<f32>, SumJoinKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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
    let clk = self.clock.time();
    let x_val = self.x.val.get_clk(clk, txn, node);
    let batch_sz = x_val.batch_size();
    if self.y.val.accumulate(txn, node, |val| val.reshape_mut(batch_sz).set_constant(0.0)) {
      self.y.val.get_mut(txn, node).reshape_mut(batch_sz)
        .add(1.0, x_val.reshape(batch_sz));
    }
  }

  fn _backward(&self, txn: TxnId) {
    let node = self._id();
    let clk = self.clock.time();
    let y_grad = self.y.grad.get_clk(clk, txn, node);
    let batch_sz = y_grad.batch_size();
    if self.x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
      self.x.grad.get_mut(txn, node).reshape_mut(batch_sz)
        .add(1.0, y_grad.reshape(batch_sz));
    }
  }

  /*fn _reset_clock(&self) {
    self.curr_clk.set(0);
  }

  fn _set_clock(&self, new_clk: usize) {
    self.curr_clk.set(new_clk);
  }*/
}

pub fn sink<Op, A>(x_: Rc<Op>) -> Rc<ArraySink<Op, A>> where Op: AVar<AData<A>> {
  let x = x_.data();
  Rc::new(ArraySink{
    node:   NodeId::new(),
    x_:     x_,
    x:      x,
    _m:     PhantomData,
  })
}

pub struct ArraySink<Op, A> where Op: AVar<AData<A>> {
  node: NodeId,
  x_:   Rc<Op>,
  x:    AData<A>,
  _m:   PhantomData<A>,
}

/*(impl<Op, A> Deref for ArraySink<Op, A> where Op: AVar<AData<A>> {
  type Target = Op;

  fn deref(&self) -> &Op {
    &*self.x_
  }
}*/

//impl<Op> AutodiffSink<Op> for ArraySink<Op, f32> where Op: AVar<AData<f32>> {
impl<Op> AutodiffSink for ArraySink<Op, f32> where Op: AVar<AData<f32>> {
  fn _op(&self) -> &AOp {
    &*self.x_
  }

  fn _set_source(&self, txn: TxnId) {
    let node = self.node;
    if self.x.grad.overwrite(txn, node) {
      *self.x.grad.get_excl(txn, node) = 1.0;
    }
  }
}

pub struct GradientSink<A> {
  node: NodeId,
  x_:   Rc<AVar<AData<A>>>,
  x:    AData<A>,
}

impl GradientSink<f32> {
  pub fn _set_grad_sink(&self, txn: TxnId) {
    let node = self.node;
    if self.x.grad.overwrite(txn, node) {
      *self.x.grad.get_excl(txn, node) = 1.0;
    }
  }
}

impl GradientSinkExt for GradientSink<f32> {
  fn gradient(&self, txn: TxnId) {
    let epoch = Epoch::new(self.x_._id());
    self.x_._push(epoch, &mut |op| { op._forward(txn); });
    self.x_._pop(epoch, &mut |_op| {});
    self._set_grad_sink(txn);
    let epoch = Epoch::new(self.x_._id());
    self.x_._push(epoch, &mut |_op| {});
    self.x_._pop(epoch, &mut |op| { op._backward(txn); });
  }
}

pub struct GaussNewtonSink<A> {
  node: NodeId,
  x_:   Rc<AVar<AData<A>>>,
  rx_:  Rc<AVar<AData<A>>>,
  x:    AData<A>,
  rx:   AData<A>,
}

impl GaussNewtonSink<Batch<f32>> {
  pub fn _set_grad_sink(&self, txn: TxnId) {
    let node = self.node;
    if self.x.grad.overwrite(txn, node) {
      self.x.grad.get_excl(txn, node)
        .copy_from_slice(&*self.rx.val.get(txn, node));
    }
  }
}

impl GaussNewtonSinkExt for GaussNewtonSink<Batch<f32>> {
  fn gauss_newton_vector_product(&self, txn: TxnId) {
    let epoch = Epoch::new(self.x_._id());
    self.x_._push(epoch, &mut |op| { op._forward(txn); });
    self.x_._pop(epoch, &mut |_op| {});
    let epoch = Epoch::new(self.rx_._id());
    self.rx_._push(epoch, &mut |op| { op._forward(txn); });
    self.rx_._pop(epoch, &mut |_op| {});
    self._set_grad_sink(txn);
    let epoch = Epoch::new(self.x_._id());
    self.x_._push(epoch, &mut |_op| {});
    self.x_._pop(epoch, &mut |op| { op._backward(txn); });
  }
}

pub struct HessianSink<A> {
  node: NodeId,
  x_:   Rc<AVar<AData<A>>>,
  rx_:  Rc<AVar<AData<A>>>,
  x:    AData<A>,
  rx:   AData<A>,
}

impl HessianSink<f32> {
  pub fn _set_grad_sink(&self, txn: TxnId) {
    let node = self.node;
    if self.x.grad.overwrite(txn, node) {
      *self.x.grad.get_excl(txn, node) = 1.0;
    }
  }

  pub fn _set_r_grad_sink(&self, txn: TxnId) {
    let node = self.node;
    if self.rx.grad.overwrite(txn, node) {
      *self.rx.grad.get_excl(txn, node) = 0.0;
    }
  }
}

impl HessianSinkExt for HessianSink<f32> {
  fn hessian_vector_product(&self, txn: TxnId) {
    let epoch = Epoch::new(self.x_._id());
    self.x_._push(epoch, &mut |op| { op._forward(txn); });
    self.x_._pop(epoch, &mut |_op| {});
    let epoch = Epoch::new(self.rx_._id());
    self.rx_._push(epoch, &mut |op| { op._forward(txn); });
    self.rx_._pop(epoch, &mut |_op| {});
    self._set_grad_sink(txn);
    let epoch = Epoch::new(self.x_._id());
    self.x_._push(epoch, &mut |_op| {});
    self.x_._pop(epoch, &mut |op| { op._backward(txn); });
    self._set_r_grad_sink(txn);
    let epoch = Epoch::new(self.rx_._id());
    self.rx_._push(epoch, &mut |_op| {});
    self.rx_._pop(epoch, &mut |op| { op._backward(txn); });
  }
}

pub trait LstSqLossExt<Op, Target> {
  fn lst_sq_loss(huber_clip: bool, x_: Rc<Op>, target_: Rc<Target>) -> Rc<Self> where Self: 'static + Sized;
}

pub fn lst_sq_loss<Op, Target, A, Loss>(huber_clip: bool, x_: Rc<Op>, t_: Rc<Target>) -> Rc<LstSqLoss<A, Loss>> where Op: 'static + AVar<AData<A>>, Target: 'static + AVar<AData<A>>, A: 'static, Loss: 'static, LstSqLoss<A, Loss>: LstSqLossExt<Op, Target> {
  <LstSqLoss<A, Loss> as LstSqLossExt<Op, Target>>::lst_sq_loss(huber_clip, x_, t_)
}

pub struct LstSqLoss<A, Loss> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  target_:  Rc<AVar<AData<A>>>,
  x:        AData<A>,
  target:   AData<A>,
  loss:     AData<Loss>,
  clip:     bool,
}

impl<A, Loss> LstSqLoss<A, Loss> {
  pub fn new(clip: bool, x_: Rc<AVar<AData<A>>>, target_: Rc<AVar<AData<A>>>, /*clk_horizon: usize,*/ loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> Rc<LstSqLoss<A, Loss>> {
    let node = NodeId::new();
    let x = x_.data();
    let target = target_.data();
    let loss = AData::new(/*clk_horizon,*/ loss_alloc.clone());
    Rc::new(LstSqLoss{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      x_:       x_,
      target_:  target_,
      x:        x,
      target:   target,
      loss:     loss,
      clip:     clip,
    })
  }
}

impl<A, Loss> AVar<AData<Loss>> for LstSqLoss<A, Loss> where LstSqLoss<A, Loss>: AOp {
  default fn _owned_data(&self) -> &AData<Loss> {
    &self.loss
  }
}

pub struct EntropyLink;
pub struct KL2Link{pub epsilon: f64}
pub struct NLLLink;
pub struct LRLink{pub truncate: f64}

pub struct KL1LossLink;
pub struct KL2LossLink;
pub struct LRLossLink;
pub struct NLLLossLink;

pub trait LikelihoodLossLink {}

impl LikelihoodLossLink for KL1LossLink {}
impl LikelihoodLossLink for KL2LossLink {}
impl LikelihoodLossLink for LRLossLink {}
impl LikelihoodLossLink for NLLLossLink {}

pub struct SoftmaxSelfLoss<A, Loss, Link> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  x:        AData<A>,
  prob:     AData<A>,
  loss:     AData<Loss>,
  link:     Link,
  adj:      RefCell<Option<Rc<AVar<()>>>>,
}

impl<A, Loss, Link> SoftmaxSelfLoss<A, Loss, Link>
where A: 'static, Loss: 'static, Link: 'static,
      //SoftmaxSelfLoss<A, Loss, Link>: AOp,
      Self: AOp,
{
  pub fn new(x_: Rc<AVar<AData<A>>>, link: Link, /*clk_horizon: usize,*/ prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> (Rc<Self>, Rc<PassOp<A>>, Rc<PassOp<Loss>>) {
    let node = NodeId::new();
    let x = x_.data();
    let prob = AData::new(/*clk_horizon,*/ prob_alloc.clone());
    let loss = AData::new(/*clk_horizon,*/ loss_alloc.clone());
    let softmax_ = Rc::new(SoftmaxSelfLoss{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       x_,
      x:        x,
      prob:     prob.clone(),
      loss:     loss.clone(),
      link:     link,
      adj:      RefCell::new(None),
    });
    let prob_ = PassOp::new(Some(AOp::from(softmax_.clone())), prob);
    let loss_ = PassOp::new(Some(AOp::from(softmax_.clone())), loss);
    (softmax_, prob_, loss_)
  }
}

impl<A, Loss, Link> AVar<()> for SoftmaxSelfLoss<A, Loss, Link>
where A: 'static, Loss: 'static, Link: 'static,
      //SoftmaxSelfLoss<A, Loss, Link>: AOp,
      Self: AOp,
{
  default fn _owned_data(&self) -> &() {
    unreachable!();
  }

  default fn data(&self) -> () {
    ()
  }

  default fn vars(&self) -> VarSet {
    var_set()
  }

  default fn adjoint(&self) -> Rc<AVar<()>> {
    if self.adj.borrow().is_none() {
      *self.adj.borrow_mut() = Some(self._make_adjoint());
    }
    self.adj.borrow().as_ref().unwrap().clone()
  }
}

pub struct SoftmaxLoss2<A, Target, Loss, Link> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  target_:  Rc<AVar<AData<Target>>>,
  x:        AData<A>,
  target:   AData<Target>,
  prob:     AData<A>,
  loss:     AData<Loss>,
  link:     Link,
  adj:      RefCell<Option<Rc<AVar<()>>>>,
}

impl<A, Target, Loss, Link> SoftmaxLoss2<A, Target, Loss, Link>
where A: 'static, Target: 'static, Loss: 'static, Link: 'static,
      //SoftmaxLoss2<A, Target, Loss, Link>: AOp,
      Self: AOp,
{
  pub fn new(x_: Rc<AVar<AData<A>>>, target_: Rc<AVar<AData<Target>>>, link: Link, /*clk_horizon: usize,*/ prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> (Rc<Self>, Rc<PassOp<A>>, Rc<PassOp<Loss>>) {
    let node = NodeId::new();
    let x = x_.data();
    let target = target_.data();
    let prob = AData::new(/*clk_horizon,*/ prob_alloc.clone());
    let loss = AData::new(/*clk_horizon,*/ loss_alloc.clone());
    let softmax_ = Rc::new(SoftmaxLoss2{
      node_id:  node,
      stack:    OperatorStack::new(node, 2),
      x_:       x_,
      target_:  target_,
      x:        x,
      target:   target,
      prob:     prob.clone(),
      loss:     loss.clone(),
      link:     link,
      adj:      RefCell::new(None),
    });
    let prob_ = PassOp::new(Some(AOp::from(softmax_.clone())), prob);
    let loss_ = PassOp::new(Some(AOp::from(softmax_.clone())), loss);
    (softmax_, prob_, loss_)
  }
}

impl<A, Target, Loss, Link> AVar<()> for SoftmaxLoss2<A, Target, Loss, Link>
where A: 'static, Target: 'static, Loss: 'static, Link: 'static,
      //SoftmaxLoss2<A, Target, Loss, Link>: AOp,
      Self: AOp,
{
  default fn _owned_data(&self) -> &() {
    unreachable!();
  }

  default fn data(&self) -> () {
    ()
  }

  default fn vars(&self) -> VarSet {
    var_set()
  }

  default fn adjoint(&self) -> Rc<AVar<()>> {
    if self.adj.borrow().is_none() {
      *self.adj.borrow_mut() = Some(self._make_adjoint());
    }
    self.adj.borrow().as_ref().unwrap().clone()
  }
}

pub struct SoftmaxLoss3<A, T1, T2, Loss, Link> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  t1_:      Rc<AVar<AData<T1>>>,
  t2_:      Rc<AVar<AData<T2>>>,
  x:        AData<A>,
  t1:       AData<T1>,
  t2:       AData<T2>,
  prob:     AData<A>,
  loss:     AData<Loss>,
  link:     Link,
  adj:      RefCell<Option<Rc<AVar<()>>>>,
}

impl<A, T1, T2, Loss, Link> SoftmaxLoss3<A, T1, T2, Loss, Link>
where A: 'static, T1: 'static, T2: 'static, Loss: 'static, Link: 'static,
      //SoftmaxLoss3<A, T1, T2, Loss, Link>: AOp,
      Self: AOp,
{
  pub fn new(x_: Rc<AVar<AData<A>>>, t1_: Rc<AVar<AData<T1>>>, t2_: Rc<AVar<AData<T2>>>, link: Link, /*clk_horizon: usize,*/ prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> (Rc<Self>, Rc<PassOp<A>>, Rc<PassOp<Loss>>) {
    let node = NodeId::new();
    let x = x_.data();
    let t1 = t1_.data();
    let t2 = t2_.data();
    let prob = AData::new(/*clk_horizon,*/ prob_alloc.clone());
    let loss = AData::new(/*clk_horizon,*/ loss_alloc.clone());
    let softmax_ = Rc::new(SoftmaxLoss3{
      node_id:  node,
      stack:    OperatorStack::new(node, 3),
      x_:       x_,
      t1_:      t1_,
      t2_:      t2_,
      x:        x,
      t1:       t1,
      t2:       t2,
      prob:     prob.clone(),
      loss:     loss.clone(),
      link:     link,
      adj:      RefCell::new(None),
    });
    let prob_ = PassOp::new(Some(AOp::from(softmax_.clone())), prob);
    let loss_ = PassOp::new(Some(AOp::from(softmax_.clone())), loss);
    (softmax_, prob_, loss_)
  }
}

impl<A, T1, T2, Loss, Link> AVar<()> for SoftmaxLoss3<A, T1, T2, Loss, Link>
where A: 'static, T1: 'static, T2: 'static, Loss: 'static, Link: 'static,
      //SoftmaxLoss3<A, T1, T2, Loss, Link>: AOp,
      Self: AOp,
{
  default fn _owned_data(&self) -> &() {
    unreachable!();
  }

  default fn data(&self) -> () {
    ()
  }

  default fn vars(&self) -> VarSet {
    var_set()
  }

  default fn adjoint(&self) -> Rc<AVar<()>> {
    if self.adj.borrow().is_none() {
      *self.adj.borrow_mut() = Some(self._make_adjoint());
    }
    self.adj.borrow().as_ref().unwrap().clone()
  }
}

pub struct SoftmaxLoss<A, Target, Loss, Link> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AVar<AData<A>>>,
  target_:  Option<Rc<AVar<AData<Target>>>>,
  prob_:    Weak<PassOp<A>>,
  loss_:    Weak<PassOp<Loss>>,
  x:        AData<A>,
  target:   Option<AData<Target>>,
  prob:     AData<A>,
  loss:     AData<Loss>,
  /*logit:        AData<A>,
  max_logit:    AData<Loss>,
  factor:       AData<A>,
  max_factor:   AData<Loss>,*/
  link:     Link,
}

impl<A, Target, Loss, Link> SoftmaxLoss<A, Target, Loss, Link> where SoftmaxLoss<A, Target, Loss, Link>: AOp, A: 'static, Target: 'static, Loss: 'static, Link: 'static {
  //pub fn new(x_: Rc<AVar<AData<A>>>, target_: Option<Rc<AVar<AData<Target>>>>, link: Link, /*clk_horizon: usize,*/ prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> Rc<Self> {
  pub fn new<Op>(x_: Rc<Op>, target_: Option<Rc<AVar<AData<Target>>>>, link: Link, /*clk_horizon: usize,*/ prob_alloc: Rc<Fn(TxnId, NodeId) -> A>, loss_alloc: Rc<Fn(TxnId, NodeId) -> Loss>) -> (Rc<Self>, Rc<PassOp<A>>, Rc<PassOp<Loss>>) where Op: 'static + AVar<AData<A>> {
    let node = NodeId::new();
    let in_degree = match target_ {
      None      => 1,
      Some(_)   => 2,
    };
    //let x__: Rc<AOp> = ArrayOp::downgrade(x_.clone());
    let x = x_.data();
    let target = target_.as_ref().map(|t_| t_.data());
    let prob = AData::new(/*clk_horizon,*/ prob_alloc.clone());
    let loss = AData::new(/*clk_horizon,*/ loss_alloc.clone());
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
      loss:     loss, //AData::new(/*clk_horizon,*/ alloc.clone()),
      //logit:    AData::new(1, alloc.clone()),
      link:     link,
    });
    *prob_.x_.borrow_mut() = Some(AOp::from(softmax.clone()));
    *loss_.x_.borrow_mut() = Some(AOp::from(softmax.clone()));
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
  fn softmax_kl2_loss(x_: Rc<AVar<AData<A>>>, target_: Rc<AVar<AData<A>>>) -> Rc<SoftmaxLoss<A, A, L, KL2LossLink>>;
}

pub trait SoftmaxNLLLossExt<Op, A, Target, L> where Op: AVar<AData<A>> {
  //fn softmax_nll_loss(x_: Rc<Op>, target_: Rc<AVar<AData<Target>>>) -> (Rc<SoftmaxLoss<A, Target, L, NLLLossLink>>, Rc<PassOp<A>>, Rc<PassOp<L>>);
  fn softmax_nll_loss(x_: Rc<Op>, target_: Rc<AVar<AData<Target>>>) -> (Rc<PassOp<A>>, Rc<PassOp<L>>);
}

//pub fn softmax_nll_loss<Op, A, Target, L>(x_: Rc<Op>, target_: Rc<AVar<AData<Target>>>) -> (Rc<SoftmaxLoss<A, Target, L, NLLLossLink>>, Rc<PassOp<A>>, Rc<PassOp<L>>) where Rc<Op>: SoftmaxNLLLossExt<Op, A, Target, L>, Op: AVar<AData<A>> {
pub fn softmax_nll_loss<Op, A, Target, L>(x_: Rc<Op>, target_: Rc<AVar<AData<Target>>>) -> (Rc<PassOp<A>>, Rc<PassOp<L>>) where Rc<Op>: SoftmaxNLLLossExt<Op, A, Target, L>, Op: AVar<AData<A>> {
  <Rc<Op> as SoftmaxNLLLossExt<Op, A, Target, L>>::softmax_nll_loss(x_, target_)
}

/*impl<S, Target, Loss, Link> AOp for SoftmaxLoss<BatchArray1d<f32, S>, Target, Loss, Link> where S: DerefMut<Target=[f32]>, Link: LikelihoodLossLink {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      if let Some(ref target_) = self.target_ {
        target_._push(epoch, apply);
      }
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

impl<S> AOp for SoftmaxLoss<BatchArray1d<f32, S>, BatchArray1d<f32, S>, Batch<f32>, KL2LossLink> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      if let Some(ref target_) = self.target_ {
        target_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AOp for SoftmaxLoss<BatchArray1d<f32, S>, Batch<(u32, f32)>, Batch<f32>, LRLossLink> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      if let Some(ref target_) = self.target_ {
        target_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AOp for SoftmaxLoss<BatchArray1d<f32, S>, Batch<u32>, Batch<f32>, NLLLossLink> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    if 1 == self.stack.push(epoch) {
      self.x_._push(epoch, apply);
      if let Some(ref target_) = self.target_ {
        target_._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
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

  fn _backward(&self, txn: TxnId) {
    unimplemented!();
  }
}
