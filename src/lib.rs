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

#![feature(conservative_impl_trait)]
//#![feature(get_type_id)]
#![feature(slice_patterns)]
#![feature(specialization)]

extern crate async_execution;
#[cfg(feature = "cuda")] extern crate cuda;
#[cfg(feature = "cuda")] extern crate cuda_dnn;
extern crate densearray;
#[cfg(feature = "cuda")] extern crate devicemem_cuda;
extern crate fnv;
extern crate rng;

#[macro_use] extern crate lazy_static;
extern crate libc;
extern crate rand;

pub use VarKind::*;

//use arithmetic::*;
//use densearray::prelude::*;
use fnv::{FnvHashMap, FnvHashSet};

use rand::{Rng, SeedableRng, thread_rng};
use rand::chacha::{ChaChaRng};
use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
//use std::collections::{HashMap, HashSet};
use std::env;
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};
use std::sync::{Arc};

pub mod ffi;
pub mod ops;
pub mod prelude;

/*lazy_static! {
  pub static ref GLOBAL_CONFIG: GlobalConfig = GlobalConfig::default();
}*/

thread_local!(pub static GLOBAL_CONFIG: GlobalConfig = GlobalConfig::default());

thread_local!(static NODE_ID_COUNTER: Cell<u64> = Cell::new(0));
thread_local!(static TXN_ID_COUNTER:  Cell<u64> = Cell::new(0));
thread_local!(static EPOCH_COUNTER:   Cell<u64> = Cell::new(0));
thread_local!(static CLK_DOM_COUNTER: Cell<u64> = Cell::new(0));

thread_local!(static DEFAULT_CONS:    ConstructorConfig = ConstructorConfig::_default());

pub struct GlobalConfig {
  pub deterministic:    bool,
}

impl Default for GlobalConfig {
  fn default() -> Self {
    GlobalConfig{
      deterministic:    {
        env::var("ARRAYDIFF_CFG_DETERMINISTIC").ok()
          .and_then(|value| match value.parse() {
            Err(_) => {
              println!("failed to parse 'deterministic' config key: accepted values are 'false' and 'true'");
              None
            }
            Ok(x) => Some(x),
          })
          .unwrap_or(false)
      },
    }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(u64);

impl NodeId {
  pub fn new() -> NodeId {
    NODE_ID_COUNTER.with(|counter| {
      let prev_count = counter.get();
      counter.set(prev_count + 1);
      let next_count = counter.get();
      assert_eq!(next_count, prev_count + 1);
      assert!(next_count != 0);
      NodeId(next_count)
    })
  }
}

pub fn txn() -> TxnId {
  TxnId::new()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TxnId(u64);

impl TxnId {
  pub fn new() -> TxnId {
    TXN_ID_COUNTER.with(|counter| {
      let prev_count = counter.get();
      counter.set(prev_count + 1);
      let next_count = counter.get();
      assert_eq!(next_count, prev_count + 1);
      assert!(next_count != 0);
      TxnId(next_count)
    })
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EpochNr(u64);

impl EpochNr {
  pub fn new() -> EpochNr {
    EPOCH_COUNTER.with(|counter| {
      let prev_count = counter.get();
      counter.set(prev_count + 1);
      let next_count = counter.get();
      assert_eq!(next_count, prev_count + 1);
      assert!(next_count != 0);
      EpochNr(next_count)
    })
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ClockDom(u64);

impl ClockDom {
  pub fn new() -> ClockDom {
    CLK_DOM_COUNTER.with(|counter| {
      let prev_count = counter.get();
      counter.set(prev_count + 1);
      let next_count = counter.get();
      assert_eq!(next_count, prev_count + 1);
      assert!(next_count != 0);
      ClockDom(next_count)
    })
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Epoch {
  pub root:     NodeId,
  pub epoch_nr: EpochNr,
}

impl Epoch {
  pub fn new(root: NodeId) -> Epoch {
    Epoch{
      root:     root,
      epoch_nr: EpochNr::new(),
    }
  }
}

pub struct OperatorStackEntry {
  epoch:        Epoch,
  push_count:   usize,
  pop_count:    usize,
}

pub struct OperatorStack {
  node_id:      NodeId,
  in_degree:    usize,
  entries:      RefCell<Vec<OperatorStackEntry>>,
}

impl OperatorStack {
  pub fn new(node_id: NodeId, in_degree: usize) -> OperatorStack {
    OperatorStack{
      node_id:      node_id,
      in_degree:    in_degree,
      entries:      RefCell::new(vec![]),
    }
  }

  pub fn push(&self, epoch: Epoch) -> usize {
    let mut entries = self.entries.borrow_mut();
    if !entries.is_empty() && epoch == entries.last().unwrap().epoch {
      entries.last_mut().unwrap().push_count += 1;
    } else {
      entries.push(OperatorStackEntry{
        epoch:      epoch,
        push_count: 1,
        pop_count:  0,
      });
    }
    entries.last().unwrap().push_count
  }

  pub fn degree(&self, epoch: Epoch) -> usize {
    let mut entries = self.entries.borrow();
    assert!(!entries.is_empty());
    let level = entries.len() - 1;
    assert_eq!(epoch, entries[level].epoch);
    entries[level].push_count
  }

  pub fn pop(&self, epoch: Epoch) -> usize {
    let mut entries = self.entries.borrow_mut();
    assert!(!entries.is_empty());
    let level = entries.len() - 1;
    assert_eq!(epoch, entries[level].epoch);
    entries[level].pop_count += 1;
    let saved_pop_count = entries[level].pop_count;
    if entries[level].push_count == entries[level].pop_count {
      entries.pop();
    }
    saved_pop_count
  }
}

pub struct Clock {
  dom:      ClockDom,
  horizon:  usize,
  time:     Cell<usize>,
}

impl Clock {
  pub fn new(horizon: usize) -> Rc<Clock> {
    Rc::new(Clock{
      dom:      ClockDom::new(),
      horizon:  horizon,
      time:     Cell::new(0),
    })
  }

  pub fn horizon(&self) -> usize {
    self.horizon
  }

  pub fn reset_time(&self) {
    self.time.set(0);
  }

  pub fn set_time(&self, new_time: usize) {
    assert!(new_time < self.horizon);
    self.time.set(new_time);
  }

  pub fn time(&self) -> usize {
    self.time.get()
  }
}

#[derive(Clone)]
pub struct ConstructorConfig {
  pub clock:    Rc<Clock>,
}

impl ConstructorConfig {
  fn _default() -> Self {
    ConstructorConfig{
      clock:    Clock::new(1),
    }
  }
}

pub struct ConstructorStack {
  stack:    Vec<ConstructorConfig>,
}

impl ConstructorStack {
  pub fn current(&self) -> ConstructorConfig {
    match self.stack.len() {
      0 => DEFAULT_CONS.with(|cfg| cfg.clone()),
      _ => self.stack[self.stack.len() - 1].clone()
    }
  }

  pub fn push(&mut self, cfg: ConstructorConfig) {
    self.stack.push(cfg);
  }

  pub fn pop(&mut self) -> Option<ConstructorConfig> {
    self.stack.pop()
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symbol {
  pub node_id:  NodeId,
}

impl Symbol {
  pub fn new() -> Symbol {
    Symbol{node_id: NodeId::new()}
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarKind {
  Val,
  Grad,
  RVal,
  RGrad,
  Val2,
  Grad2,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Var {
  pub node_id:  NodeId,
  pub kind:     VarKind,
}

impl Var {
  pub fn new(kind: VarKind) -> Var {
    Var{
      node_id:  NodeId::new(),
      kind:     kind,
    }
  }

  pub fn singleton(&self) -> VarSet {
    VarSet::empty().add(self.clone())
  }
}

pub fn var_set() -> VarSet {
  VarSet::empty()
}

#[derive(Clone)]
pub struct VarSetMask {
  mask:     FnvHashSet<Var>,
}

#[derive(Clone)]
pub struct VarSet {
  inner:    FnvHashSet<Var>,
  mask:     FnvHashSet<Var>,
}

impl VarSet {
  pub fn empty() -> VarSet {
    VarSet{
      inner:    FnvHashSet::default(),
      mask:     FnvHashSet::default(),
    }
  }

  pub fn insert_all(&mut self, other_vars: &VarSet) {
    for var in other_vars.inner.iter() {
      self.inner.insert(var.clone());
    }
  }

  pub fn add(mut self, var: Var) -> Self {
    self.inner.insert(var);
    self
  }

  pub fn union(mut self, other_vars: VarSet) -> Self {
    for var in other_vars.inner.iter() {
      self.inner.insert(var.clone());
    }
    self
  }

  pub fn filter<F>(&self, f: F) -> Self where F: Fn(&Var) -> bool {
    let mut new_vars = VarSet::empty();
    for var in self.inner.iter() {
      if f(var) {
        new_vars = new_vars.add(var.clone());
      }
    }
    new_vars
  }

  pub fn clear(&mut self) {
    self.inner.clear();
    self.mask.clear();
  }

  pub fn contains(&mut self, var: &Var) -> bool {
    self.inner.contains(var)
  }

  pub fn unmask_all(&mut self) {
    self.mask.clear();
  }

  pub fn mask(&mut self, var: Var) -> bool {
    let contained = self.inner.contains(&var);
    if contained {
      let masked = self.mask.contains(&var);
      self.mask.insert(var);
      !masked
    } else {
      false
    }
  }

  pub fn is_unmasked(&mut self, var: &Var) -> bool {
    !self.mask.contains(var)
  }
}

pub trait AOp {
  fn _id(&self) -> NodeId;
  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp));
  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp));
  fn _traverse_fwd(&self, apply: &mut FnMut(&AOp)) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, apply);
    self._pop(epoch, &mut |_op| {});
  }
  fn _traverse_bwd(&self, apply: &mut FnMut(&AOp)) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, apply);
  }

  //fn _serial_size(&self, _txn: TxnId, _vars: &mut VarSet) -> usize { unimplemented!(); }
  fn _copy_val(&self, _dst_txn: TxnId, _dst_vars: &mut VarSet, _src_txn: TxnId, _src_vars: &mut VarSet, offset: usize, _src: &AOp) -> usize { offset }
  fn _load_val(&self, _txn: TxnId, _vars: &mut VarSet, offset: usize, _reader: &mut Any) -> usize { offset }
  fn _load_r_val(&self, _txn: TxnId, _vars: &mut VarSet, offset: usize, _reader: &mut Any) -> usize { offset }
  fn _store_val(&self, _txn: TxnId, _vars: &mut VarSet, offset: usize, _writer: &mut Any) -> usize { offset }
  fn _store_grad(&self, _txn: TxnId, _vars: &mut VarSet, offset: usize, _writer: &mut Any) -> usize { offset }
  fn _store_r_grad(&self, _txn: TxnId, _vars: &mut VarSet, offset: usize, _writer: &mut Any) -> usize { offset }
  fn _store_grad2(&self, _txn: TxnId, _vars: &mut VarSet, offset: usize, _writer: &mut Any) -> usize { offset }
  fn _persist(&self, _txn: TxnId, _vars: &mut VarSet) {}

  fn _init(&self, _txn: TxnId, _seed_rng: Rc<RefCell<ChaChaRng>>) {}
  fn _forward(&self, txn: TxnId);
  fn _backward(&self, _txn: TxnId) { unimplemented!(); }
  fn _backward_store_grad(&self, _txn: TxnId, _vars: &mut VarSet, _offset: usize, _writer: &mut Any) -> usize { unimplemented!(); }
  /*fn _r_forward(&self, _txn: TxnId) { unimplemented!(); }
  fn _r_backward(&self, _txn: TxnId) { unimplemented!(); }
  fn _backward2(&self, _txn: TxnId) { unimplemented!(); }*/

  fn _reset_clock(&self) {}
  fn _set_clock(&self, _clk: usize) { unimplemented!(); }

  fn from(op: Rc<Self>) -> Rc<AOp> where Self: 'static + Sized { op }
  fn from_shared(op: Arc<Self>) -> Arc<AOp> where Self: 'static + Sized { op }
  fn from_owned(op: Box<Self>) -> Box<AOp> where Self: 'static + Sized { op }

  /*fn serial_size(&self, txn: TxnId, vars: &mut VarSet) -> usize {
    let epoch = Epoch::new(self._id());
    let mut size = 0;
    vars.unmask_all();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { size += op._serial_size(txn, vars); });
    vars.unmask_all();
    size
  }*/

  fn incremental_val_size(&self, txn: TxnId, vars: &mut VarSet) -> usize {
    let epoch = Epoch::new(self._id());
    let mut offset = 0;
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._store_val(txn, vars, offset, &mut NullIo);
    });
    offset
  }

  fn val_size(&self, txn: TxnId, vars: &mut VarSet) -> usize {
    let epoch = Epoch::new(self._id());
    let mut offset = 0;
    vars.unmask_all();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._store_val(txn, vars, offset, &mut NullIo);
    });
    vars.unmask_all();
    offset
  }

  fn max_val_delta(&self, txn: TxnId, vars: &mut VarSet) -> Option<usize> {
    let epoch = Epoch::new(self._id());
    let mut offset = 0;
    let mut max_size = None;
    vars.unmask_all();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      let next_offset = op._store_val(txn, vars, offset, &mut NullIo);
      let delta = next_offset - offset;
      match max_size {
        None => max_size = Some(delta),
        Some(prev_max_size) => if prev_max_size < delta {
          max_size = Some(delta);
        },
      }
      offset = next_offset;
    });
    vars.unmask_all();
    max_size
  }

  fn load_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut Any) -> usize {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    //reader.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._load_val(txn, vars, offset, reader);
    });
    vars.unmask_all();
    offset
  }

  /*fn load_r_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, reader: &mut SerialIoBuf) -> usize {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    reader.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._load_r_val(txn, vars, offset, reader.as_any());
    });
    vars.unmask_all();
    offset
  }*/

  fn store_val(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    //writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._store_val(txn, vars, offset, writer);
    });
    vars.unmask_all();
    offset
  }

  fn store_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut Any) -> usize {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    //writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._store_grad(txn, vars, offset, writer);
    });
    vars.unmask_all();
    offset
  }

  /*fn store_r_grad(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut SerialIoBuf) -> usize {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._store_r_grad(txn, vars, offset, writer.as_any());
    });
    vars.unmask_all();
    offset
  }*/

  /*fn store_grad2(&self, txn: TxnId, vars: &mut VarSet, mut offset: usize, writer: &mut SerialIoBuf) -> usize {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| {
      offset = op._store_grad2(txn, vars, offset, writer.as_any());
    });
    vars.unmask_all();
    offset
  }*/

  fn init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, {
      let seed_rng = seed_rng.clone();
      &mut move |op| { op._init(txn, seed_rng.clone()); }
    });
  }

  fn persist(&self, txn: TxnId, vars: &mut VarSet) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._persist(txn, vars); });
    vars.unmask_all();
  }

  fn reset_clock(&self) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._reset_clock(); });
    self._pop(epoch, &mut |_op| {});
  }

  fn set_clock(&self, clk: usize) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._set_clock(clk); });
    self._pop(epoch, &mut |_op| {});
  }

  fn eval(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
  }
}

pub trait GradientSinkExt {
  fn gradient(&self, txn: TxnId);
}

pub trait GaussNewtonSinkExt {
  fn gauss_newton_vector_product(&self, txn: TxnId);
}

pub trait HessianSinkExt {
  fn hessian_vector_product(&self, txn: TxnId);
}

//pub trait AutodiffSink<Op>: Deref<Target=Op> where Op: AOp {
pub trait AutodiffSink : AOp {
  fn _op(&self) -> &AOp;
  fn _set_source(&self, txn: TxnId);

  fn gradient(&self, txn: TxnId) {
    let epoch = Epoch::new(self._op()._id());
    self._set_source(txn);
    self._op()._push(epoch, &mut |op| { op._forward(txn); });
    self._op()._pop(epoch, &mut |_op| {});
    self._op()._push(epoch, &mut |_op| {});
    self._op()._pop(epoch, &mut |op| { op._backward(txn); });
  }

  /*fn gauss_newton_vector_product(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._set_source(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |op| { op._r_forward(txn, true); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, true); });
  }

  fn hessian_vector_product(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._set_source(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, false); });
    self._push(epoch, &mut |op| { op._r_forward(txn, false); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._r_backward(txn); });
  }

  fn hessian_diagonal(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._set_source(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, false); });
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward2(txn); });
  }*/
}

impl<Op> AOp for Op where Op: AutodiffSink {
  default fn _id(&self) -> NodeId {
    self._op()._id()
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    self._op()._push(epoch, apply);
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AOp)) {
    self._op()._pop(epoch, apply);
  }

  default fn _persist(&self, txn: TxnId, vars: &mut VarSet) {
    self._op()._persist(txn, vars);
  }

  default fn _init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {
    self._op()._init(txn, seed_rng);
  }

  default fn _forward(&self, txn: TxnId) {
    self._op()._forward(txn);
  }

  default fn _backward(&self, txn: TxnId) {
    self._op()._backward(txn);
  }

  /*default fn _r_forward(&self, txn: TxnId) {
    self._op()._r_forward(txn);
  }

  default fn _r_backward(&self, txn: TxnId) {
    self._op()._r_backward(txn);
  }

  default fn _backward2(&self, txn: TxnId) {
    self._op()._backward2(txn);
  }*/
}

pub trait AVarOutput: Clone {
  fn _vars(&self) -> VarSet;
}

impl AVarOutput for () {
  fn _vars(&self) -> VarSet {
    var_set()
  }
}

pub trait AVar<Out>: AOp where Out: AVarOutput {
  fn _owned_data(&self) -> &Out;
  fn _make_adjoint(&self) -> Rc<AVar<Out>> { unimplemented!(); }

  fn from(op: Rc<Self>) -> Rc<AVar<Out>> where Self: 'static + Sized { op }
  fn from_shared(op: Arc<Self>) -> Arc<AVar<Out>> where Self: 'static + Sized { op }
  fn from_owned(op: Box<Self>) -> Box<AVar<Out>> where Self: 'static + Sized { op }

  fn data(&self) -> Out { self._owned_data().clone() }
  fn vars(&self) -> VarSet { self._owned_data()._vars() }
  fn adjoint(&self) -> Rc<AVar<Out>> { unimplemented!(); }
}

/*impl<Op> AVar<()> for Op where Op: AVar<()> {
  default fn _owned_data(&self) -> &() { unreachable!(); }
  default fn data(&self) -> () { () }
  default fn vars(&self) -> VarSet { var_set() }
}*/

// Requires working trait aliases.
// See rust-lang issue #41517 and RFC #1733 for details.
/*pub trait ArrayOp<A> = AVar<Data=ArrayData<A>>;*/

pub struct NullIo;
pub struct ZeroIo;
pub struct MaxCapacityIo;

/*pub trait SerialIoBuf: Any {
  fn reset(&mut self);
  fn as_any(&mut self) -> &mut Any;
}

impl SerialIoBuf for ZeroIo {
  fn reset(&mut self) {
  }

  fn as_any(&mut self) -> &mut Any {
    self
  }
}

pub trait CursorIoBufExt<'a> {
  type Ref: ?Sized;
  type Mut: ?Sized;

  fn read_buf(&'a mut self, length: usize) -> Self::Ref;
  fn write_buf(&'a mut self, length: usize) -> Self::Mut;
}

pub struct CursorIoBuf<A> {
  buffer:   A,
  offset:   usize,
}

impl<A> CursorIoBuf<A> {
  pub fn new(buffer: A) -> Self {
    CursorIoBuf{
      buffer:   buffer,
      offset:   0,
    }
  }
}

impl<A> Deref for CursorIoBuf<A> {
  type Target = A;

  fn deref(&self) -> &A {
    &self.buffer
  }
}

impl<A> DerefMut for CursorIoBuf<A> {
  fn deref_mut(&mut self) -> &mut A {
    &mut self.buffer
  }
}

impl<A> SerialIoBuf for CursorIoBuf<A> where A: 'static {
  fn reset(&mut self) {
    self.offset = 0;
  }

  fn as_any(&mut self) -> &mut Any {
    self
  }
}

impl<'a, T> CursorIoBufExt<'a> for CursorIoBuf<Vec<T>> where T: 'a {
  type Ref = &'a [T];
  type Mut = &'a mut [T];

  fn read_buf(&'a mut self, length: usize) -> &'a [T] {
    let buf = &self.buffer[self.offset .. self.offset + length];
    self.offset += length;
    buf
  }

  fn write_buf(&'a mut self, length: usize) -> &'a mut [T] {
    let buf = &mut self.buffer[self.offset .. self.offset + length];
    self.offset += length;
    buf
  }
}*/

pub trait ArrayStorage<Idx> {
  fn alloc(dim: Idx) -> Self where Self: Sized;
}

impl ArrayStorage<usize> for Vec<f32> {
  fn alloc(dim: usize) -> Self {
    let mut buf = Vec::with_capacity(dim);
    buf.resize(dim, 0.0);
    buf
  }
}

pub trait BatchArrayStorage<Idx> {
  fn alloc(dim: Idx, batch_sz: usize) -> Self where Self: Sized;
}

impl BatchArrayStorage<usize> for Vec<f32> {
  fn alloc(dim: usize, batch_sz: usize) -> Self {
    let mut buf = Vec::with_capacity(dim * batch_sz);
    buf.resize(dim * batch_sz, 0.0);
    buf
  }
}

pub struct TxnCopyVar<A> where A: Copy {
  // FIXME: clocked values.
  curr_txn: Cell<Option<TxnId>>,
  value:    Cell<Option<A>>,
}

impl<A> TxnCopyVar<A> where A: Copy {
  pub fn new() -> Self {
    TxnCopyVar{
      curr_txn: Cell::new(None),
      value:    Cell::new(None),
    }
  }

  pub fn persist(&self, txn: TxnId) {
    match self.curr_txn.get() {
      None => {
        self.curr_txn.set(Some(txn));
        self.value.set(None);
      }
      Some(_) => {
        self.curr_txn.set(Some(txn));
      }
    }
  }

  pub fn get(&self, txn: TxnId) -> A {
    match self.curr_txn.get() {
      None => {
        panic!("the `TxnCopyVar` is invalid");
      }
      Some(prev_txn) => {
        assert_eq!(prev_txn, txn);
        match self.value.get() {
          None => panic!("the `TxnCopyVar` is valid but is missing a value"),
          Some(v) => v,
        }
      }
    }
  }

  pub fn set(&self, txn: TxnId, new_value: A) {
    match self.curr_txn.get() {
      None => {
        self.curr_txn.set(Some(txn));
        self.value.set(Some(new_value));
      }
      Some(prev_txn) => {
        if prev_txn != txn {
          self.curr_txn.set(Some(txn));
          self.value.set(Some(new_value));
        } else {
          // TODO: should panic here?
          panic!("trying to set a `TxnCopyVar` twice in the same txn");
        }
      }
    }
  }
}

pub struct TxnVarClkBuf<A> {
  _clk:         usize,
  curr_txn:     Cell<Option<TxnId>>,
  rollover:     Cell<bool>,
  reads:        RefCell<FnvHashSet<(NodeId, usize)>>,
  freezes:      RefCell<FnvHashMap<(NodeId, usize), Symbol>>,
  writes:       RefCell<FnvHashMap<(NodeId, usize), Symbol>>,
  read_writes:  RefCell<FnvHashSet<(NodeId, usize, Symbol)>>,
  coarse_rws:   RefCell<FnvHashSet<(NodeId, usize)>>,
  buffer:       RefCell<Option<A>>,
}

impl<A> TxnVarClkBuf<A> {
  pub fn new(clk: usize) -> Self {
    TxnVarClkBuf{
      _clk:         clk,
      curr_txn:     Cell::new(None),
      rollover:     Cell::new(false),
      reads:        RefCell::new(FnvHashSet::default()),
      freezes:      RefCell::new(FnvHashMap::default()),
      writes:       RefCell::new(FnvHashMap::default()),
      read_writes:  RefCell::new(FnvHashSet::default()),
      coarse_rws:   RefCell::new(FnvHashSet::default()),
      buffer:       RefCell::new(None),
    }
  }
}

pub struct TxnVarBuf<A> {
  curr_txn:     Cell<Option<TxnId>>,
  rollover:     Cell<bool>,
  reads:        RefCell<FnvHashSet<NodeId>>,
  freezes:      RefCell<FnvHashSet<NodeId>>,
  writes:       RefCell<FnvHashMap<NodeId, Symbol>>,
  read_writes:  RefCell<FnvHashSet<(NodeId, Symbol)>>,
  coarse_rws:   RefCell<FnvHashSet<NodeId>>,
  buffer:       RefCell<Option<A>>,
}

impl<A> TxnVarBuf<A> {
  pub fn new() -> Self {
    TxnVarBuf{
      curr_txn:     Cell::new(None),
      rollover:     Cell::new(false),
      reads:        RefCell::new(FnvHashSet::default()),
      freezes:      RefCell::new(FnvHashSet::default()),
      writes:       RefCell::new(FnvHashMap::default()),
      read_writes:  RefCell::new(FnvHashSet::default()),
      coarse_rws:   RefCell::new(FnvHashSet::default()),
      buffer:       RefCell::new(None),
    }
  }
}

pub struct TxnVar<A> {
  symbol:   Symbol,
  var:      Var,
  alloc:    Rc<Fn(TxnId, NodeId) -> A>,
  curr_clk: Rc<Cell<usize>>,
  clk_bufs: Vec<Rc<TxnVarBuf<A>>>,
  //clk_bufs: Vec<Rc<TxnVarClkBuf<A>>>,
}

impl<A> TxnVar<A> {
  pub fn new(symbol: Symbol, kind: VarKind, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Self {
    let mut clk_bufs = Vec::with_capacity(clk_horizon);
    for _ in 0 .. clk_horizon {
      clk_bufs.push(Rc::new(TxnVarBuf::new()));
    }
    TxnVar{
      symbol:   symbol,
      var:      Var::new(kind),
      alloc:    alloc,
      curr_clk: Rc::new(Cell::new(0)),
      clk_bufs: clk_bufs,
    }
  }

  /// Clone this variable "by reference," but also assign it a unique symbol.
  /// Each instance of the same `TxnVar` should have a unique symbol in order
  /// to distinguish different operands.
  pub fn dup(&self, new_symbol: Symbol) -> Self {
    TxnVar{
      symbol:   new_symbol,
      var:      self.var.clone(),
      alloc:    self.alloc.clone(),
      curr_clk: self.curr_clk.clone(),
      clk_bufs: self.clk_bufs.clone(),
    }
  }

  pub fn var(&self) -> Var {
    self.var.clone()
  }

  pub fn reset_clock(&self) {
    self.curr_clk.set(0);
  }

  pub fn set_clock(&self, clk: usize) {
    self.curr_clk.set(clk);
  }

  /// Invalidate this variable.
  pub fn invalidate(&self) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    buf.curr_txn.set(None);
    buf.rollover.set(false);
    buf.reads.borrow_mut().clear();
    buf.freezes.borrow_mut().clear();
    buf.writes.borrow_mut().clear();
    buf.read_writes.borrow_mut().clear();
    buf.coarse_rws.borrow_mut().clear();
  }

  /// Rollover this variable to a new transaction if this variable is
  /// a member of the provided variable set.
  pub fn rollover(&self, txn: TxnId, vars: &mut VarSet) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    if vars.contains(&self.var) {
      match buf.curr_txn.get() {
        Some(prev_txn) => {
          if prev_txn == txn {
            // Do nothing.
          } else {
            buf.rollover.set(true);
            /*buf.curr_txn.set(Some(txn));
            buf.reads.borrow_mut().clear();
            buf.writes.borrow_mut().clear();
            buf.read_writes.borrow_mut().clear();
            buf.coarse_rws.borrow_mut().clear();*/
          }
        }
        None => {
          // Do nothing.
          /*//buf.rollover.set(true);
          buf.curr_txn.set(Some(txn));
          buf.reads.borrow_mut().clear();
          buf.writes.borrow_mut().clear();
          buf.read_writes.borrow_mut().clear();
          buf.coarse_rws.borrow_mut().clear();*/
        }
      }
    } else {
      /*buf.curr_txn.set(None);
      buf.rollover.set(false);
      buf.reads.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();*/
    }
  }

  /// Query this variable's availability to be overwritten,
  /// i.e. exclusive write. See `.get_excl()` for details.
  pub fn overwrite(&self, txn: TxnId, node: NodeId) -> bool {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    let mut incomplete_write = false;
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      incomplete_write = true;
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      if curr_txn != txn {
        incomplete_write = true;
        new_txn = true;
      } else {
        assert!(!buf.reads.borrow().contains(&node));
        assert!(!buf.freezes.borrow().contains(&node));
        assert!(!buf.coarse_rws.borrow().contains(&node));
        let written = buf.writes.borrow().contains_key(&node);
        if written {
          assert_eq!(1, buf.writes.borrow().len());
          assert_eq!(self.symbol, *buf.writes.borrow().get(&node).unwrap());
        }
        incomplete_write = !written;
      }
    }
    if new_txn || buf.rollover.get() {
      assert!(incomplete_write);
      buf.curr_txn.set(Some(txn));
      buf.reads.borrow_mut().clear();
      buf.freezes.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
      let mut buffer = buf.buffer.borrow_mut();
      if buffer.is_none() {
        *buffer = Some((self.alloc)(txn, node));
      }
      buf.rollover.set(false);
    }
    incomplete_write
  }

  /// Query this variable's availability for accumulation,
  /// i.e. read-write. See `.get_mut()` for details.
  pub fn accumulate<F>(&self, txn: TxnId, node: NodeId, init: F) -> bool where F: Fn(&mut A) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    let mut incomplete_write = false;
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      incomplete_write = true;
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      if curr_txn != txn {
        incomplete_write = true;
        new_txn = true;
      } else {
        assert!(!buf.reads.borrow().contains(&node));
        assert!(!buf.writes.borrow().contains_key(&node));
        let rw = buf.read_writes.borrow().contains(&(node, self.symbol));
        let coarse_rw = buf.coarse_rws.borrow().contains(&node);
        if !coarse_rw {
          assert!(!rw);
        }
        if rw {
          assert!(coarse_rw);
        }
        incomplete_write = !rw;
      }
    }
    if new_txn || buf.rollover.get() {
      assert!(incomplete_write);
      buf.curr_txn.set(Some(txn));
      buf.reads.borrow_mut().clear();
      buf.freezes.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
      let mut buffer = buf.buffer.borrow_mut();
      if buffer.is_none() {
        *buffer = Some((self.alloc)(txn, node));
      }
      if !buf.rollover.get() {
        init(&mut *buffer.as_mut().unwrap());
      }
      buf.rollover.set(false);
    }
    incomplete_write
  }

  pub fn get(&self, txn: TxnId, node: NodeId) -> Ref<A> {
    let clk = self.curr_clk.get();
    self.get_clk(clk, txn, node)
  }

  pub fn get_prev(&self, txn: TxnId, node: NodeId) -> Option<Ref<A>> {
    let clk = self.curr_clk.get();
    match clk {
      0   => None,
      clk => Some(self.get_clk(clk - 1, txn, node)),
    }
  }

  /// Reads satisfy the following transactional rules:
  /// - A read on a variable is mutually exclusive with read-writes
  ///   and exclusive writes.
  pub fn get_clk(&self, clk: usize, txn: TxnId, node: NodeId) -> Ref<A> {
    let buf = &self.clk_bufs[clk];
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      new_txn = curr_txn != txn;
    }
    // FIXME(20170216): may need to record the current clock in
    // read/write events.
    if new_txn {
      assert!(buf.rollover.get());
      buf.curr_txn.set(Some(txn));
      buf.reads.borrow_mut().clear();
      buf.freezes.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
      buf.rollover.set(false);
    }
    //assert!(!buf.writes.borrow().contains_key(&node));
    if buf.writes.borrow().contains_key(&node) {
      if !buf.freezes.borrow().contains(&node) {
        buf.freezes.borrow_mut().insert(node);
      }
    }
    assert!(!buf.coarse_rws.borrow().contains(&node));
    buf.reads.borrow_mut().insert(node);
    Ref::map(buf.buffer.borrow(), |buffer| {
      if let Some(buffer) = buffer.as_ref() {
        buffer
      } else {
        panic!("trying to read from unallocated buffer");
      }
    })
  }

  /// Exclusive writes satisfy the following transaction rules:
  /// - An exclusive write on a variable is mutually exclusive with
  ///   reads and read-writes.
  /// - An exclusive write can be performed on a variable by only one
  ///   operand symbol; attempting to exclusively write to the same
  ///   variable using two different symbols is illegal.
  pub fn get_excl(&self, txn: TxnId, node: NodeId) -> RefMut<A> {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      new_txn = curr_txn != txn;
    }
    /*if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.reads.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
    }*/
    assert!(!new_txn);
    assert!(!buf.rollover.get());
    assert!(!buf.reads.borrow().contains(&node));
    assert!(!buf.freezes.borrow().contains(&node));
    assert!(!buf.coarse_rws.borrow().contains(&node));
    if buf.writes.borrow().contains_key(&node) {
      assert_eq!(1, buf.writes.borrow().len());
      assert_eq!(self.symbol, *buf.writes.borrow().get(&node).unwrap());
    } else {
      buf.writes.borrow_mut().insert(node, self.symbol);
    }
    RefMut::map(buf.buffer.borrow_mut(), |buf| {
      if let Some(buf) = buf.as_mut() {
        buf
      } else {
        panic!("trying to write to unallocated buffer");
      }
    })
  }

  /// Read-writes satisfy the following transactional rules:
  /// - A read-write on a variable is mutually exclusive with
  ///   reads and exclusive writes.
  pub fn get_mut(&self, txn: TxnId, node: NodeId) -> RefMut<A> {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      new_txn = curr_txn != txn;
    }
    /*if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.reads.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
    }*/
    assert!(!new_txn);
    assert!(!buf.rollover.get());
    assert!(!buf.reads.borrow().contains(&node));
    assert!(!buf.freezes.borrow().contains(&node));
    assert!(!buf.writes.borrow().contains_key(&node));
    let rw = buf.read_writes.borrow().contains(&(node, self.symbol));
    let coarse_rw = buf.coarse_rws.borrow().contains(&node);
    if !coarse_rw {
      assert!(!rw);
    }
    if rw {
      assert!(coarse_rw);
    } else {
      buf.read_writes.borrow_mut().insert((node, self.symbol));
      buf.coarse_rws.borrow_mut().insert(node);
    }
    RefMut::map(buf.buffer.borrow_mut(), |buf| {
      if let Some(buf) = buf.as_mut() {
        buf
      } else {
        panic!("trying to read-write to unallocated buffer");
      }
    })
  }
}

pub struct AConst<A> {
  pub val:      A,
}

pub type ArrayData<A> = AData<A>;

pub struct AData<A> {
  symbol:       Symbol,
  clk_horizon:  usize,
  //alloc:        Rc<Fn(TxnId, NodeId) -> A>,
  pub val:      TxnVar<A>,
  pub grad:     TxnVar<A>,
  pub r_val:    TxnVar<A>,
  pub r_grad:   TxnVar<A>,
  pub val2:     TxnVar<A>,
  pub grad2:    TxnVar<A>,
}

impl<A> Clone for AData<A> {
  fn clone(&self) -> Self {
    let new_symbol = Symbol::new();
    AData{
      symbol:       new_symbol,
      clk_horizon:  self.clk_horizon,
      val:          self.val.dup(new_symbol),
      grad:         self.grad.dup(new_symbol),
      r_val:        self.r_val.dup(new_symbol),
      r_grad:       self.r_grad.dup(new_symbol),
      val2:         self.grad2.dup(new_symbol),
      grad2:        self.grad2.dup(new_symbol),
    }
  }
}

impl<A> AVarOutput for AData<A> {
  fn _vars(&self) -> VarSet {
    VarSet::empty()
      .add(self.val.var())
      .add(self.grad.var())
      /*.add(self.r_val.var())
      .add(self.r_grad.var())
      .add(self.val2.var())
      .add(self.grad2.var())*/
  }
}

impl<A> AData<A> {
  pub fn new(clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Self {
    let symbol = Symbol::new();
    AData{
      symbol:       symbol,
      clk_horizon:  clk_horizon,
      val:      TxnVar::new(symbol, Val,    clk_horizon, alloc.clone()),
      grad:     TxnVar::new(symbol, Grad,   clk_horizon, alloc.clone()),
      r_val:    TxnVar::new(symbol, RVal,   clk_horizon, alloc.clone()),
      r_grad:   TxnVar::new(symbol, RGrad,  clk_horizon, alloc.clone()),
      val2:     TxnVar::new(symbol, Grad2,  clk_horizon, alloc.clone()),
      grad2:    TxnVar::new(symbol, Grad2,  clk_horizon, alloc.clone()),
    }
  }

  pub fn rollover_all(&self, txn: TxnId, vars: &mut VarSet) {
    self.val.rollover(txn, vars);
    self.grad.rollover(txn, vars);
    /*self.r_val.rollover(txn, vars);
    self.r_grad.rollover(txn, vars);
    self.val2.rollover(txn, vars);
    self.grad2.rollover(txn, vars);*/
  }

  pub fn horizon(&self) -> usize {
    self.clk_horizon
  }

  pub fn reset_clock_all(&self) {
    self.val.reset_clock();
    self.grad.reset_clock();
    /*self.r_val.reset_clock();
    self.r_grad.reset_clock();
    self.val2.reset_clock();
    self.grad2.reset_clock();*/
  }

  pub fn set_clock_all(&self, clk: usize) {
    self.val.set_clock(clk);
    self.grad.set_clock(clk);
    /*self.r_val.set_clock(clk);
    self.r_grad.set_clock(clk);
    self.val2.set_clock(clk);
    self.grad2.set_clock(clk);*/
  }
}

pub fn master_rng() -> (ChaChaRng, Vec<u32>) {
  let mut seed = Vec::with_capacity(8);
  for _ in 0 .. 8 {
    seed.push(thread_rng().next_u32());
  }
  assert_eq!(8, seed.len());
  let rng = ChaChaRng::from_seed(&seed);
  (rng, seed)
}

pub fn spawn_rng(master_rng: &mut ChaChaRng) -> ChaChaRng {
  let mut seed = Vec::with_capacity(8);
  for _ in 0 .. 8 {
    seed.push(master_rng.next_u32());
  }
  assert_eq!(8, seed.len());
  ChaChaRng::from_seed(&seed)
}

pub fn init_master_rng() -> (Rc<RefCell<ChaChaRng>>, Vec<u32>) {
  let mut seed = Vec::with_capacity(8);
  for _ in 0 .. 8 {
    seed.push(thread_rng().next_u32());
  }
  assert_eq!(8, seed.len());
  let rng = Rc::new(RefCell::new(ChaChaRng::from_seed(&seed)));
  (rng, seed)
}

pub fn init_seed_rng(master_rng: &mut ChaChaRng) -> Rc<RefCell<ChaChaRng>> {
  let mut seed = Vec::with_capacity(8);
  for _ in 0 .. 8 {
    seed.push(master_rng.next_u32());
  }
  assert_eq!(8, seed.len());
  let rng = Rc::new(RefCell::new(ChaChaRng::from_seed(&seed)));
  rng
}

pub fn init_spawn_rng(master_rng: &mut ChaChaRng) -> ChaChaRng {
  let mut seed = Vec::with_capacity(8);
  for _ in 0 .. 8 {
    seed.push(master_rng.next_u32());
  }
  assert_eq!(8, seed.len());
  ChaChaRng::from_seed(&seed)
}
