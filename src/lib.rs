#![feature(specialization)]

extern crate densearray;
#[cfg(cuda)] extern crate devicemem_cuda;

//use arithmetic::*;
//use densearray::prelude::*;

use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashSet};
use std::rc::{Rc};

pub mod ops;
#[cfg(cuda)] pub mod ops_cuda;
pub mod prelude;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(u64);

impl NodeId {
  pub fn new() -> NodeId {
    unimplemented!();
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TxnId(u64);

impl TxnId {
  pub fn new() -> TxnId {
    unimplemented!();
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EpochNr(u64);

impl EpochNr {
  pub fn new() -> EpochNr {
    unimplemented!();
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
  curr_epoch:   Cell<Epoch>,
  entries:      RefCell<Vec<OperatorStackEntry>>,
}

impl OperatorStack {
  pub fn new(node_id: NodeId, in_degree: usize) -> OperatorStack {
    unimplemented!();
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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct DataRef {
  pub node_id:  NodeId,
}

impl DataRef {
  pub fn new() -> DataRef {
    DataRef{node_id: NodeId::new()}
  }
}

#[derive(Clone)]
pub struct DataRefSet {
  inner:    HashSet<DataRef>,
  mask:     HashSet<DataRef>,
}

impl DataRefSet {
  pub fn new() -> DataRefSet {
    unimplemented!();
  }

  pub fn clear(&mut self) {
    self.inner.clear();
    self.mask.clear();
  }

  pub fn unmask_all(&mut self) {
    self.mask.clear();
  }

  pub fn mask(&mut self, data_ref: DataRef) {
    self.mask.insert(data_ref);
  }
}

pub trait DiffOperator {
  fn _id(&self) -> NodeId;
  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator));
  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator));

  fn _data_sz(&self, ref_set: &mut DataRefSet) -> usize { 0 }
  fn _load(&self, txn: TxnId, ref_set: &mut DataRefSet, reader: &mut Any) {}
  fn _store(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {}

  fn _clear(&self, txn: TxnId) {}
  fn _forward(&self, txn: TxnId);
  fn _backward(&self, txn: TxnId, /*ref_set: &DataRefSet,*/ gauss_newton: bool) { unimplemented!(); }
  fn _r_forward(&self, txn: TxnId, /*ref_set: &DataRefSet,*/ gauss_newton: bool) { unimplemented!(); }
  fn _r_backward(&self, txn: TxnId, /*ref_set: &DataRefSet*/) { unimplemented!(); }
  fn _clock(&self, txn: TxnId) { unimplemented!(); }

  fn data_sz(&self, ref_set: &mut DataRefSet) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._data_sz(ref_set); });
  }

  fn load(&self, txn: TxnId, ref_set: &mut DataRefSet, reader: &mut Any) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._load(txn, ref_set, reader); });
  }

  fn store(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._store(txn, ref_set, writer); });
  }

  fn clear(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._clear(txn); });
    self._pop(epoch, &mut |_op| {});
  }

  fn eval(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
  }

  fn clock(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._clock(txn); });
    self._pop(epoch, &mut |_op| {});
  }
}

pub trait DiffLoss: DiffOperator {
  fn gradient(&self, txn: TxnId, /*ref_set: &DataRefSet*/) {
    let epoch = Epoch::new(self._id());
    self.clear(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, /*ref_set,*/ false); });
  }

  fn gauss_newton_vector_product(&self, txn: TxnId, /*ref_set: &DataRefSet*/) {
    let epoch = Epoch::new(self._id());
    self.clear(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |op| { op._r_forward(txn, /*ref_set,*/ true); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, /*ref_set,*/ true); });
  }

  fn hessian_vector_product(&self, txn: TxnId, /*ref_set: &DataRefSet*/) {
    let epoch = Epoch::new(self._id());
    self.clear(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |op| { op._r_forward(txn, /*ref_set,*/ false); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, /*ref_set,*/ false); });
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._r_backward(txn, /*ref_set*/); });
  }
}

pub trait ArrayOp<A>: DiffOperator {
  fn data(&self) -> Rc<ArrayData<A>>;
}

pub trait ArrayAllocator<A> {
  fn alloc(&self) -> A;
}

pub fn alloc_fn<A, F>(f: F) -> FnArrayAllocator<A, F> where F: Fn() -> A {
  unimplemented!();
}

pub struct FnArrayAllocator<A, F> where F: Fn() -> A {
  cons:     F,
}

pub struct TxnBuf<A> {
  curr_txn: Cell<Option<TxnId>>,
  writers:  RefCell<HashSet<NodeId>>,
  buffer:   RefCell<Option<A>>,
}

impl<A> TxnBuf<A> {
  pub fn invalidate(&self, txn: TxnId, node: NodeId) -> bool {
    let mut invalid = false;
    let mut new_txn = false;
    if self.curr_txn.get().is_none() {
      invalid = true;
      new_txn = true;
    } else {
      let curr_txn = self.curr_txn.get().unwrap();
      if curr_txn != txn {
        invalid = true;
        new_txn = true;
      } else {
        invalid = !self.writers.borrow().contains(&node);
      }
    }
    if new_txn {
      self.curr_txn.set(None);
      self.writers.borrow_mut().clear();
    }
    invalid
  }

  pub fn maybe_alloc<F>(&self, init: F) where F: Fn(&mut A) {
    unimplemented!();
  }

  pub fn get(&self, txn: TxnId) -> Ref<A> {
    let mut invalid = false;
    if self.curr_txn.get().is_none() {
      invalid = true;
    } else {
      let curr_txn = self.curr_txn.get().unwrap();
      invalid = curr_txn != txn;
    }
    assert!(!invalid);
    Ref::map(self.buffer.borrow(), |buf| {
      if let Some(buf) = buf.as_ref() {
        buf
      } else {
        panic!("trying to read from unallocated buffer");
      }
    })
  }

  pub fn get_mut(&self, txn: TxnId, node: NodeId) -> RefMut<A> {
    let mut new_txn = false;
    if self.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = self.curr_txn.get().unwrap();
      new_txn = curr_txn != txn;
    }
    if new_txn {
      self.curr_txn.set(Some(txn));
      self.writers.borrow_mut().clear();
    }
    self.writers.borrow_mut().insert(node);
    RefMut::map(self.buffer.borrow_mut(), |buf| {
      if let Some(buf) = buf.as_mut() {
        buf
      } else {
        panic!("trying to write to unallocated buffer");
      }
    })
  }

  pub fn maybe_get_mut(&self, txn: TxnId, node: NodeId) -> Option<RefMut<A>> {
    let buf = self.buffer.borrow_mut();
    if buf.is_none() {
      None
    } else {
      Some(self.get_mut(txn, node))
    }
  }
}

pub struct ArrayData<A> {
  allocator:    Rc<ArrayAllocator<A>>,
  pub val:      TxnBuf<A>,
  pub grad:     TxnBuf<A>,
  pub r_val:    TxnBuf<A>,
  pub r_grad:   TxnBuf<A>,
}
