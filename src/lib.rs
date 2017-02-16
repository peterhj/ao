#![feature(conservative_impl_trait)]
//#![feature(get_type_id)]
#![feature(specialization)]

extern crate densearray;
#[cfg(feature = "cuda")] extern crate devicemem_cuda;
extern crate rng;

extern crate libc;
extern crate rand;

pub use VarKind::*;

//use arithmetic::*;
//use densearray::prelude::*;

use rand::{Rng, SeedableRng, thread_rng};
use rand::chacha::{ChaChaRng};
use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc};

pub mod ffi;
pub mod ops;
pub mod prelude;

thread_local!(static NODE_ID_COUNTER: Cell<u64> = Cell::new(0));
thread_local!(static TXN_ID_COUNTER:  Cell<u64> = Cell::new(0));
thread_local!(static EPOCH_COUNTER:   Cell<u64> = Cell::new(0));

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
  Grad2,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Var {
  pub node_id:  NodeId,
  pub kind:     VarKind,
}

impl Var {
  /*pub fn new() -> Var {
    Var{node_id: NodeId::new()}
  }*/

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

#[derive(Clone)]
pub struct VarSet {
  inner:    HashSet<Var>,
  mask:     HashSet<Var>,
}

impl VarSet {
  pub fn empty() -> VarSet {
    VarSet{
      inner:    HashSet::new(),
      mask:     HashSet::new(),
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

  pub fn mask(&mut self, var: Var) {
    self.mask.insert(var);
  }

  pub fn is_masked(&mut self, var: &Var) -> bool {
    self.mask.contains(var)
  }
}

pub trait AutodiffOp {
  fn from(op: Rc<Self>) -> Rc<AutodiffOp> where Self: 'static + Sized { op.clone() }

  fn _id(&self) -> NodeId;
  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp));
  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOp));

  fn _data_size(&self, txn: TxnId, vars: &mut VarSet) -> usize { 0 }
  fn _load(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {}
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {}
  fn _load_r_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {}
  fn _store(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {}
  fn _store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {}
  fn _store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {}
  fn _store_r_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {}
  fn _store_grad2(&self, txn: TxnId, vars: &mut VarSet, writer: &mut Any) {}
  fn _rollover(&self, txn: TxnId, vars: &mut VarSet);

  fn _init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {}
  fn _forward(&self, txn: TxnId);
  fn _backward(&self, txn: TxnId, gauss_newton: bool) { unimplemented!(); }
  fn _r_forward(&self, txn: TxnId, gauss_newton: bool) { unimplemented!(); }
  fn _r_backward(&self, txn: TxnId) { unimplemented!(); }
  fn _backward2(&self, txn: TxnId) { unimplemented!(); }

  fn _reset_clock(&self) {}
  fn _set_clock(&self, clk: usize) { unimplemented!(); }

  fn data_size(&self, txn: TxnId, vars: &mut VarSet) -> usize {
    let epoch = Epoch::new(self._id());
    let mut size = 0;
    vars.unmask_all();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { size += op._data_size(txn, vars); });
    vars.unmask_all();
    size
  }

  fn load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut IoBuf) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    reader.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._load_val(txn, vars, reader.as_any()); });
    vars.unmask_all();
  }

  fn load_r_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut IoBuf) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    reader.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._load_r_val(txn, vars, reader.as_any()); });
    vars.unmask_all();
  }

  fn store_val(&self, txn: TxnId, vars: &mut VarSet, writer: &mut IoBuf) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._store_val(txn, vars, writer.as_any()); });
    vars.unmask_all();
  }

  fn store_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut IoBuf) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._store_grad(txn, vars, writer.as_any()); });
    vars.unmask_all();
  }

  fn store_r_grad(&self, txn: TxnId, vars: &mut VarSet, writer: &mut IoBuf) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._store_r_grad(txn, vars, writer.as_any()); });
    vars.unmask_all();
  }

  fn store_grad2(&self, txn: TxnId, vars: &mut VarSet, writer: &mut IoBuf) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    writer.reset();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._store_grad2(txn, vars, writer.as_any()); });
    vars.unmask_all();
  }

  fn init(&self, txn: TxnId, seed_rng: Rc<RefCell<ChaChaRng>>) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, {
      let seed_rng = seed_rng.clone();
      &mut move |op| { op._init(txn, seed_rng.clone()); }
    });
  }

  /*fn clear(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._clear(txn); });
    self._pop(epoch, &mut |_op| {});
  }*/

  fn rollover(&self, txn: TxnId, vars: &mut VarSet) {
    let epoch = Epoch::new(self._id());
    vars.unmask_all();
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._rollover(txn, vars); });
    vars.unmask_all();
  }

  fn eval(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
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
}

pub trait AutodiffObjective: AutodiffOp {
  fn _set_source(&self, txn: TxnId);

  fn gradient(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._set_source(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, /*vars,*/ false); });
  }

  fn gauss_newton_vector_product(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._set_source(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |op| { op._r_forward(txn, /*vars,*/ true); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, /*vars,*/ true); });
  }

  fn hessian_vector_product(&self, txn: TxnId) {
    let epoch = Epoch::new(self._id());
    self._set_source(txn);
    self._push(epoch, &mut |op| { op._forward(txn); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |op| { op._r_forward(txn, /*vars,*/ false); });
    self._pop(epoch, &mut |_op| {});
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._backward(txn, /*vars,*/ false); });
    self._push(epoch, &mut |_op| {});
    self._pop(epoch, &mut |op| { op._r_backward(txn, /*vars*/); });
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
  }
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

pub trait IoBuf: Any {
  fn reset(&mut self);
  fn as_any(&mut self) -> &mut Any;
}

pub struct CursorBuf<A> {
  buffer:   A,
  offset:   usize,
}

pub trait CursorBufExt<'a> {
  type Ref: ?Sized;
  type Mut: ?Sized;

  fn read_buf(&'a mut self, length: usize) -> Self::Ref;
  fn write_buf(&'a mut self, length: usize) -> Self::Mut;
}

impl<A> CursorBuf<A> {
  pub fn new(buffer: A) -> Self {
    CursorBuf{
      buffer:   buffer,
      offset:   0,
    }
  }
}

impl<A> Deref for CursorBuf<A> {
  type Target = A;

  fn deref(&self) -> &A {
    &self.buffer
  }
}

impl<A> DerefMut for CursorBuf<A> {
  fn deref_mut(&mut self) -> &mut A {
    &mut self.buffer
  }
}

impl<A> IoBuf for CursorBuf<A> where A: 'static {
  fn reset(&mut self) {
    self.offset = 0;
  }

  fn as_any(&mut self) -> &mut Any {
    self
  }
}

impl<'a, T> CursorBufExt<'a> for CursorBuf<Vec<T>> where T: 'a {
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
}

pub trait ArrayOp<A>: AutodiffOp {
  fn from(op: Rc<Self>) -> Rc<ArrayOp<A>> where Self: 'static + Sized { op.clone() }
  //fn data(&self) -> Rc<ArrayData<A>>;
  fn data(&self) -> ArrayDataNew<A>;
}

pub trait ArrayObjective<A>: ArrayOp<A> + AutodiffObjective {
}

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

pub struct ArrayVarBuf<A> {
  clk:      usize,
  curr_txn: Cell<Option<TxnId>>,
  readers:  RefCell<HashSet<NodeId>>,
  writers:  RefCell<HashSet<NodeId>>,
  writes:       RefCell<HashMap<NodeId, Symbol>>,
  read_writes:  RefCell<HashSet<(NodeId, Symbol)>>,
  coarse_rws:   RefCell<HashSet<NodeId>>,
  buffer:   RefCell<Option<A>>,
}

impl<A> ArrayVarBuf<A> {
  pub fn new(clk: usize) -> Self {
    ArrayVarBuf{
      clk:      clk,
      curr_txn: Cell::new(None),
      readers:  RefCell::new(HashSet::new()),
      writers:  RefCell::new(HashSet::new()),
      writes:       RefCell::new(HashMap::new()),
      read_writes:  RefCell::new(HashSet::new()),
      coarse_rws:   RefCell::new(HashSet::new()),
      buffer:   RefCell::new(None),
    }
  }
}

pub struct ArrayVarNew<A> {
  symbol:   Symbol,
  var:      Var,
  kind:     VarKind,
  alloc:    Rc<Fn(TxnId, NodeId) -> A>,
  curr_clk: Rc<Cell<usize>>,
  clk_bufs: Vec<Rc<ArrayVarBuf<A>>>,
}

impl<A> ArrayVarNew<A> {
  pub fn new(symbol: Symbol, kind: VarKind, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Self {
    let mut clk_bufs = Vec::with_capacity(clk_horizon);
    for clk in 0 .. clk_horizon {
      clk_bufs.push(Rc::new(ArrayVarBuf::new(clk)));
    }
    ArrayVarNew{
      symbol:   symbol,
      var:      Var::new(kind),
      kind:     kind,
      alloc:    alloc,
      curr_clk: Rc::new(Cell::new(0)),
      clk_bufs: clk_bufs,
    }
  }

  pub fn dup(&self, new_symbol: Symbol) -> Self {
    ArrayVarNew{
      symbol:   new_symbol,
      var:      self.var.clone(),
      kind:     self.kind,
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
    buf.readers.borrow_mut().clear();
    buf.writes.borrow_mut().clear();
    buf.read_writes.borrow_mut().clear();
    buf.coarse_rws.borrow_mut().clear();
  }

  /// Rollover this variable to a new transaction if this variable is
  /// a member of the provided variable set. Otherwise, invalidate it.
  pub fn rollover(&self, txn: TxnId, vars: &mut VarSet) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    if vars.contains(&self.var) {
      buf.curr_txn.set(Some(txn));
    } else {
      buf.curr_txn.set(None);
    }
    buf.readers.borrow_mut().clear();
    buf.writes.borrow_mut().clear();
    buf.read_writes.borrow_mut().clear();
    buf.coarse_rws.borrow_mut().clear();
  }

  /// Query this variable's availability to be overwritten,
  /// i.e. exclusive write.
  ///
  /// Exclusive writes satisfy the following rules:
  /// - An exclusive write on a variable is mutually exclusive with
  ///   reads and read-writes.
  /// - An exclusive write can be performed on a variable by only one
  ///   operand symbol; attempting to exclusively write to the same
  ///   variable using two different symbols is illegal.
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
        assert!(!buf.readers.borrow().contains(&node));
        assert!(!buf.coarse_rws.borrow().contains(&node));
        let written = buf.writes.borrow().contains_key(&node);
        if written {
          assert_eq!(self.symbol, *buf.writes.borrow().get(&node).unwrap());
        }
        incomplete_write = !written;
      }
    }
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
      let mut buffer = buf.buffer.borrow_mut();
      if buffer.is_none() {
        *buffer = Some((self.alloc)(txn, node));
      }
      //init(&mut *buffer.as_mut().unwrap());
    }
    incomplete_write
  }

  /// Query this variable's availability for accumulation,
  /// i.e. read-write.
  ///
  /// Read-writes satisfy the following rules:
  /// - A read-write on a variable is mutually exclusive with
  ///   reads and exclusive writes.
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
        assert!(!buf.readers.borrow().contains(&node));
        assert!(!buf.writes.borrow().contains_key(&node));
        let rw = buf.read_writes.borrow().contains(&(node, self.symbol));
        if rw {
          assert!(buf.coarse_rws.borrow().contains(&node));
        }
        incomplete_write = !rw;
      }
    }
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
      let mut buffer = buf.buffer.borrow_mut();
      if buffer.is_none() {
        *buffer = Some((self.alloc)(txn, node));
      }
      init(&mut *buffer.as_mut().unwrap());
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

  pub fn get_clk(&self, clk: usize, txn: TxnId, node: NodeId) -> Ref<A> {
    let buf = &self.clk_bufs[clk];
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      new_txn = curr_txn != txn;
    }
    assert!(!new_txn);
    assert!(!buf.writes.borrow().contains_key(&node));
    assert!(!buf.coarse_rws.borrow().contains(&node));
    buf.readers.borrow_mut().insert(node);
    Ref::map(buf.buffer.borrow(), |buffer| {
      if let Some(buffer) = buffer.as_ref() {
        buffer
      } else {
        panic!("trying to read from unallocated buffer");
      }
    })
  }

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
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
    }
    assert!(!buf.readers.borrow().contains(&node));
    assert!(!buf.coarse_rws.borrow().contains(&node));
    if buf.writes.borrow().contains_key(&node) {
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
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writes.borrow_mut().clear();
      buf.read_writes.borrow_mut().clear();
      buf.coarse_rws.borrow_mut().clear();
    }
    assert!(!buf.readers.borrow().contains(&node));
    assert!(!buf.writes.borrow().contains_key(&node));
    if buf.read_writes.borrow().contains(&(node, self.symbol)) {
      assert!(buf.coarse_rws.borrow().contains(&node));
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

pub struct ArrayVar<A> {
  var:      Var,
  kind:     VarKind,
  alloc:    Rc<Fn(TxnId, NodeId) -> A>,
  curr_clk: Cell<usize>,
  clk_bufs: Vec<ArrayVarBuf<A>>,
}

impl<A> ArrayVar<A> {
  pub fn new(kind: VarKind, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Self {
    let mut clk_bufs = Vec::with_capacity(clk_horizon);
    for clk in 0 .. clk_horizon {
      clk_bufs.push(ArrayVarBuf::new(clk));
    }
    ArrayVar{
      var:      Var::new(kind),
      kind:     kind,
      alloc:    alloc,
      curr_clk: Cell::new(0),
      clk_bufs: clk_bufs,
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

  pub fn invalidate(&self) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    buf.curr_txn.set(None);
    buf.readers.borrow_mut().clear();
    buf.writers.borrow_mut().clear();
    //buf.writers_.borrow_mut().clear();
  }

  pub fn rollover(&self, txn: TxnId, vars: &mut VarSet) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    if vars.contains(&self.var) {
      buf.curr_txn.set(Some(txn));
    } else {
      buf.curr_txn.set(None);
    }
    buf.readers.borrow_mut().clear();
    buf.writers.borrow_mut().clear();
    //buf.writers_.borrow_mut().clear();
  }

  pub fn write(&self, txn: TxnId, node: NodeId) -> bool {
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
        incomplete_write = !buf.writers.borrow().contains(&node);
      }
    }
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writers.borrow_mut().clear();
      //buf.writers_.borrow_mut().clear();
      let mut buffer = buf.buffer.borrow_mut();
      if buffer.is_none() {
        *buffer = Some((self.alloc)(txn, node));
      }
      //init(&mut *buffer.as_mut().unwrap());
    }
    incomplete_write
  }

  pub fn accumulate<F>(&self, txn: TxnId, node: NodeId, init: F) -> bool where F: Fn(&mut A) {
    let clk = self.curr_clk.get();
    let buf = &self.clk_bufs[clk];
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      if curr_txn != txn {
        new_txn = true;
      }
    }
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writers.borrow_mut().clear();
      //buf.writers_.borrow_mut().clear();
      let mut buffer = buf.buffer.borrow_mut();
      if buffer.is_none() {
        *buffer = Some((self.alloc)(txn, node));
      }
      init(&mut *buffer.as_mut().unwrap());
    }
    // FIXME(20170215): incorrect, need to account for symbols.
    true
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

  pub fn get_clk(&self, clk: usize, txn: TxnId, node: NodeId) -> Ref<A> {
    let buf = &self.clk_bufs[clk];
    let mut new_txn = false;
    if buf.curr_txn.get().is_none() {
      new_txn = true;
    } else {
      let curr_txn = buf.curr_txn.get().unwrap();
      new_txn = curr_txn != txn;
    }
    assert!(!new_txn);
    assert!(!buf.writers.borrow().contains(&node));
    buf.readers.borrow_mut().insert(node);
    Ref::map(buf.buffer.borrow(), |buffer| {
      if let Some(buffer) = buffer.as_ref() {
        buffer
      } else {
        panic!("trying to read from unallocated buffer");
      }
    })
  }

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
    if new_txn {
      buf.curr_txn.set(Some(txn));
      buf.readers.borrow_mut().clear();
      buf.writers.borrow_mut().clear();
    }
    assert!(!buf.readers.borrow().contains(&node));
    buf.writers.borrow_mut().insert(node);
    RefMut::map(buf.buffer.borrow_mut(), |buf| {
      if let Some(buf) = buf.as_mut() {
        buf
      } else {
        panic!("trying to write to unallocated buffer");
      }
    })
  }
}

pub struct ArrayDataNew<A> {
  symbol:       Symbol,
  clk_horizon:  usize,
  alloc:        Rc<Fn(TxnId, NodeId) -> A>,
  pub val:      ArrayVarNew<A>,
  pub grad:     ArrayVarNew<A>,
  pub r_val:    ArrayVarNew<A>,
  pub r_grad:   ArrayVarNew<A>,
  pub grad2:    ArrayVarNew<A>,
}

impl<A> Clone for ArrayDataNew<A> {
  fn clone(&self) -> Self {
    let new_symbol = Symbol::new();
    ArrayDataNew{
      symbol:       new_symbol,
      clk_horizon:  self.clk_horizon,
      alloc:        self.alloc.clone(),
      val:      self.val.dup(new_symbol),
      grad:     self.grad.dup(new_symbol),
      r_val:    self.r_val.dup(new_symbol),
      r_grad:   self.r_grad.dup(new_symbol),
      grad2:    self.grad2.dup(new_symbol),
    }
  }
}

impl<A> ArrayDataNew<A> {
  pub fn new(clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Self {
    let symbol = Symbol::new();
    ArrayDataNew{
      symbol:       symbol,
      clk_horizon:  clk_horizon,
      alloc:        alloc.clone(),
      val:      ArrayVarNew::new(symbol, Val, clk_horizon, alloc.clone()),
      grad:     ArrayVarNew::new(symbol, Grad, clk_horizon, alloc.clone()),
      r_val:    ArrayVarNew::new(symbol, RVal, clk_horizon, alloc.clone()),
      r_grad:   ArrayVarNew::new(symbol, RGrad, clk_horizon, alloc.clone()),
      grad2:    ArrayVarNew::new(symbol, Grad2, clk_horizon, alloc.clone()),
    }
  }

  pub fn horizon(&self) -> usize {
    self.clk_horizon
  }

  pub fn vars(&self) -> VarSet {
    VarSet::empty()
      .add(self.val.var())
      .add(self.grad.var())
      .add(self.r_val.var())
      .add(self.r_grad.var())
      .add(self.grad2.var())
  }

  pub fn reset_clock_all(&self) {
    self.val.reset_clock();
    self.grad.reset_clock();
    self.r_val.reset_clock();
    self.r_grad.reset_clock();
    self.grad2.reset_clock();
  }

  pub fn set_clock_all(&self, clk: usize) {
    self.val.set_clock(clk);
    self.grad.set_clock(clk);
    self.r_val.set_clock(clk);
    self.r_grad.set_clock(clk);
    self.grad2.set_clock(clk);
  }

  pub fn rollover_all(&self, txn: TxnId, vars: &mut VarSet) {
    self.val.rollover(txn, vars);
    self.grad.rollover(txn, vars);
    self.r_val.rollover(txn, vars);
    self.r_grad.rollover(txn, vars);
    self.grad2.rollover(txn, vars);
  }
}

pub struct ArrayData<A> {
  clk_horizon:  usize,
  alloc:        Rc<Fn(TxnId, NodeId) -> A>,
  pub val:      ArrayVar<A>,
  pub grad:     ArrayVar<A>,
  pub r_val:    ArrayVar<A>,
  pub r_grad:   ArrayVar<A>,
  pub grad2:    ArrayVar<A>,
}

impl<A> ArrayData<A> {
  pub fn new(clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<Self> {
    Rc::new(ArrayData{
      clk_horizon:  clk_horizon,
      alloc:        alloc.clone(),
      val:      ArrayVar::new(Val, clk_horizon, alloc.clone()),
      grad:     ArrayVar::new(Grad, clk_horizon, alloc.clone()),
      r_val:    ArrayVar::new(RVal, clk_horizon, alloc.clone()),
      r_grad:   ArrayVar::new(RGrad, clk_horizon, alloc.clone()),
      grad2:    ArrayVar::new(Grad2, clk_horizon, alloc.clone()),
    })
  }

  /*pub fn refs(&self) -> (Var, Var, Var, Var) {
    unimplemented!();
  }*/

  pub fn horizon(&self) -> usize {
    self.clk_horizon
  }

  pub fn vars(&self) -> VarSet {
    VarSet::empty()
      .add(self.val.var())
      .add(self.grad.var())
      .add(self.r_val.var())
      .add(self.r_grad.var())
      .add(self.grad2.var())
  }

  pub fn reset_clock_all(&self) {
    self.val.reset_clock();
    self.grad.reset_clock();
    self.r_val.reset_clock();
    self.r_grad.reset_clock();
    self.grad2.reset_clock();
  }

  pub fn set_clock_all(&self, clk: usize) {
    self.val.set_clock(clk);
    self.grad.set_clock(clk);
    self.r_val.set_clock(clk);
    self.r_grad.set_clock(clk);
    self.grad2.set_clock(clk);
  }

  pub fn rollover_all(&self, txn: TxnId, vars: &mut VarSet) {
    self.val.rollover(txn, vars);
    self.grad.rollover(txn, vars);
    self.r_val.rollover(txn, vars);
    self.r_grad.rollover(txn, vars);
    self.grad2.rollover(txn, vars);
  }
}
