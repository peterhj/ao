use arithmetic::*;
use densearray::prelude::*;
//use operator::prelude::*;

use std::cell::{Cell, RefCell};
use std::collections::{HashSet};
use std::marker::{PhantomData};
use std::rc::{Rc};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

#[derive(Clone, Copy)]
pub struct EpochNr(u64);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchNr(u64);

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Epoch {
  pub epoch_nr: EpochNr,
  pub root:     NodeId,
}

pub struct OperatorStackEntry/*<Data>*/ {
  epoch:        Epoch,
  push_count:   usize,
  pop_count:    usize,
  //data:         Data,
}

pub struct OperatorStack/*<Data=()>*/ {
  node_id:      NodeId,
  in_degree:    usize,
  curr_epoch:   Cell<Epoch>,
  entries:      RefCell<Vec<OperatorStackEntry/*<Data>*/>>,
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
      entries.push(NodeStackEntry{
        epoch:      epoch,
        push_count: 1,
        pop_count:  0,
      });
    }
    entries.last().unwrap().push_count
  }

  pub fn degree(&self, epoch: Epoch) -> usize {
    assert!(!entries.is_empty());
    let level = entries.len() - 1;
    assert_eq!(epoch, entries[level].epoch);
    entries[level].push_count
  }

  pub fn pop(&self, epoch: Epoch) {
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
pub struct VarRef {
  pub node_id:  NodeId,
}

impl VarRef {
  pub fn new() -> VarRef {
    VarRef{node_id: NodeId::new()}
  }
}

#[derive(Clone)]
pub struct VarSet {
  inner:    HashSet<VarRef>,
  mask:     HashSet<VarRef>,
}

pub trait DiffOperatorIo<IoBuf: ?Sized> {
  fn _load(&self, vars: &mut VarSet, reader: &mut IoBuf, offset: usize) -> usize { 0 }
  fn _store(&self, vars: &mut VarSet, writer: &mut IoBuf, offset: usize) -> usize { 0 }
}

pub trait DiffOperatorCore {
  fn _reset_batch(&self) { unimplemented!(); }
  fn _unmask_batch_weight(&self) { unimplemented!(); }
  fn _forward(&self) { unimplemented!(); }
  fn _backward(&self, vars: &VarSet) { unimplemented!(); }
  fn _backward2(&self, vars: &VarSet) { unimplemented!(); }
  fn _r_unmask_batch_weight(&self) { unimplemented!(); }
  fn _r_forward(&self, vars: &VarSet) { unimplemented!(); }
  fn _r_backward(&self, vars: &VarSet) { unimplemented!(); }
  fn _clock(&self) { unimplemented!(); }
}

pub trait DiffOperator<IoBuf: ?Sized>: DiffOperatorCore + DiffOperatorIo<IoBuf> {
  fn _push_fwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<IoBuf>));
  fn _pop_fwd(&self, epoch: Epoch);
  fn _push_bwd(&self, epoch: Epoch);
  fn _pop_bwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<IoBuf>));
}

pub trait DiffLossIo {
  fn load(&self, vars: &mut VarSet, src: &mut Any);
  fn store(&self, vars: &mut VarSet, dst: &mut Any);

  fn load_batch_weight(&self, vars: &mut VarSet, src: &mut Any);
  fn load_r_direction(&self, vars: &mut VarSet, src: &mut Any);

  fn store_scalar_loss(&self, dst: &mut Any);
  fn store_grad(&self, vars: &mut VarSet, dst: &mut Any);
  fn store_grad2(&self, vars: &mut VarSet, dst: &mut Any);
  fn store_r_grad(&self, vars: &mut VarSet, dst: &mut Any);
}

pub trait DiffLoss: DiffLossIo {
  fn reset_batch(&self);
  fn apply_batch_weight(&self);
  fn forward(&self);
  fn backward(&self, vars: &VarSet);
  fn backward2(&self, vars: &VarSet);
  fn r_apply_batch_weight(&self);
  fn r_forward(&self);
  fn r_backward(&self, vars: &VarSet);
  fn clock(&self);

  fn eval(&self);
  fn grad(&self, vars: &VarSet);
  fn grad2(&self, vars: &VarSet);
  fn jacobian_vector_product(&self, vars: &VarSet);
  fn gauss_newton_vector_product(&self, vars: &VarSet);
  fn hessian_vector_product(&self, vars: &VarSet);
}

pub struct MimoOp {
  stack:    OperatorStack,
  in_ops:   Vec<Rc<DiffOperator<()>>>,
}

impl DiffOperator<()> for MimoOp {
  fn _push_fwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if self.stack.push(epoch) == 1 {
      for op in self.in_ops.iter() {
        let mut op = op;
        op._push_fwd(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop_fwd(&self, epoch: Epoch) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      for op in self.in_ops.iter() {
        op._pop_fwd(epoch);
      }
    }
  }

  fn _push_bwd(&self, epoch: Epoch) {
    if self.stack.push(epoch) == 1 {
      for op in self.in_ops.iter() {
        op._push_bwd(epoch);
      }
    }
  }

  fn _pop_bwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      for op in self.in_ops.iter() {
        let mut op = op;
        op._pop_bwd(epoch, apply);
      }
    }
  }
}

pub struct BatchMuxOp {
  stack:    OperatorStack,
  in_op:    Rc<DiffOperator<()>>,
  num_batches:  usize,
  batch_offset: Cell<Option<usize>>,
}

impl DiffOperator<()> for BatchMuxOp {
  fn _push_fwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if self.stack.push(epoch) == 1 {
      if let None = self.batch_offset {
        self.in_op._push_fwd(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop_fwd(&self, epoch: Epoch) {
    // FIXME
  }

  fn _push_bwd(&self, epoch: Epoch) {
    // FIXME
  }

  fn _pop_bwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    // FIXME
  }
}

pub struct BatchDemuxOp {
  node_id:  NodeId,
  stack:    OperatorStack,
  mux_op:   Rc<BatchMuxOp>,
  in_op:    Rc<DiffOperator<()>>,
  num_batches:  usize,
  batch_offset: usize,
}

impl DiffOperator<()> for BatchDemuxOp {
  fn _push_fwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if self.stack.push(epoch) == 1 {
      self.mux_op._push_fwd(epoch, apply);
      for batch_idx in 0 .. self.num_batches {
        let batch_epoch = Epoch::new(self.node_id);
        self.batch_offset = batch_idx;
        self.mux_op.batch_offset = Some(batch_idx);
        self.in_op._push_fwd(batch_epoch, apply);
        self.in_op._pop_fwd(batch_epoch);
      }
      self.batch_offset = 0;
      self.mux_op.batch_offset = None;
      apply(self);
    }
  }

  fn _pop_fwd(&self, epoch: Epoch) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      /*for batch_idx in 0 .. self.num_batches {
        let batch_epoch = Epoch::new(self.node_id);
        self.mux_op.batch_offset = Some(batch_idx);
        self.in_op._pop_fwd(batch_epoch);
      }
      self.mux_op.batch_offset = None;*/
      self.mux_op._pop_fwd(epoch);
    }
  }

  fn _push_bwd(&self, epoch: Epoch) {
    if self.stack.push(epoch) == 1 {
      self.mux_op._push_bwd(epoch);
      /*for batch_idx in 0 .. self.num_batches {
        let batch_epoch = Epoch::new(self.node_id);
        self.mux_op.batch_offset = Some(batch_idx);
        self.in_op._push_bwd(batch_epoch);
      }
      self.mux_op.batch_offset = None;*/
    }
  }

  fn _pop_bwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      for batch_idx in 0 .. self.num_batches {
        let batch_epoch = Epoch::new(self.node_id);
        self.batch_offset = batch_idx;
        self.mux_op.batch_offset = Some(batch_idx);
        self.in_op._push_bwd(batch_epoch);
        self.in_op._pop_bwd(batch_epoch, apply);
      }
      self.batch_offset = 0;
      self.mux_op.batch_offset = None;
      self.mux_op._pop_bwd(epoch, apply);
    }
  }
}

pub struct ScalarLoss {
  node_id:  NodeId,
  stack:    OperatorStack,
  in_op:    Rc<DiffOperator<()>>,
}

impl ScalarLoss {
  pub fn forward(&self) {
    let epoch = Epoch::new(self.node_id);
    self._push_fwd(epoch, |op| op._forward());
    self._pop_fwd(epoch);
  }

  pub fn backward(&self) {
    let epoch = Epoch::new(self.node_id);
    self._push_bwd(epoch);
    self._pop_bwd(epoch, |op| op._backward());
  }
}

impl DiffOperator<()> for ScalarLoss {
  fn _push_fwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if 1 == self.stack.push(epoch) {
      self.in_op._push_fwd(epoch, apply);
      apply(self);
    }
  }

  fn _pop_fwd(&self, epoch: Epoch) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      self.in_op._pop_fwd(epoch);
    }
  }

  fn _push_bwd(&self, epoch: Epoch) {
    if 1 == self.stack.push(epoch) {
      self.in_op._push_bwd(epoch);
    }
  }

  fn _pop_bwd(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator<()>)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.in_op._pop_bwd(epoch, apply);
    }
  }
}

pub trait ArrayOp {
  type Target;
}

pub trait VectorOps<Rhs, Target> where Rhs: ArrayOp<Target=Target> {
  fn linear(self, x: Rhs) -> LinearOp<Self, Rhs, Target> where Self: Sized {
    LinearOp::new(self, x)
  }
}

pub trait TensorOps {
  fn conv2d<Rhs, Out>(&self, x: Rhs) -> Conv2dOp<Self, Rhs, Out> where Self: Sized;
}

pub fn var<A>(a: A) -> ArrayVar<A> {
  unimplemented!();
}

pub struct ArrayVar<A> {
  var:  A,
}

pub struct LinearOp<A, X, Target> {
  a:    A,
  b:    X,
  out:  Target,
}

impl LinearOp<A, X, Target> {
  pub fn new() -> Self {
    unimplemented!();
  }
}

impl LinearOp<A, X, Array1d<f32>>
where A:    ArrayOp<Target=Array2d<f32>>,
      X:    ArrayOp<Target=Array1d<f32>>,
{
}

impl LinearOp<A, X, XatchArray1d<f32>>
where A:    ArrayOp<Target=Array2d<f32>>,
      X:    ArrayOp<Target=XatchArray1d<f32>>,
{
}

pub struct Conv2dOp<A, X, Out> {
  a:    A,
  b:    X,
  out:  Out,
}
