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
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashSet};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

#[cfg(feature = "cuda")] pub mod cuda;

//const VEC_F32_TYPEID: TypeId = TypeId::of::<Vec<f32>>();

pub fn var<A, F>(cons: F) -> Rc<ArraySrc<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  ArraySrc::new(1, false, Rc::new(cons))
}

pub fn sequential_var<A, F>(horizon: usize, cons: F) -> Rc<ArraySrc<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  ArraySrc::new(horizon, true, Rc::new(cons))
}

/*pub fn test_var() {
  let x: Rc<ArraySrc<Array1d<f32>>> = var(|_, _| Array1d::zeros(10));
}*/

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

impl ArrayOp<Array1d<f32>> for ArraySrc<Array1d<f32>> {
  fn data(&self) -> ArrayData<Array1d<f32>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<Array1d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
        /*match reader.get_type_id() {
          VEC_F32_TYPEID => {}
          _ => {}
        }*/
        if reader.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
          let mut val = self.data.val.get_excl(txn, node);
          let val_len = val.dim();
          let reader = reader.downcast_mut::<CursorBuf<Vec<f32>>>().unwrap();
          val.as_view_mut().copy(reader.read_buf(val_len).flatten());
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
        writer.write_buf(val_len).flatten_mut().copy(val.as_view());
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
        writer.write_buf(grad_len).flatten_mut().copy(grad.as_view());
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

impl ArrayOp<Array2d<f32>> for ArraySrc<Array2d<f32>> {
  fn data(&self) -> ArrayData<Array2d<f32>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<Array2d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
        if reader.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
          let mut val = self.data.val.get_excl(txn, node);
          let val_len = val.dim().flat_len();
          let reader = reader.downcast_mut::<CursorBuf<Vec<f32>>>().unwrap();
          val.as_view_mut().flatten_mut().copy(reader.read_buf(val_len).flatten());
        } else {
          unimplemented!();
        }
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

impl ArrayOp<Array4d<f32>> for ArraySrc<Array4d<f32>> {
  fn data(&self) -> ArrayData<Array4d<f32>> {
    self.data.clone()
  }
}

impl AutodiffOp for ArraySrc<Array4d<f32>> {
  fn _load_val(&self, txn: TxnId, vars: &mut VarSet, reader: &mut Any) {
    let node = self._id();
    if vars.contains(&self.data.val.var()) {
      if self.data.val.overwrite(txn, node) {
        if reader.downcast_mut::<CursorBuf<Vec<f32>>>().is_some() {
          let mut val = self.data.val.get_excl(txn, node);
          let val_len = val.dim().flat_len();
          let reader = reader.downcast_mut::<CursorBuf<Vec<f32>>>().unwrap();
          val.as_view_mut().flatten_mut().copy(reader.read_buf(val_len).flatten());
        } else {
          unimplemented!();
        }
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

pub struct PassOp<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:       Rc<AutodiffOp>,
  data:     ArrayData<A>,
}

impl<A> PassOp<A> {
  pub fn new(x_: Rc<AutodiffOp>, data: ArrayData<A>) -> Rc<Self> {
    let node = NodeId::new();
    Rc::new(PassOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:       x_,
      data:     data,
    })
  }
}

impl<A> ArrayOp<A> for PassOp<A> {
  fn data(&self) -> ArrayData<A> {
    self.data.clone()
  }
}

impl<A> AutodiffOp for PassOp<A> {
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

  fn _forward(&self, txn: TxnId) {
  }

  fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }
}

pub struct InitializeOp<A, InitF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<ArrayOp<A>>,
  data: ArrayData<A>,
  kernel:   InitF,
}

pub fn xavier_linear_init<R>() -> impl Fn(Rc<RefCell<R>>, &mut Array2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed([seed_rng.next_u64(), seed_rng.next_u64()]);
    let half_range = (6.0 / (a.dim().0 + a.dim().1) as f64).sqrt();
    let dist = Range::new(-half_range, half_range);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub fn xavier_conv2d_init<R>(/*axes: (usize, usize)*/) -> impl Fn(Rc<RefCell<R>>, &mut Array4d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array4d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed([seed_rng.next_u64(), seed_rng.next_u64()]);
    let half_range = (6.0 / (a.dim().0 * a.dim().1 * a.dim().2 + a.dim().3) as f64).sqrt();
    let dist = Range::new(-half_range, half_range);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub fn kaiming_linear_init<R>() -> impl Fn(Rc<RefCell<R>>, &mut Array2d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array2d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed([seed_rng.next_u64(), seed_rng.next_u64()]);
    let std = (2.0 / a.dim().0 as f64).sqrt();
    let dist = Normal::new(0.0, std);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub fn kaiming_conv2d_init<R>(/*axes: (usize, usize)*/) -> impl Fn(Rc<RefCell<R>>, &mut Array4d<f32>) where R: Rng {
  move |seed_rng: Rc<RefCell<R>>, a: &mut Array4d<f32>| {
    let mut seed_rng = seed_rng.borrow_mut();
    let mut rng = Xorshiftplus128Rng::from_seed([seed_rng.next_u64(), seed_rng.next_u64()]);
    let std = (2.0 / (a.dim().0 * a.dim().1 * a.dim().2) as f64).sqrt();
    let dist = Normal::new(0.0, std);
    for e in a.as_mut_slice().iter_mut() {
      *e = dist.ind_sample(&mut rng) as f32;
    }
  }
}

pub trait InitializeExt<A, F, InitF> {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, InitF>>;
}

impl<Op, A, F> InitializeExt<A, F, Rc<F>> for Rc<Op> where Op: 'static + ArrayOp<A>, F: Fn(Rc<RefCell<ChaChaRng>>, &mut A) {
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
}

impl<A, InitF> ArrayOp<A> for InitializeOp<A, InitF> where InitializeOp<A, InitF>: AutodiffOp {
  fn data(&self) -> ArrayData<A> {
    self.data.clone()
  }
}

impl<S, F> AutodiffOp for InitializeOp<Array1d<f32, S>, Rc<F>> where S: DerefMut<Target=[f32]>, F: Fn(Rc<RefCell<ChaChaRng>>, &mut Array1d<f32, S>) {
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
  fn data(&self) -> ArrayData<A> {
    self.y.clone()
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

  default fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

pub struct FlattenTransform;

pub trait FlattenExt<A, B> {
  fn flatten(&self) -> Rc<TransformOp<A, B, FlattenTransform>>;
}

impl<S> FlattenExt<Array3d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array3d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn flatten(&self) -> Rc<TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform>> {
    let clk_horizon = self.data().horizon();
    TransformOp::new(self.clone(), clk_horizon, {
      let x = self.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim.flat_len());
        Array1d::from_storage(dim.flat_len(), buf)
      })
    })
  }
}

impl<S> TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform> where S: DerefMut<Target=[f32]> {
  pub fn new<F>(x_: Rc<ArrayOp<Array3d<f32, S>>>, clk_horizon: usize, alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> Array1d<f32, S> {
    let node = NodeId::new();
    let x = x_.data();
    Rc::new(TransformOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x_:   x_,
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc),
      kernel:   FlattenTransform,
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

impl<A, JoinF> ArrayOp<A> for JoinOp<A, JoinF> where JoinOp<A, JoinF>: AutodiffOp {
  fn data(&self) -> ArrayData<A> {
    self.y.clone()
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

pub trait SumExt<Op, A> where Op: ArrayOp<A> {
  fn sum(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<A, SumJoinKernel>>;
  fn add(&self, x_: Rc<Op>) -> Rc<JoinOp<A, SumJoinKernel>>;
}

pub fn sum<Op, A>(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<A, SumJoinKernel>> where Rc<Op>: SumExt<Op, A>, Op: ArrayOp<A> {
  <Rc<Op> as SumExt<Op, A>>::sum(xs_)
}

impl<Op, S> AxisJoinExt<Op, Array1d<f32, S>> for Rc<Op> where Op: ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn axis_join(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<Array1d<f32, S>, AxisJoinKernel>> {
    unimplemented!();
  }
}

impl<Op, S> SumExt<Op, Array1d<f32, S>> for Rc<Op> where Op: ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn sum(xs_: Vec<Rc<Op>>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> where S: DerefMut<Target=[f32]> {
    unimplemented!();
  }

  fn add(&self, x_: Rc<Op>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> {
    Self::sum(vec![self.clone(), x_])
  }
}

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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

pub trait MultiplyExt<A, V, W, B> {
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

impl<Op, S> MultiplyExt<Array1d<f32, S>, Array1d<f32, S>, f32, f32> for Rc<Op> where Op: 'static + ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn mult(&self, x_: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, None, clk_horizon, Rc::new(|_, _| 0.0_f32))
  }

  fn mult_add(&self, x_: Rc<ArrayOp<Array1d<f32, S>>>, b_: Rc<ArrayOp<f32>>) -> Rc<LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32>> {
    let clk_horizon = x_.data().horizon();
    LinearOp::new(self.clone(), x_, Some(b_), clk_horizon, Rc::new(|_, _| 0.0_f32))
  }
}

impl<S> ArrayOp<f32> for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
  fn data(&self) -> ArrayData<f32> {
    self.y.clone()
  }
}

impl<S> AutodiffObjective for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
    } else {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}

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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

impl<S> MultiplyExt<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array2d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
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

impl<S> ArrayOp<Array1d<f32, S>> for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn data(&self) -> ArrayData<Array1d<f32, S>> {
    self.y.clone()
  }
}

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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

impl<S> MultiplyExt<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array2d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + BatchArrayStorage<usize> {
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

impl<S> ArrayOp<BatchArray1d<f32, S>> for LinearOp<Array2d<f32, S>, BatchArray1d<f32, S>, BatchArray1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn data(&self) -> ArrayData<BatchArray1d<f32, S>> {
    self.y.clone()
  }
}

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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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
  tmp:  ArrayData<V>,
  kernel:   K,
}

pub struct ScaleElemKernel;
pub struct NormalizeElemKernel;

impl<S> AutodiffOp for ElemLinearOp<Array1d<f32, S>, BatchArray3d<f32, S>, NormalizeElemKernel> where S: DerefMut<Target=[f32]> {
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

pub trait ConvExt<Idx, A, B, V, Kernel> where Idx: ArrayIndex {
  fn conv(&self, shape: ConvShape<Idx>, x_: Rc<ArrayOp<V>>) -> Rc<ConvOp<Idx, A, B, V, Kernel>>;
  fn conv_add(&self, shape: ConvShape<Idx>, x_: Rc<ArrayOp<V>>, b_: Rc<ArrayOp<B>>) -> Rc<ConvOp<Idx, A, B, V, Kernel>>;
}

#[derive(Clone)]
pub struct ConvShape<Idx> where Idx: ArrayIndex {
  pub axes:     Idx::Axes,
  pub stride:   Idx,
  pub zero_pad: Idx,
}

pub struct ConvOp<Idx, A, B, V, Kernel> where Idx: ArrayIndex {
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
  kernel:   Kernel,
}

impl<Idx, A, B, V, Kernel> ConvOp<Idx, A, B, V, Kernel> where Idx: ArrayIndex {
  pub fn new(shape: ConvShape<Idx>, a_: Rc<ArrayOp<A>>, x_: Rc<ArrayOp<V>>, b_: Option<Rc<ArrayOp<B>>>, kernel: Kernel, clk_horizon: usize, alloc: Rc<Fn(TxnId, NodeId) -> V>) -> Rc<Self> {
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
      kernel:   kernel,
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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
pub enum BatchNormAverage {
  Geometric(f64),
  Arithmetic,
  ArithmeticCountCutoff(usize),
  ArithmeticRateCutoff(f64),
}

impl BatchNormAverage {
  pub fn rate(self, update_ct: usize) -> f64 {
    let n = (update_ct + 1) as f64;
    match self {
      BatchNormAverage::Geometric(rate) => rate,
      BatchNormAverage::Arithmetic => 1.0 / n,
      BatchNormAverage::ArithmeticCountCutoff(max_ct) => {
        if update_ct >= max_ct { 0.0 }
        else { 1.0 / n }
      }
      BatchNormAverage::ArithmeticRateCutoff(max_rate) => {
        let rate = 1.0 / n;
        if rate >= max_rate { 0.0 }
        else { rate }
      }
    }
  }
}

#[derive(Clone, Copy)]
pub enum BatchNormUnbias {
  Normalize,
}

#[derive(Clone)]
pub struct BatchNormConfig {
  pub average:  BatchNormAverage,
  pub unbias:   BatchNormUnbias,
}

#[derive(Clone, Copy)]
pub enum BatchNormMode {
  DiffThrough,
  UseRunningStats,
}

pub struct BatchNormState<Idx> where Idx: ArrayIndex {
  pub axes:         Idx::Axes,
  pub cfg:          BatchNormConfig,
  pub mode:         BatchNormMode,
  pub batch_ct:     usize,
  pub update_ct:    usize,
}

pub struct BatchNormControl {
  ops:  Vec<Rc<BatchNormOpExt>>,
}

impl BatchNormControl {
  pub fn configure(&self, f: &Fn(&mut BatchNormConfig)) {
    for op in self.ops.iter() {
      op._configure(f);
    }
  }

  pub fn set_mode(&self, mode: BatchNormMode) {
    for op in self.ops.iter() {
      op._set_mode(mode);
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
}

pub trait BatchNormOpExt {
  fn _configure(&self, f: &Fn(&mut BatchNormConfig));
  fn _set_mode(&self, mode: BatchNormMode);
  fn _accumulate(&self, txn: TxnId);
  fn _update_stats(&self, prev_txn: TxnId, next_txn: TxnId);
}

pub struct BatchNormOp<Idx, A, M> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  state:    Rc<RefCell<BatchNormState<Idx>>>,
  x_:       Rc<ArrayOp<A>>,
  x:        ArrayData<A>,
  mean:     ArrayData<M>,
  mean_acc: ArrayData<M>,
  mean_run: ArrayData<M>,
  var:      ArrayData<M>,
  var_acc:  ArrayData<M>,
  var_run:  ArrayData<M>,
  mean_op:      Weak<PassOp<M>>,
  mean_acc_op:  Rc<ArraySrc<M>>,
  mean_run_op:  Rc<ArraySrc<M>>,
  var_op:       Weak<PassOp<M>>,
  var_acc_op:   Rc<ArraySrc<M>>,
  var_run_op:   Rc<ArraySrc<M>>,
}

impl<Idx, A, M> BatchNormOp<Idx, A, M> where Idx: ArrayIndex {
  pub fn mean(&self) -> Rc<PassOp<M>> {
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
  }
}

//impl<Idx, A, M> BatchNormOpExt for BatchNormOp<Idx, A, M> where Idx: ArrayIndex {
impl<S> BatchNormOpExt for BatchNormOp<usize, BatchArray3d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _configure(&self, f: &Fn(&mut BatchNormConfig)) {
    // FIXME(20170214): only safe to mutate state at the beginning of a txn.
    let mut state = self.state.borrow_mut();
    f(&mut state.cfg);
  }

  fn _set_mode(&self, mode: BatchNormMode) {
    // FIXME(20170214): only safe to mutate state at the beginning of a txn.
    let mut state = self.state.borrow_mut();
    state.mode = mode;
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
    let rate = state.cfg.average.rate(state.update_ct) as f32;
    // FIXME: rather than directly average with `rate`, should use a
    // normalized rate for bias correction.
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

impl<S> AutodiffOp for BatchNormOp<usize, BatchArray3d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
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

pub struct BatchJoinOp<A, Scalar, JoinF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x_:   Rc<ArrayOp<A>>,
  x:    ArrayData<A>,
  y:    ArrayData<Scalar>,
  kernel:   JoinF,
}

pub trait BatchSumExt<A, Scalar> {
  fn batch_sum(x_: Rc<ArrayOp<A>>) -> Rc<BatchJoinOp<A, Scalar, SumJoinKernel>>;
}

impl BatchSumExt<Batch<f32>, f32> for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn batch_sum(x_: Rc<ArrayOp<Batch<f32>>>) -> Rc<Self> {
    unimplemented!();
  }
}

impl AutodiffObjective for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if !self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}

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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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

  fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
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
  x:        ArrayData<A>,
  target:   ArrayData<Target>,
  loss:     ArrayData<Loss>,
  link:     Link,
}

pub trait SoftmaxKLLossExt<A, L> {
  fn softmax_kl2_loss(x: Rc<ArrayOp<A>>, target: Rc<ArrayOp<A>>) -> Rc<SoftmaxLoss<A, A, L, KL2LossLink>>;
}

impl<S, Target, Loss, Link> AutodiffOp for SoftmaxLoss<BatchArray1d<f32, S>, Target, Loss, Link> where S: DerefMut<Target=[f32]>, Link: LikelihoodLossLink {
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

  default fn _rollover(&self, txn: TxnId, vars: &mut VarSet) {
    self.loss.rollover_all(txn, vars);
  }

  default fn _forward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOp for SoftmaxLoss<BatchArray1d<f32, S>, BatchArray1d<f32, S>, Batch<f32>, KL2LossLink> where S: DerefMut<Target=[f32]> {
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
