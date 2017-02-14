use prelude::*;
use ffi::*;

use densearray::prelude::*;

use std::any::{Any, /*TypeId*/};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashSet};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

//const VEC_F32_TYPEID: TypeId = TypeId::of::<Vec<f32>>();

/*pub fn constant<A, F>(cons: F) -> Rc<Constant<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  let alloc: Rc<Fn(TxnId, NodeId) -> A> = Rc::new(cons);
  Rc::new(Constant{data: ArrayData::new(1, alloc)})
}

pub struct Constant<A> {
  data: Rc<ArrayData<A>>,
}*/

pub fn var<A, F>(cons: F) -> Rc<Var<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  Var::new(1, false, Rc::new(cons))
}

pub fn sequential_var<A, F>(horizon: usize, cons: F) -> Rc<Var<A>> where F: 'static + Fn(TxnId, NodeId) -> A {
  Var::new(horizon, true, Rc::new(cons))
}

/*pub fn test_var() {
  let x: Rc<Var<Array1d<f32>>> = var(|_, _| Array1d::zeros(10));
}*/

pub struct Var<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  data:     Rc<ArrayData<A>>,
  clock:    bool,
}

impl<A> Var<A> {
  pub fn new(horizon: usize, clock: bool, alloc: Rc<Fn(TxnId, NodeId) -> A>) -> Rc<Self> {
    let node = NodeId::new();
    Rc::new(Var{
      node_id:  node,
      stack:    OperatorStack::new(node, 0),
      data:     ArrayData::new(horizon, alloc),
      clock:    clock,
    })
  }
}

impl<A> ArrayOp<A> for Var<A> {
  fn data(&self) -> Rc<ArrayData<A>> {
    self.data.clone()
  }
}

impl<A> AutodiffOperator for Var<A> {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
    }
  }

  default fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.data.rollover_all(txn, ref_set);
  }

  default fn _forward(&self, txn: TxnId) {
  }

  default fn _backward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  default fn _r_forward(&self, _txn: TxnId, _gauss_newton: bool) {
  }

  default fn _r_backward(&self, _txn: TxnId) {
  }

  default fn _reset_clock(&self) {
    if self.clock {
      self.data.reset_clock_all();
    }
  }

  default fn _set_clock(&self, clk: usize) {
    if self.clock {
      self.data.set_clock_all(clk);
    }
  }
}

impl AutodiffOperator for Var<Array1d<f32>> {
  fn _load(&self, txn: TxnId, ref_set: &mut DataRefSet, reader: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if self.data.val.write(txn, node) {
        /*match reader.get_type_id() {
          VEC_F32_TYPEID => {}
          _ => {}
        }*/
        if let Some(reader) = reader.downcast_mut::<Vec<f32>>() {
          self.data.val.get_mut(txn, node).as_mut_slice().copy_from_slice(reader);
        } else {
          unimplemented!();
        }
      }
    }
  }

  fn _store(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.val._ref()) {
      if let Some(writer) = writer.downcast_mut::<Vec<f32>>() {
        writer.copy_from_slice(self.data.val.get(txn, node).as_slice());
      } else {
        unimplemented!();
      }
    }
  }

  fn _store_grad(&self, txn: TxnId, ref_set: &mut DataRefSet, writer: &mut Any) {
    let node = self._id();
    if ref_set.contains(&self.data.grad._ref()) {
      if let Some(writer) = writer.downcast_mut::<Vec<f32>>() {
        writer.copy_from_slice(self.data.grad.get(txn, node).as_slice());
      } else {
        unimplemented!();
      }
    }
  }
}

pub struct InitializeOp<A, InitF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<ArrayOp<A>>,
  xdat: Rc<ArrayData<A>>,
  kernel:   InitF,
}

pub trait InitializeExt<A, F, InitF> {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, InitF>>;
}

impl<Op, A, F> InitializeExt<A, F, Rc<F>> for Rc<Op> where Op: 'static + ArrayOp<A>, F: Fn(&mut A) {
  fn initialize(&self, f: F) -> Rc<InitializeOp<A, Rc<F>>> {
    let node = NodeId::new();
    let stack = OperatorStack::new(node, 1);
    Rc::new(InitializeOp{
      node_id:  node,
      stack:    stack,
      x:    self.clone(),
      xdat: self.data(),
      kernel:   Rc::new(f),
    })
  }
}

impl<A, InitF> ArrayOp<A> for InitializeOp<A, InitF> where InitializeOp<A, InitF>: AutodiffOperator {
  fn data(&self) -> Rc<ArrayData<A>> {
    self.xdat.clone()
  }
}

impl<S, F> AutodiffOperator for InitializeOp<Array1d<f32, S>, Rc<F>> where S: DerefMut<Target=[f32]>, F: Fn(&mut Array1d<f32, S>) {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    // Do nothing, `xdat` belongs to `x`.
  }

  fn _init(&self, txn: TxnId) {
    let node = self._id();
    if self.xdat.val.write(txn, node) {
      (self.kernel)(&mut *self.xdat.val.get_mut(txn, node));
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

pub trait SpecialMapExt<T, A> {
  fn exp(&self) -> Rc<MapOp<A, ExpMapKernel>>;
  fn rect(&self) -> Rc<MapOp<A, RectMapKernel>>;
  fn leak_rect(&self, c: T) -> Rc<MapOp<A, LeakRectMapKernel<T>>>;
  fn logistic(&self) -> Rc<MapOp<A, LogisticMapKernel>>;
  fn tanh(&self) -> Rc<MapOp<A, TanhMapKernel>>;
}

/*pub fn rectify<A>(x: Rc<ArrayOp<A>>) -> MapOp<A, RectMapKernel> {
  unimplemented!();
}

pub fn tanh<A>(x: Rc<ArrayOp<A>>) -> MapOp<A, TanhMapKernel> {
  unimplemented!();
}*/

pub struct MapOp<A, MapF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<ArrayOp<A>>,
  y:    Rc<ArrayData<A>>,
  kernel:   MapF,
}

impl<A, MapF> MapOp<A, MapF> {
  pub fn new<F>(kernel: MapF, x: Rc<ArrayOp<A>>, clk_horizon: usize, alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> A {
    let node = NodeId::new();
    Rc::new(MapOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc),
      kernel:   kernel,
    })
  }
}

impl<S, MapF> AutodiffOperator for MapOp<Array1d<f32, S>, MapF> where S: DerefMut<Target=[f32]>, MapF: SpecialMapKernel {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
    }
  }

  default fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  default fn _forward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOperator for MapOp<Array1d<f32, S>, RectMapKernel> where S: DerefMut<Target=[f32]> {
  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      let x_dim = x.val.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_fwd_f32(
          x_dim,
          x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_rect_bwd_f32(
          y_dim,
          x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
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

impl<S> AutodiffOperator for MapOp<Array1d<f32, S>, LogisticMapKernel> where S: DerefMut<Target=[f32]> {
  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      let x_dim = x.val.get(txn, node).dim();
      unsafe { arraydiff_kernel_logistic_fwd_f32(
          x_dim,
          x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_logistic_bwd_f32(
          y_dim,
          x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
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

impl<S> AutodiffOperator for MapOp<Array1d<f32, S>, TanhMapKernel> where S: DerefMut<Target=[f32]> {
  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      let x_dim = x.val.get(txn, node).dim();
      unsafe { arraydiff_kernel_tanh_fwd_f32(
          x_dim,
          x.val.get(txn, node).as_view().as_ptr(),
          self.y.val.get_mut(txn, node).as_view_mut().as_mut_ptr(),
      ) };
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let y_dim = self.y.grad.get(txn, node).dim();
      unsafe { arraydiff_kernel_tanh_bwd_f32(
          y_dim,
          x.val.get(txn, node).as_view().as_ptr(),
          self.y.grad.get(txn, node).as_view().as_ptr(),
          x.grad.get_mut(txn, node).as_view_mut().as_mut_ptr(),
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
  x:    Rc<ArrayOp<A>>,
  y:    Rc<ArrayData<B>>,
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
  pub fn new<F>(x: Rc<ArrayOp<Array3d<f32, S>>>, clk_horizon: usize, alloc: Rc<F>) -> Rc<Self> where F: 'static + Fn(TxnId, NodeId) -> Array1d<f32, S> {
    let node = NodeId::new();
    Rc::new(TransformOp{
      node_id:  node,
      stack:    OperatorStack::new(node, 1),
      x:    x,
      y:    ArrayData::new(clk_horizon, alloc),
      kernel:   FlattenTransform,
    })
  }
}

impl<S> AutodiffOperator for TransformOp<Array3d<f32, S>, Array1d<f32, S>, FlattenTransform> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      self.y.val.get_mut(txn, node).as_view_mut().copy(x.val.get(txn, node).as_view().flatten());
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      x.grad.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.y.grad.get(txn, node).as_view());
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    if self.y.r_val.write(txn, node) {
      self.y.r_val.get_mut(txn, node).as_view_mut().copy(x.r_val.get(txn, node).as_view().flatten());
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    if x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      x.r_grad.get_mut(txn, node).as_view_mut().flatten_mut().add(1.0, self.y.r_grad.get(txn, node).as_view());
    }
  }
}

pub struct JoinOp<A, JoinF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  xs:   Vec<Rc<ArrayOp<A>>>,
  y:    Rc<ArrayData<A>>,
  kernel:   JoinF,
}

pub struct AxisJoinKernel;
pub struct SumJoinKernel;

pub trait AxisJoinExt<A> {
  fn axis_join(xs: Vec<Rc<ArrayOp<A>>>) -> Rc<JoinOp<A, AxisJoinKernel>>;
}

pub trait AddExt<A> {
  fn add(&self, x: Rc<ArrayOp<A>>) -> Rc<JoinOp<A, SumJoinKernel>>;
}

pub trait SumExt<A> {
  fn sum(xs: Vec<Rc<ArrayOp<A>>>) -> Rc<JoinOp<A, SumJoinKernel>>;
}

impl<S> AxisJoinExt<Array1d<f32, S>> for JoinOp<Array1d<f32, S>, AxisJoinKernel> where S: DerefMut<Target=[f32]> {
  fn axis_join(xs: Vec<Rc<ArrayOp<Array1d<f32, S>>>>) -> Rc<Self> {
    unimplemented!();
  }
}

impl<S> AddExt<Array1d<f32, S>> for Rc<ArrayOp<Array1d<f32, S>>> where S: DerefMut<Target=[f32]> {
  fn add(&self, x: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<JoinOp<Array1d<f32, S>, SumJoinKernel>> {
    unimplemented!();
  }
}

impl<S> SumExt<Array1d<f32, S>> for JoinOp<Array1d<f32, S>, SumJoinKernel> where S: DerefMut<Target=[f32]> {
  fn sum(xs: Vec<Rc<ArrayOp<Array1d<f32, S>>>>) -> Rc<Self> {
    unimplemented!();
  }
}

impl<S> AutodiffOperator for JoinOp<Array1d<f32, S>, SumJoinKernel> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      for x in self.xs.iter() {
        x._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      for x in self.xs.iter().rev() {
        x._pop(epoch, apply);
      }
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  /*fn _clear(&self, txn: TxnId) {
    self.y.val.invalidate();
    self.y.grad.invalidate();
    self.y.r_val.invalidate();
    self.y.r_grad.invalidate();
  }*/

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    if self.y.val.write(txn, node) {
      let x0 = self.xs[0].data();
      self.y.val.get_mut(txn, node).as_view_mut().copy(x0.val.get(txn, node).as_view());
      for x in self.xs.iter().skip(1) {
        let x = x.data();
        self.y.val.get_mut(txn, node).as_view_mut().add(1.0, x.val.get(txn, node).as_view());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    for x in self.xs.iter() {
      let x = x.data();
      if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
        x.grad.get_mut(txn, node).as_view_mut().add(1.0, self.y.grad.get(txn, node).as_view());
      }
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    if self.y.r_val.write(txn, node) {
      let x0 = self.xs[0].data();
      self.y.r_val.get_mut(txn, node).as_view_mut().copy(x0.r_val.get(txn, node).as_view());
      for x in self.xs.iter().skip(1) {
        let x = x.data();
        self.y.r_val.get_mut(txn, node).as_view_mut().add(1.0, x.r_val.get(txn, node).as_view());
      }
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    for x in self.xs.iter() {
      let x = x.data();
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
  y:    Rc<ArrayData<A>>,
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
  fn multiply(&self, x: Rc<ArrayOp<V>>) -> Rc<LinearOp<A, V, W, B>>;
  fn multiply_add(&self, x: Rc<ArrayOp<V>>, b: Rc<ArrayOp<B>>) -> Rc<LinearOp<A, V, W, B>>;
}

pub struct LinearOp<A, V, W, B> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a:    Rc<ArrayOp<A>>,
  x:    Rc<ArrayOp<V>>,
  b:    Option<Rc<ArrayOp<B>>>,
  y:    Rc<ArrayData<W>>,
}

impl<A, V, W, B> LinearOp<A, V, W, B> {
  pub fn new<F>(a: Rc<ArrayOp<A>>, x: Rc<ArrayOp<V>>, b: Option<Rc<ArrayOp<B>>>, clk_horizon: usize, alloc: Rc<F>) -> Rc<LinearOp<A, V, W, B>> where F: 'static + Fn(TxnId, NodeId) -> W {
    let node = NodeId::new();
    let in_degree = match b {
      None    => 2,
      Some(_) => 3,
    };
    Rc::new(LinearOp{
      node_id:  node,
      stack:    OperatorStack::new(node, in_degree),
      a:    a,
      x:    x,
      b:    b,
      y:    ArrayData::new(clk_horizon, alloc),
    })
  }
}

impl<Op, S> MultiplyExt<Array1d<f32, S>, Array1d<f32, S>, f32, f32> for Rc<Op> where Op: 'static + ArrayOp<Array1d<f32, S>>, S: DerefMut<Target=[f32]> {
  fn multiply(&self, x: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32>> {
    let clk_horizon = x.data().horizon();
    LinearOp::new(self.clone(), x, None, clk_horizon, Rc::new(|_, _| 0.0_f32))
  }

  fn multiply_add(&self, x: Rc<ArrayOp<Array1d<f32, S>>>, b: Rc<ArrayOp<f32>>) -> Rc<LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32>> {
    let clk_horizon = x.data().horizon();
    LinearOp::new(self.clone(), x, Some(b), clk_horizon, Rc::new(|_, _| 0.0_f32))
  }
}

impl<S> ArrayOp<f32> for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
  fn data(&self) -> Rc<ArrayData<f32>> {
    self.y.clone()
  }
}

impl<S> AutodiffObjective for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    //println!("DEBUG: LinearOp: set source");
    if !self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}

impl<S> AutodiffOperator for LinearOp<Array1d<f32, S>, Array1d<f32, S>, f32, f32> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.a._push(epoch, apply);
      self.x._push(epoch, apply);
      if let Some(ref b) = self.b {
        b._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref b) = self.b {
        b._pop(epoch, apply);
      }
      self.x._pop(epoch, apply);
      self.a._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      *self.y.val.get_mut(txn, node) = a.val.get(txn, node).as_view().inner_prod(1.0, x.val.get(txn, node).as_view());
      if let Some(ref b) = self.b {
        let b = b.data();
        *self.y.val.get_mut(txn, node) += *b.val.get(txn, node);
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      //println!("DEBUG: LinearOp: backward");
      a.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), x.val.get(txn, node).as_view());
    }
    if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      x.grad.get_mut(txn, node).as_view_mut().add(*self.y.grad.get(txn, node), a.val.get(txn, node).as_view());
    }
    if let Some(ref b) = self.b {
      let b = b.data();
      if b.grad.accumulate(txn, node, |g| *g = 0.0) {
        *b.grad.get_mut(txn, node) += *self.y.grad.get(txn, node);
      }
    }
  }
}

impl<S> MultiplyExt<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> for Rc<ArrayOp<Array2d<f32, S>>> where S: 'static + DerefMut<Target=[f32]> + ArrayStorage<usize> {
  fn multiply(&self, x: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>>> {
    let clk_horizon = x.data().horizon();
    LinearOp::new(self.clone(), x.clone(), None, clk_horizon, {
      let x = x.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }

  fn multiply_add(&self, x: Rc<ArrayOp<Array1d<f32, S>>>, b: Rc<ArrayOp<Array1d<f32, S>>>) -> Rc<LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>>> {
    let clk_horizon = x.data().horizon();
    LinearOp::new(self.clone(), x.clone(), Some(b), clk_horizon, {
      let x = x.clone().data();
      Rc::new(move |txn, node| {
        let dim = x.val.get(txn, node).dim();
        let buf = <S as ArrayStorage<usize>>::alloc(dim);
        Array1d::from_storage(dim, buf)
      })
    })
  }
}

impl<S> ArrayOp<Array1d<f32, S>> for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn data(&self) -> Rc<ArrayData<Array1d<f32, S>>> {
    self.y.clone()
  }
}

impl<S> AutodiffOperator for LinearOp<Array2d<f32, S>, Array1d<f32, S>, Array1d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.a._push(epoch, apply);
      self.x._push(epoch, apply);
      if let Some(ref b) = self.b {
        b._push(epoch, apply);
      }
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref b) = self.b {
        b._pop(epoch, apply);
      }
      self.x._pop(epoch, apply);
      self.a._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      self.y.val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn, node).as_view(), Transpose::N,
          x.val.get(txn, node).as_view(),
          0.0,
      );
      if let Some(ref b) = self.b {
        let b = b.data();
        self.y.val.get_mut(txn, node).as_view_mut().add(1.0, b.val.get(txn, node).as_view());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if a.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      let x_dim = x.val.get(txn, node).dim();
      let y_dim = self.y.val.get(txn, node).dim();
      a.grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn, node).as_view().reshape((y_dim, 1)), Transpose::N,
          x.val.get(txn, node).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
    }
    if x.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
      x.grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn, node).as_view(), Transpose::T,
          self.y.grad.get(txn, node).as_view(),
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      let b = b.data();
      if b.grad.accumulate(txn, node, |grad| grad.as_view_mut().set_constant(0.0)) {
        b.grad.get_mut(txn, node).as_view_mut().copy(self.y.grad.get(txn, node).as_view());
      }
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if self.y.r_val.write(txn, node) {
      self.y.r_val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.r_val.get(txn, node).as_view(), Transpose::N,
          x.val.get(txn, node).as_view(),
          0.0,
      );
      self.y.r_val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn, node).as_view(), Transpose::N,
          x.r_val.get(txn, node).as_view(),
          1.0,
      );
      if let Some(ref b) = self.b {
        let b = b.data();
        self.y.r_val.get_mut(txn, node).as_view_mut().add(1.0, b.r_val.get(txn, node).as_view());
      }
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if a.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      let x_dim = x.val.get(txn, node).dim();
      let y_dim = self.y.grad.get(txn, node).dim();
      a.r_grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.r_grad.get(txn, node).as_view().reshape((y_dim, 1)), Transpose::N,
          x.val.get(txn, node).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
      a.r_grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn, node).as_view().reshape((y_dim, 1)), Transpose::N,
          x.r_val.get(txn, node).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
    }
    if x.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
      x.r_grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.r_val.get(txn, node).as_view(), Transpose::T,
          self.y.grad.get(txn, node).as_view(),
          1.0,
      );
      x.r_grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn, node).as_view(), Transpose::T,
          self.y.r_grad.get(txn, node).as_view(),
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      let b = b.data();
      if b.r_grad.accumulate(txn, node, |r_grad| r_grad.as_view_mut().set_constant(0.0)) {
        b.r_grad.get_mut(txn, node).as_view_mut().add(1.0, self.y.r_grad.get(txn, node).as_view());
      }
    }
  }
}

pub trait ConvExt<Idx, A, B, V>: ArrayOp<A> where Idx: ArrayIndex {
  fn conv(&self, shape: ConvShape<Idx>, x: Rc<ArrayOp<V>>) -> Rc<ConvOp<Idx, A, B, V>>;
  fn conv_add(&self, shape: ConvShape<Idx>, x: Rc<ArrayOp<V>>, b: Rc<ArrayOp<B>>) -> Rc<ConvOp<Idx, A, B, V>>;
}

/*pub fn conv<Idx, A, B, V>(shape: ConvShape<Idx>, a: Rc<ArrayOp<A>>, x: Rc<ArrayOp<V>>, b: Option<Rc<ArrayOp<B>>>) -> Rc<ArrayOp<V>> where Idx: ArrayIndex {
  unimplemented!();
}*/

#[derive(Clone)]
pub struct ConvShape<Idx> where Idx: ArrayIndex {
  pub axes:     Idx::Axes,
  pub stride:   Idx,
  pub zero_pad: Idx,
}

pub struct ConvOp<Idx, A, B, V> where Idx: ArrayIndex {
  node_id:  NodeId,
  stack:    OperatorStack,
  shape:    ConvShape<Idx>,
  a:    Rc<ArrayOp<A>>,
  x:    Rc<ArrayOp<V>>,
  b:    Option<Rc<ArrayOp<B>>>,
  y:    Rc<ArrayData<V>>,
}

impl<S> AutodiffOperator for ConvOp<(usize, usize), Array4d<f32, S>, Array1d<f32, S>, Array3d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.a._push(epoch, apply);
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
      self.a._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let a = self.a.data();
    let x = self.x.data();
    // TODO
    if let Some(ref b) = self.b {
      let b = b.data();
    }
    unimplemented!();
  }

  fn _backward(&self, txn: TxnId, gauss_newton: bool) {
    // TODO
    unimplemented!();
  }
}

pub struct BatchJoinOp<A, Scalar, JoinF> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<ArrayOp<A>>,
  y:    Rc<ArrayData<Scalar>>,
  kernel:   JoinF,
}

pub trait BatchSumExt<A, Scalar> {
  fn batch_sum(x: Rc<ArrayOp<A>>) -> Rc<BatchJoinOp<A, Scalar, SumJoinKernel>>;
}

impl BatchSumExt<Batch<f32>, f32> for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn batch_sum(x: Rc<ArrayOp<Batch<f32>>>) -> Rc<Self> {
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

impl AutodiffOperator for BatchJoinOp<Batch<f32>, f32, SumJoinKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    if self.y.val.write(txn, node) {
      let x_val = x.val.get(txn, node);
      let mut y_val = self.y.val.get_mut(txn, node);
      *y_val = 0.0;
      let batch_sz = x_val.batch_size();
      for i in 0 .. batch_sz {
        *y_val += x_val[i];
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    let batch_sz = x.val.get(txn, node).batch_size();
    if x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
      let mut x_grad = x.grad.get_mut(txn, node);
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
  x:    Rc<ArrayOp<A>>,
  y:    Rc<ArrayData<B>>,
  kernel:   JoinF,
}

impl AutodiffOperator for SequentialJoinOp<Batch<f32>, Batch<f32>, SumJoinKernel> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.y.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    let x_val = x.val.get(txn, node);
    let batch_sz = x_val.batch_size();
    if self.y.val.accumulate(txn, node, |val| val.reshape_mut(batch_sz).set_constant(0.0)) {
      self.y.val.get_mut(txn, node).reshape_mut(batch_sz)
        .add(1.0, x_val.reshape(batch_sz));
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let x = self.x.data();
    let y_grad = self.y.grad.get(txn, node);
    let batch_sz = y_grad.batch_size();
    if x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
      x.grad.get_mut(txn, node).reshape_mut(batch_sz)
        .add(1.0, y_grad.reshape(batch_sz));
    }
  }

  fn _reset_clock(&self) {
  }

  fn _set_clock(&self, _clk: usize) {
  }
}

pub struct LstSqLoss<A> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:        Rc<ArrayOp<A>>,
  target:   Rc<ArrayOp<A>>,
  loss:     Rc<ArrayData<A>>,
}

/*impl AutodiffObjective for LstSqLoss<f32> {
  fn _set_source(&self, txn: TxnId) {
    let node = self._id();
    if !self.y.grad.accumulate(txn, node, |g| *g = 1.0) {
      assert_eq!(1.0, *self.y.grad.get_mut(txn, node));
    }
  }
}*/

impl AutodiffOperator for LstSqLoss<Batch<f32>> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      self.target._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.target._pop(epoch, apply);
      self.x._pop(epoch, apply);
    }
  }

  fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.loss.rollover_all(txn, ref_set);
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let x = self.x.data();
    let target = self.target.data();
    let batch_sz = x.val.get(txn, node).batch_size();
    if self.loss.val.write(txn, node) {
      let mut loss_val = self.loss.val.get_mut(txn, node);
      loss_val.set_batch_size(batch_sz, 0.0);
      loss_val.reshape_mut(batch_sz).copy(x.val.get(txn, node).reshape(batch_sz));
      loss_val.reshape_mut(batch_sz).add(-1.0, target.val.get(txn, node).reshape(batch_sz));
      loss_val.reshape_mut(batch_sz).square();
      loss_val.reshape_mut(batch_sz).scale(0.5);
    }
  }

  fn _backward(&self, txn: TxnId, gauss_newton: bool) {
    let node = self._id();
    if gauss_newton {
      unimplemented!();
    } else {
      let x = self.x.data();
      let target = self.target.data();
      let batch_sz = x.val.get(txn, node).batch_size();
      if x.grad.accumulate(txn, node, |grad| grad.reshape_mut(batch_sz).set_constant(0.0)) {
        x.grad.get_mut(txn, node).set_batch_size(batch_sz, 0.0);
        // FIXME(20170209): mix with `loss.grad`; requires a tmp variable.
        x.grad.get_mut(txn, node).reshape_mut(batch_sz).add(1.0, x.val.get(txn, node).reshape(batch_sz));
        x.grad.get_mut(txn, node).reshape_mut(batch_sz).add(-1.0, target.val.get(txn, node).reshape(batch_sz));
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

pub struct SoftmaxLoss<A, Target, Link> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:        Rc<ArrayOp<A>>,
  target:   Option<Rc<ArrayOp<Target>>>,
  loss:     Rc<ArrayData<A>>,
  link:     Link,
}

impl<S, Target, Link> AutodiffOperator for SoftmaxLoss<BatchArray1d<f32, S>, Target, Link> where S: DerefMut<Target=[f32]>, Link: LikelihoodLossLink {
  default fn _id(&self) -> NodeId {
    self.node_id
  }

  default fn _push(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.x._push(epoch, apply);
      if let Some(ref target) = self.target {
        target._push(epoch, apply);
      }
      apply(self);
    }
  }

  default fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&AutodiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      if let Some(ref target) = self.target {
        target._pop(epoch, apply);
      }
      self.x._pop(epoch, apply);
    }
  }

  default fn _rollover(&self, txn: TxnId, ref_set: &mut DataRefSet) {
    self.loss.rollover_all(txn, ref_set);
  }

  default fn _forward(&self, txn: TxnId) {
    unimplemented!();
  }
}

impl<S> AutodiffOperator for SoftmaxLoss<BatchArray1d<f32, S>, BatchArray1d<f32, S>, KL2LossLink> where S: DerefMut<Target=[f32]> {
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

impl<S> AutodiffOperator for SoftmaxLoss<BatchArray1d<f32, S>, Vec<(u32, f32)>, LRLossLink> where S: DerefMut<Target=[f32]> {
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

impl<S> AutodiffOperator for SoftmaxLoss<BatchArray1d<f32, S>, Vec<u32>, NLLLossLink> where S: DerefMut<Target=[f32]> {
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
