use prelude::*;

use densearray::prelude::*;

use std::any::{Any};
use std::cell::{Cell, RefCell, Ref, RefMut};
use std::collections::{HashSet};
use std::marker::{PhantomData};
use std::ops::{Deref, DerefMut};
use std::rc::{Rc, Weak};

pub fn var<A, F>(cons: F) -> Var<A, F> where F: Fn() -> A {
  unimplemented!();
}

pub struct Var<A, F> where F: Fn() -> A {
  cons: F,
  data: Rc<ArrayData<A>>,
}

pub fn initializer<A, F>(x: Rc<ArrayOp<A>>, f: F) -> InitializeOp<A, F> where F: Fn(&mut A) {
  unimplemented!();
}

pub struct InitializeOp<A, InitF> {
  x:    Rc<ArrayOp<A>>,
  xdat: Rc<ArrayData<A>>,
  init: InitF,
}

pub fn flatten<A, B>(x: Rc<ArrayOp<A>>) -> FlattenOp<A, B> {
  unimplemented!();
}

pub struct FlattenOp<A, B> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<ArrayOp<A>>,
  y:    Rc<ArrayData<B>>,
}

pub fn rectify<A>(x: Rc<ArrayOp<A>>) -> MapOp<A, RectMapKernel> {
  unimplemented!();
}

pub fn tanh<A>(x: Rc<ArrayOp<A>>) -> MapOp<A, TanhMapKernel> {
  unimplemented!();
}

pub struct RectMapKernel;
pub struct TanhMapKernel;

pub struct MapOp<A, MapK> {
  node_id:  NodeId,
  stack:    OperatorStack,
  x:    Rc<ArrayOp<A>>,
  y:    Rc<ArrayData<A>>,
  _mrk: PhantomData<MapK>,
}

pub fn linear_map<A, V>(a: Rc<ArrayOp<A>>, x: Rc<ArrayOp<V>>) -> LinearOp<A, V> {
  unimplemented!();
}

pub fn affine_map<A, V>(a: Rc<ArrayOp<A>>, x: Rc<ArrayOp<V>>, b: Rc<ArrayOp<V>>) -> LinearOp<A, V> {
  unimplemented!();
}

pub struct LinearOp<A, V> {
  node_id:  NodeId,
  stack:    OperatorStack,
  a:    Rc<ArrayOp<A>>,
  x:    Rc<ArrayOp<V>>,
  b:    Option<Rc<ArrayOp<V>>>,
  y:    Rc<ArrayData<V>>,
}

impl<S> DiffOperator for LinearOp<Array2d<f32, S>, Array1d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.a._push(epoch, apply);
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
      self.a._pop(epoch, apply);
    }
  }

  fn _clear(&self, txn: TxnId) {
    let node = self._id();
    if let Some(mut val) = self.y.val.maybe_get_mut(txn, node) {
      val.as_view_mut().set_constant(0.0);
    }
    if let Some(mut grad) = self.y.grad.maybe_get_mut(txn, node) {
      grad.as_view_mut().set_constant(0.0);
    }
    if let Some(mut r_val) = self.y.r_val.maybe_get_mut(txn, node) {
      r_val.as_view_mut().set_constant(0.0);
    }
    if let Some(mut r_grad) = self.y.r_grad.maybe_get_mut(txn, node) {
      r_grad.as_view_mut().set_constant(0.0);
    }
  }

  fn _forward(&self, txn: TxnId) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if self.y.val.invalidate(txn, node) {
      self.y.val.maybe_alloc(|_val| {});
      self.y.val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn).as_view(), Transpose::N,
          x.val.get(txn).as_view(),
          0.0,
      );
      if let Some(ref b) = self.b {
        let b = b.data();
        self.y.val.get_mut(txn, node).as_view_mut().add(1.0, b.val.get(txn).as_view());
      }
    }
  }

  fn _backward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    let x_dim = x.val.get(txn).dim();
    let y_dim = self.y.val.get(txn).dim();
    if a.grad.invalidate(txn, node) {
      a.grad.maybe_alloc(|grad| grad.as_view_mut().set_constant(0.0));
      a.grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn).as_view().reshape((y_dim, 1)), Transpose::N,
          x.val.get(txn).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
    }
    if x.grad.invalidate(txn, node) {
      x.grad.maybe_alloc(|grad| grad.as_view_mut().set_constant(0.0));
      x.grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn).as_view(), Transpose::T,
          self.y.grad.get(txn).as_view(),
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      let b = b.data();
      if b.grad.invalidate(txn, node) {
        b.grad.maybe_alloc(|grad| grad.as_view_mut().set_constant(0.0));
        b.grad.get_mut(txn, node).as_view_mut().copy(self.y.grad.get(txn).as_view());
      }
    }
  }

  fn _r_forward(&self, txn: TxnId, _gauss_newton: bool) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if self.y.r_val.invalidate(txn, node) {
      self.y.r_val.maybe_alloc(|_r_val| {});
      self.y.r_val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.r_val.get(txn).as_view(), Transpose::N,
          x.val.get(txn).as_view(),
          0.0,
      );
      self.y.r_val.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn).as_view(), Transpose::N,
          x.r_val.get(txn).as_view(),
          1.0,
      );
      if let Some(ref b) = self.b {
        let b = b.data();
        self.y.r_val.get_mut(txn, node).as_view_mut().add(1.0, b.r_val.get(txn).as_view());
      }
    }
  }

  fn _r_backward(&self, txn: TxnId) {
    let node = self._id();
    let a = self.a.data();
    let x = self.x.data();
    if a.r_grad.invalidate(txn, node) {
      let x_dim = x.val.get(txn).dim();
      let y_dim = self.y.grad.get(txn).dim();
      a.r_grad.maybe_alloc(|r_grad| r_grad.as_view_mut().set_constant(0.0));
      a.r_grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.r_grad.get(txn).as_view().reshape((y_dim, 1)), Transpose::N,
          x.val.get(txn).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
      a.r_grad.get_mut(txn, node).as_view_mut().matrix_prod(
          1.0,
          self.y.grad.get(txn).as_view().reshape((y_dim, 1)), Transpose::N,
          x.r_val.get(txn).as_view().reshape((x_dim, 1)), Transpose::T,
          1.0,
      );
    }
    if x.r_grad.invalidate(txn, node) {
      x.r_grad.maybe_alloc(|r_grad| r_grad.as_view_mut().set_constant(0.0));
      x.r_grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.r_val.get(txn).as_view(), Transpose::T,
          self.y.grad.get(txn).as_view(),
          1.0,
      );
      x.r_grad.get_mut(txn, node).as_view_mut().matrix_vector_prod(
          1.0,
          a.val.get(txn).as_view(), Transpose::T,
          self.y.r_grad.get(txn).as_view(),
          1.0,
      );
    }
    if let Some(ref b) = self.b {
      let b = b.data();
      if b.r_grad.invalidate(txn, node) {
        b.r_grad.maybe_alloc(|r_grad| r_grad.as_view_mut().set_constant(0.0));
        b.r_grad.get_mut(txn, node).as_view_mut().copy(self.y.r_grad.get(txn).as_view());
      }
    }
  }
}

pub fn conv<Idx, A, B, V>(shape: ConvShape<Idx>, a: Rc<ArrayOp<A>>, x: Rc<ArrayOp<V>>, b: Option<Rc<ArrayOp<B>>>) -> Rc<ArrayOp<V>> where Idx: ArrayIndex {
  unimplemented!();
}

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

impl<S> DiffOperator for ConvOp<(usize, usize), Array4d<f32, S>, Array1d<f32, S>, Array3d<f32, S>> where S: DerefMut<Target=[f32]> {
  fn _id(&self) -> NodeId {
    self.node_id
  }

  fn _push(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator)) {
    if 1 == self.stack.push(epoch) {
      self.a._push(epoch, apply);
      self.x._push(epoch, apply);
      apply(self);
    }
  }

  fn _pop(&self, epoch: Epoch, apply: &mut FnMut(&DiffOperator)) {
    if self.stack.degree(epoch) == self.stack.pop(epoch) {
      apply(self);
      self.x._pop(epoch, apply);
      self.a._pop(epoch, apply);
    }
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
