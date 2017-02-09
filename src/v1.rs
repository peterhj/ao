extern crate arithmetic;
extern crate densearray;
extern crate operator;

use arithmetic::*;
use densearray::prelude::*;
use operator::prelude::*;

use std::cell::{Cell};
use std::marker::{PhantomData};
use std::rc::{Rc};

pub mod prelude;
//pub mod v2;
pub mod v3;

#[derive(Clone, Copy, Debug)]
pub enum ShapeDesc {
  Dim(usize),
  Batch(usize),
  X(usize),
  Y(usize),
  Z(usize),
  T(usize),
}

pub enum Param<A> {
  New,
  Shared(Rc<ParamBlock<A>>),
}

pub trait ArrayDiffOperator<A, S, IoBuf: ?Sized>: DiffOperator<S, IoBuf> {
  fn _out(&self, arm: usize) -> Rc<ArrayDiffVar<A>>;
}

pub struct ArrayDiffVar<A> {
  pub clock:    Cell<usize>,
  pub data:     Vec<Rc<VarBlock<A>>>,
}

impl<T> ArrayDiffVar<Vec<T>> where T: 'static + Copy + PseudoField {
  pub fn buf(dim: usize, batch_sz: usize) -> ArrayDiffVar<Vec<T>> {
    ArrayDiffVar{
      clock:    Cell::new(0),
      data:     vec![VarBlock::new(DefaultVarAllocator::new(move || {
        let mut mem = Vec::with_capacity(dim * batch_sz);
        mem.resize(dim * batch_sz, T::zero());
        mem
      }))],
    }
  }
}

impl<A> ArrayDiffVar<A> {
  /*pub fn recursive(horizon: usize, trunc_horizon: Option<usize>) -> ArrayDiffVar<A> {
    assert!(horizon >= 1);
    if let Some(trunc_horizon) = trunc_horizon {
      assert!(trunc_horizon >= 1);
      assert!(trunc_horizon <= horizon);
    }
    unimplemented!();
  }*/

  pub fn _var(&self, t: usize) -> Rc<VarBlock<A>> {
    self.data[t].clone()
  }

  pub fn var(&self) -> Rc<VarBlock<A>> {
    self.data[self.clock.get()].clone()
  }

  pub fn prev_var(&self) -> Option<Rc<VarBlock<A>>> {
    match self.clock.get() {
      0   => None,
      clk => Some(self.data[clk-1].clone()),
    }
  }

  pub fn reset_clock(&self) {
    self.clock.set(0);
  }

  pub fn clock(&self) {
    let next_clk = self.clock.get() + 1;
    assert!(next_clk != 0);
    self.clock.set(next_clk);
  }
}

pub fn constant<A>(value: A) -> ArrayConstant<A> {
  unimplemented!();
}

pub fn param<W, A>(param: Param<W>) -> ArrayParam<W, A> {
  unimplemented!();
}

pub fn input<A>(batch_sz: usize, stride: usize) -> ArrayInput<A> {
  unimplemented!();
}

pub fn input_dim<Idx, A>(batch_sz: usize, dim: Idx) -> ArrayInput<A> {
  unimplemented!();
}

pub fn add_join<S, IoBuf>(ops: Vec<(usize, Rc<DiffOperator<S, IoBuf>>)>) -> ArrayAddJoinOp {
  unimplemented!();
}

pub trait ArrayDiffExt {
  type T: Copy;

  fn copy_split(self, num_copies: usize) -> Vec<(usize, Rc<ArrayCopySplitOp>)>;
  fn scale_const<A, S>(self, c: Self::T) -> Rc<ArrayScaleConstOp<Self::T, A, S>>;
}

pub struct ArrayConstant<A> {
  node:     OperatorNode,
  out:      Rc<ArrayDiffVar<A>>,
}

pub struct ArrayParam<W, A> {
  node:     OperatorNode,
  out:      Rc<ArrayDiffVar<A>>,
  inner:    Rc<ParamBlock<W>>,
}

pub struct ArrayInput<A> {
  node:     OperatorNode,
  out:      Rc<ArrayDiffVar<A>>,
  batch_sz: usize,
  stride:   usize,
}

pub struct ArrayAddJoinOp {
}

pub struct ArrayCopySplitOp {
}

pub struct ArrayScaleConstOp<T, A, S> {
  node:     OperatorNode,
  in_op:    Rc<DiffOperator<S, [T]>>,
  in_:      Rc<ArrayDiffVar<A>>,
  out:      Rc<ArrayDiffVar<A>>,
  scale_c:  T,
}

impl<T, A, S> ArrayScaleConstOp<T, A, S> {
}

/*impl<T> ArrayDiffExt for ArrayScaleConstOp<T> {
  type T = T;
}*/

pub trait VectorDiffExt {
  type T: Copy;

  //fn inner_prod(self, rhs: Self::Vector) -> VectorInnerProdOp<Self::T>;
  fn add<A, S>(self, bias: Param<Array1d<Self::T>>) -> Rc<VectorAddOp<Self::T, A, S>>;
  fn linear<A, S>(self, out_chan: usize, weights: Param<Array2d<Self::T>>, bias: Option<Param<Array1d<Self::T>>>) -> Rc<VectorLinearOp<Self::T, A, S>>;

  fn softmax_nll(self) -> Rc<VectorSoftmaxNLLLoss>;
}

pub struct VectorAddOp<T, A, S> where T: Copy {
  node:     OperatorNode,
  in_op:    Rc<DiffOperator<S, [T]>>,
  in_:      Rc<ArrayDiffVar<A>>,
  out:      Rc<ArrayDiffVar<A>>,
  bias:     Rc<ParamBlock<Array1d<T>>>,
}

pub struct VectorLinearOp<T, A, S> where T: Copy {
  node:     OperatorNode,
  in_op:    Rc<DiffOperator<S, [T]>>,
  in_:      Rc<ArrayDiffVar<A>>,
  out:      Rc<ArrayDiffVar<A>>,
  out_chan: usize,
  weights:  Rc<ParamBlock<Array2d<T>>>,
  bias:     Option<Rc<ParamBlock<Array1d<T>>>>,
}

pub struct VectorSoftmaxNLLLoss {
}

pub fn concat_join<S, IoBuf>(ops: Vec<(usize, Rc<DiffOperator<S, IoBuf>>)>) -> TensorConcatJoinOp {
  unimplemented!();
}

pub trait TensorDiffExt {
  type T: Copy;

  fn conv2d(self, cfg: TensorConv<(usize, usize)>, weights: Param<Array4d<Self::T>>, bias: Option<Param<Array1d<Self::T>>>) -> TensorConv2dOp<Self::T>;
  //fn scale2d(self, kernel: Param<TensorConv<(usize, usize)>, Array4d<Self::T>>) -> TensorScale2dOp;
}

pub struct TensorConcatJoinOp {
}

pub struct TensorConv<Idx> {
  pub kernel_dim:   Idx,
  pub stride_dim:   Idx,
  pub pad_dim:      Idx,
  pub out_chan:     usize,
  //pub bias:         bool,
}

pub struct TensorConv2dOp<T> where T: Copy {
  cfg:      TensorConv<(usize, usize)>,
  weights:  Rc<ParamBlock<Array4d<T>>>,
  bias:     Option<Rc<ParamBlock<Array1d<T>>>>,
}
