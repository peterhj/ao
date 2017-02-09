use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

pub struct Array1dVar {
}

pub struct Array2dVar {
}

pub struct MatVecProdOp<MatTy, VecTy> {
  a:    Rc<MatTy>,
  x:    Rc<VecTy>,
  y:    Weak<VecTy>,
}

impl Into<Rc<Array1dVar>> for MatVecProdOp<Array2dVar, Array1dVar> {
  fn into(self) -> Rc<Array1dVar> {
    Weak::upgrade(&self.y).unwrap()
  }
}
