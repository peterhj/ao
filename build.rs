extern crate gcc;
extern crate walkdir;

use walkdir::{WalkDir};

use std::env;
use std::path::{PathBuf};

fn main() {
  println!("cargo:rerun-if-changed=build.rs");

  let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  let out_dir = env::var("OUT_DIR").unwrap();
  let cc = env::var("CC").unwrap_or("gcc".to_owned());

  let mut kernels_src_dir = PathBuf::from(manifest_dir.clone());
  kernels_src_dir.push("kernels");
  for entry in WalkDir::new(kernels_src_dir.to_str().unwrap()) {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }

  /*let mut omp_kernels_src_dir = PathBuf::from(manifest_dir.clone());
  omp_kernels_src_dir.push("omp_kernels");
  for entry in WalkDir::new(omp_kernels_src_dir.to_str().unwrap()) {
    let entry = entry.unwrap();
    println!("cargo:rerun-if-changed={}", entry.path().display());
  }*/

  gcc::Config::new()
    .compiler(&cc)
    .opt_level(3)
    .pic(true)
    .flag("-std=gnu99")
    .flag("-march=native")
    .flag("-fno-strict-aliasing")
    .flag("-Ikernels")
    .file("kernels/special_map.c")
    .compile("libarraydiff_kernels.a");

  /*let openmp_cc = if cfg!(not(feature = "iomp")) {
    env::var("CC").unwrap_or("gcc".to_owned())
  } else {
    "icc".to_owned()
  };
  let mut openmp_gcc = gcc::Config::new();
  openmp_gcc
    .compiler(&openmp_cc);
  if cfg!(not(feature = "iomp")) {
    openmp_gcc
      .opt_level(3)
      .pic(true)
      .flag("-std=gnu99")
      .flag("-march=native")
      .flag("-fno-strict-aliasing")
      .flag("-fopenmp");
  } else {
    openmp_gcc
      .opt_level(2)
      .pic(true)
      .flag("-std=c99")
      .flag("-qopenmp")
      .flag("-qno-offload")
      .flag("-xMIC-AVX512");
    /*if cfg!(feature = "knl") {
      openmp_gcc
        .flag("-qno-offload")
        .flag("-xMIC-AVX512");
    }*/
  }
  openmp_gcc
    .flag("-Ikernels")
    .flag("-DNEURALOPS_OMP")
    .file("omp_kernels/activate.c")
    .file("omp_kernels/conv.c")
    .file("omp_kernels/image.c")
    .file("omp_kernels/interpolate.c")
    .file("omp_kernels/pool.c")
    .compile("libneuralops_omp_kernels.a");*/

  println!("cargo:rustc-link-search=native={}", out_dir);
}
