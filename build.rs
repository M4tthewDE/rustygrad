use std::env;

extern crate cc;

fn main() {
    // don't redeclare -ccbin
    env::set_var("NVCC_PREPEND_FLAGS", "");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode=arch=compute_75,code=sm_75")
        .file("resources/backend.cu")
        .compile("libbackend.a");

    // Path to the CUDA runtime library
    let cuda_lib_dir = "/usr/local/cuda/lib64";

    println!("cargo:rerun-if-changed=resources/backend.cu");
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
}
