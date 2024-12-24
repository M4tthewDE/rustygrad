extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode=arch=compute_75,code=sm_75")
        .flag("-ccbin=clang")
        .file("resources/backend.cu")
        .compile("libbackend.a");

    // Path to the CUDA runtime library
    let cuda_lib_dir = "/opt/cuda/lib64";

    println!("cargo:rerun-if-changed=resources/backend.cu");
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
}
