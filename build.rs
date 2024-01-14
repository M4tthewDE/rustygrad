extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .file("resources/add.cu")
        .compile("libadd.a");

    // Path to the CUDA runtime library
    let cuda_lib_dir = "/usr/local/cuda/lib64";

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
}
