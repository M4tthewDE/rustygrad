extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_61,code=sm_61")
        .file("resources/add.cu")
        .compile("libadd.a");

    // Path to the CUDA runtime library
    let cuda_lib_dir = "/opt/cuda/targets/x86_64-linux/lib/";

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
}
