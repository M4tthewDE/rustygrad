fn main() {
    // Path to the CUDA runtime library
    let cuda_lib_dir = "/opt/cuda/targets/x86_64-linux/lib/";

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=cudart");
}
