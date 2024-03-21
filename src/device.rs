#[derive(Clone, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Default for Device {
    fn default() -> Self {
        if std::env::var("CUDA").is_ok() {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}
