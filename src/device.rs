#[derive(Clone, Debug)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Default for Device {
    fn default() -> Self {
        if std::env::var("GPU").is_ok() {
            Device::Gpu
        } else {
            Device::Cpu
        }
    }
}
