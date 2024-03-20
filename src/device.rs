use std::sync::Mutex;

use lazy_static::lazy_static;

lazy_static! {
    static ref DEVICE: Mutex<Device> = Mutex::new(Device::from_env());
}

#[derive(Clone)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    fn from_env() -> Device {
        if std::env::var("CUDA").is_ok() {
            Device::Cuda
        } else {
            Device::Cpu
        }
    }
}

pub fn get_device() -> Device {
    DEVICE.lock().unwrap().clone()
}

pub fn set_device(device: Device) {
    *DEVICE.lock().unwrap() = device;
}
