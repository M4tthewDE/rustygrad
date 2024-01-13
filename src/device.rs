use std::sync::Mutex;

use lazy_static::lazy_static;

lazy_static! {
    static ref DEVICE: Mutex<Device> = Mutex::new(Device::Cpu);
}

#[derive(Clone)]
pub enum Device {
    Cpu,
    Cuda,
}

pub fn get_device() -> Device {
    DEVICE.lock().unwrap().clone()
}

pub fn set_device(device: Device) {
    *DEVICE.lock().unwrap() = device;
}
