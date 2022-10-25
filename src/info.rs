use anyhow::Result;

use ocl::{Context, Device, Program};

pub fn print_device_info(device: &Device) -> Result<()> {
    println!("{}\n", device);
    Ok(())
}

pub fn print_context_info(context: &Context) {
    println!("{}\n", context);
}

#[allow(dead_code)]
pub fn print_program_info(program: &Program) {
    println!("{}\n", program);
}
