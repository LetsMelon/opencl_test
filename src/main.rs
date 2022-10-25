use anyhow::Result;
use ocl::enums::*;
use ocl::{Context, Device, Image, Kernel, Program, Queue, Sampler};
use std::time::Instant;

mod info;

use info::*;

static KERNEL_SRC: &'static str = include_str!("./gpu.cl");

const SIZE_POW: u32 = 10;

#[cfg(feature = "10")]
const ITERATIONS: u32 = 10;
#[cfg(feature = "100")]
const ITERATIONS: u32 = 100;
#[cfg(feature = "1000")]
const ITERATIONS: u32 = 1000;
#[cfg(feature = "10000")]
const ITERATIONS: u32 = 10000;
#[cfg(feature = "100000")]
const ITERATIONS: u32 = 100000;
#[cfg(feature = "1000000")]
const ITERATIONS: u32 = 1000000;

fn generate_image() -> image::ImageBuffer<image::Rgba<u8>, Vec<u8>> {
    let size = 2_u32.pow(SIZE_POW);
    let diff = (0.046875 * size as f64) as u32;
    let img = image::ImageBuffer::from_fn(size, size, |x, y| {
        let near_midline = (x + y < (size + diff)) && (x + y > (size - diff));

        if near_midline {
            image::Rgba([196, 50, 50, 255u8])
        } else {
            image::Rgba([50, 50, 50, 255u8])
        }
    });

    img
}

/// Generates and image then sends it through a kernel and optionally saves.
fn main() -> Result<()> {
    let mut img = generate_image();

    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()?;
    print_context_info(&context);

    let device = context.devices()[0];
    print_device_info(&device)?;

    let queue = Queue::new(&context, device, None)?;

    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .build(&context)?;
    // print_program_info(&program);

    /*     let sup_img_formats = Image::<u8>::supported_formats(
        &context,
        ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d,
    )?; */

    let dims = img.dimensions();

    let src_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build()?;

    let dst_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_WRITE_ONLY
                | ocl::flags::MEM_HOST_READ_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build()?;

    // Not sure why you'd bother creating a sampler on the host but here's how:
    let sampler = Sampler::new(&context, true, AddressingMode::None, FilterMode::Nearest)?;

    let mut durations = Vec::new();

    for i in 0..ITERATIONS {
        let now = Instant::now();

        let kernel = Kernel::builder()
            .program(&program)
            .name("my_fct")
            .queue(queue.clone())
            .global_work_size(&dims)
            .arg_sampler(&sampler)
            .arg(&i)
            .arg(&ITERATIONS)
            .arg(&src_image)
            .arg(&dst_image)
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        dst_image.read(&mut img).enq()?;

        let p1 = img.get_pixel(dims.0 / 4, dims.1 / 4).0;
        let p2 = img.get_pixel(dims.0 / 2, dims.1 / 2).0;
        let p3 = img.get_pixel(dims.0 / 4 * 3, dims.1 / 4 * 3).0;

        assert_eq!(p1[0], 64);
        assert_eq!(p1[1], 64);

        assert_eq!(p2[0], 128);
        assert_eq!(p2[1], 128);

        assert_eq!(p3[0], 191);
        assert_eq!(p3[1], 191);

        let delta = now.elapsed().as_millis() as u64;

        durations.push(delta);
    }

    assert_eq!(durations.len() as u32, ITERATIONS);

    println!("=========================");

    let summed = durations.iter().sum::<u64>();
    let avg = (summed as f64) / (ITERATIONS as f64);
    let bytes = dims.0 * dims.1 * 4;

    println!("dims: {:?}", dims);
    println!("iterations: {}", ITERATIONS);
    println!("sum: {} ms", summed);
    println!("avg: {:.2} ms", avg);
    println!("Iters/s: {:.0}", 1000.0 / avg);
    println!("Bytes: {}", bytes);
    println!(
        "GB/s: {:.2}",
        (1000.0 / avg * (bytes as f64)) / 1024_f64.powi(3)
    );

    Ok(())
}
