extern crate image;
extern crate ocl;

use ocl::enums::{
    AddressingMode, FilterMode, ImageChannelDataType, ImageChannelOrder, MemObjectType,
};
use ocl::{Context, Device, Image, Kernel, Program, Queue, Sampler};
use std::path::Path;

const SAVE_IMAGES_TO_DISK: bool = true;
static BEFORE_IMAGE_FILE_NAME: &'static str = "before_example_image.png";
static AFTER_IMAGE_FILE_NAME: &'static str = "after_example_image.png";

static KERNEL_SRC: &'static str = include_str!("./gpu.cl");

const SIZE_POW: u32 = 12;

/// Generates a diagonal reddish stripe and a grey background.
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
fn main() {
    println!("Running 'examples/image.rs::main()'...");
    let mut img = generate_image();

    if SAVE_IMAGES_TO_DISK {
        img.save(&Path::new(BEFORE_IMAGE_FILE_NAME)).unwrap();
    }

    let context = Context::builder()
        .devices(Device::specifier().first())
        .build()
        .unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(KERNEL_SRC)
        .devices(device)
        .build(&context)
        .unwrap();

    let sup_img_formats = Image::<u8>::supported_formats(
        &context,
        ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d,
    )
    .unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());
    // println!("Image Formats: {:#?}.", sup_img_formats);

    let dims = img.dimensions();
    println!("dims: {:?}", dims);

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
        .build()
        .unwrap();

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
        .build()
        .unwrap();

    // Not sure why you'd bother creating a sampler on the host but here's how:
    let sampler = Sampler::new(&context, true, AddressingMode::None, FilterMode::Nearest).unwrap();

    let max: u32 = 100;
    for i in 0..max {
        let kernel = Kernel::builder()
            .program(&program)
            .name("my_fct")
            .queue(queue.clone())
            .global_work_size(&dims)
            .arg_sampler(&sampler)
            .arg(&i)
            .arg(&max)
            .arg(&src_image)
            .arg(&dst_image)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        dst_image.read(&mut img).enq().unwrap();

        let p1 = img.get_pixel(dims.0 / 4, dims.1 / 4).0;
        let p2 = img.get_pixel(dims.0 / 2, dims.1 / 2).0;
        let p3 = img.get_pixel(dims.0 / 4 * 3, dims.1 / 4 * 3).0;

        assert_eq!(p1[0], 64);
        assert_eq!(p1[1], 64);

        assert_eq!(p2[0], 128);
        assert_eq!(p2[1], 128);

        assert_eq!(p3[0], 191);
        assert_eq!(p3[1], 191);
    }

    // let pixels_nice = img.chunks_exact(4).collect::<Vec<&[u8]>>();
    // for x in 0..dims.0 {
    //     print!("{x}:\t");
    //     for y in 0..dims.1 {
    //         print!("{:?}  ", pixels_nice[(x * dims.0 + y) as usize]);
    //     }
    //     println!("");
    // }
    // img.save(&Path::new(AFTER_IMAGE_FILE_NAME)).unwrap();
}
