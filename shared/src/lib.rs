#![allow(clippy::all)]

use compute_engine::{BaseEngine, ComputeEngine};
use image::{ImageBuffer, Rgba};
use math::Vertex;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CopyImageToBufferInfo, RenderPassBeginInfo, SubpassContents,
    },
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    pipeline::{
        graphics::{
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, Subpass},
};

mod shader_vertex {
    vulkano_shaders::shader! {ty: "vertex", path: "src/shader.vert"}
}

mod shader_fragment {
    vulkano_shaders::shader! {ty: "fragment", path: "src/shader.frag"}
}

#[cfg(test)]
mod tests;

pub fn entrypoint() {
    // Prepare Engine
    let compute_engine = ComputeEngine::new();

    // Print information
    ComputeEngine::print_api_information(compute_engine.get_instance(), log::Level::Info);

    // Set vertices for triangle
    let vertex1 = Vertex {
        position: [-0.5, -0.5],
    };
    let vertex2 = Vertex {
        position: [0.0, 0.5],
    };
    let vertex3 = Vertex {
        position: [0.5, -0.25],
    };

    // Create vertex buffer
    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            vertex_buffer: true,
            ..Default::default()
        },
        false,
        vec![vertex1, vertex2, vertex3].into_iter(),
    )
    .unwrap();

    // Create Output buffer
    let output_buffer = CpuAccessibleBuffer::from_iter(
        compute_engine.get_logical_device().get_device(),
        BufferUsage {
            transfer_dst: true,
            ..Default::default()
        },
        false,
        (0..1024 * 1024 * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    // Load Shaders
    let vertex_shader = shader_vertex::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create vertex shader module");
    let fragment_shader = shader_fragment::load(compute_engine.get_logical_device().get_device())
        .expect("failed to create fragment shader module");

    // Define Viewport
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [1024.0, 1024.0],
        depth_range: 0.0..1.0,
    };

    // Create RenderPass (prepare "rendering mode")
    // Defines the format and way of the image to be rendered
    let render_pass = vulkano::single_pass_renderpass!(
        compute_engine.get_logical_device().get_device(),
        attachments: {
            color: {
                load: Clear,    // Tells the GPU to clear the image when entering RenderPass
                store: Store,   // Tells the GPU to store any outputs to our image
                format: Format::R8G8B8A8_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap();

    // Create Image
    let image = StorageImage::new(
        compute_engine.get_logical_device().get_device(),
        ImageDimensions::Dim2d {
            width: 1024,
            height: 1024,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(compute_engine.get_logical_device().get_queue_family_index()),
    )
    .unwrap();

    // Create ImageView
    // Needed as a link between the CPU and the GPU
    let view = ImageView::new_default(image.clone()).unwrap();

    // Create FrameBuffer
    // Used to store images that are rendered.
    // But also handles attachments.
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap();

    // Create GraphicsPipeline
    let pipeline = GraphicsPipeline::start()
        // Defines the layout of our Vertex object
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        // Defines the entry point of our vertex shader
        .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
        // Defines the primitive type (e.g. triangles, quads, etc.)
        // Default is triangles.
        .input_assembly_state(InputAssemblyState::new())
        // Defines the viewport
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        // Defines the entry point of our fragment shader
        .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
        // Defines the render pass
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        // Build it! :)
        .build(compute_engine.get_logical_device().get_device())
        .unwrap();

    // Submit Command Buffer for Computation
    compute_engine.compute(&|compute_engine: &ComputeEngine| {
        let mut builder = AutoCommandBufferBuilder::primary(
            compute_engine.get_logical_device().get_device(),
            compute_engine.get_logical_device().get_queue_family_index(),
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .bind_pipeline_graphics(pipeline.clone())
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(
                3, // Vertex count
                1, // Instance count
                0, // First vertex
                0, // First instance
            )
            .unwrap()
            .end_render_pass()
            .unwrap()
            .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                image.clone(),
                output_buffer.clone(),
            ))
            .unwrap();

        builder.build().unwrap()
    });

    // Save results
    #[cfg(debug_assertions)]
    let start = Instant::now();

    let buffer_content = output_buffer.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("triangle.png").unwrap();

    #[cfg(debug_assertions)]
    let end = Instant::now();

    log::info!("Successfully saved image");

    #[cfg(debug_assertions)]
    log::debug!(
        "Storing image took: {}ms",
        end.duration_since(start).as_millis()
    );
}
