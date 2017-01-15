#![allow(dead_code)]

#[macro_use]
extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate image;
extern crate cgmath;

use cgmath as m;
use cgmath::{Zero, Rotation};

use gfx::traits::Factory;
use gfx::traits::FactoryExt;
use gfx::Device;
use glutin::VirtualKeyCode;

use std::io::Cursor;
use std::io::Write;
use std::error::Error;
use std::rc::Rc;
use std::collections::HashSet;

mod object;


gfx_defines!{
    vertex Vertex {
        pos:   [f32; 3] = "vertex_Pos",
        tex:   [f32; 2] = "vertex_TexPos",
    }

    pipeline pipe {
        // projection
        projection: gfx::Global<[[f32; 4]; 4]> = "projection",
        // globals
        blend: gfx::Global<f32> = "blending",
        // textures
        box_texture: gfx::TextureSampler<[f32; 4]> = "boxTexture",
        face_texture: gfx::TextureSampler<[f32; 4]> = "faceTexture",
        // in/output
        vbuf: gfx::VertexBuffer<Vertex> = (),
        out_color: gfx::RenderTarget<gfx::format::Rgba8> = "Target0",
        out_depth: gfx::DepthTarget<gfx::format::DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}


const VERTEX_SHADER: &'static [u8] = b"
#version 330 core

in vec3 vertex_Pos;
in vec2 vertex_TexPos;

uniform mat4 projection;

out vec2 TexPos;
out vec4 pos;

void main() {
    TexPos = vertex_TexPos;

    pos = projection * vec4(vertex_Pos.xyz, 1.0);
    gl_Position = pos;
}
";

const PIXEL_SHADER: &'static [u8] = b"
#version 330 core

in vec4 pos;
in vec2 TexPos;

out vec4 Target0;

uniform float blending;
uniform sampler2D boxTexture;
uniform sampler2D faceTexture;

void main() {
    vec4 face = texture(faceTexture, vec2(TexPos.x, 1.0 - TexPos.y));
    Target0 = mix(texture(boxTexture, TexPos).rgba, face.rgba, face.a * blending);
}
";

const CLEAR_COLOR: [f32; 4] = [0.1, 0.2, 0.3, 1.0];

fn main() {
    if let Err(e) = main_loop() {
        writeln!(&mut std::io::stderr(), "An exception occurred: {}", e).unwrap();
    }
}

fn load_texture<R, F, P>(factory: &mut F, path: P)
               -> Result<gfx::handle::ShaderResourceView<R, [f32; 4]>, Box<Error>> where
               R: gfx::Resources, F: Factory<R>, P: AsRef<std::path::Path> {
    // loading resources
    let texture = image::open(path)?.to_rgba();

    // create the actual input information for the GPU
    let (width, height) = texture.dimensions();
    let kind = gfx::texture::Kind::D2(width as _, height as _, gfx::texture::AaMode::Single);
    let (_, view) = factory.create_texture_immutable_u8::<gfx::format::Rgba8>(kind, &[&texture])?;
    Ok(view)
}

fn load_texture_embedded<R, F>(factory: &mut F, data: &[u8], format: image::ImageFormat)
               -> Result<gfx::handle::ShaderResourceView<R, [f32; 4]>, Box<Error>> where
               R: gfx::Resources, F: Factory<R> {
    let texture = image::load(Cursor::new(data), format)?.to_rgba();

    // create the actual input information for the GPU
    let (width, height) = texture.dimensions();
    let kind = gfx::texture::Kind::D2(width as _, height as _, gfx::texture::AaMode::Single);
    let (_, view) = factory.create_texture_immutable_u8::<gfx::format::Rgba8>(kind, &[&texture])?;
    Ok(view)
}

fn main_loop() -> Result<(), Box<Error>>{
    // create a window
    let builder = glutin::WindowBuilder::new()
        .with_title("Triangle example")
        .with_dimensions(1024, 768)
        .with_vsync();


    // initialization actions, getting the opengl context with the approopriate types

    // get the window(framebuffers), device (thing that submits commandbuffers), factory(thing that gets resources into the GPU)
    // renderer output (in the given type) and depth mask output)
    let (window, mut device, mut factory, render_target, depth_stencil) = 
        gfx_window_glutin::init::<gfx::format::Rgba8, gfx::format::DepthStencil>(builder);

    // this cursor is mine
    window.set_cursor_state(glutin::CursorState::Grab)?;

    // create an encoder (thing that serializes actions into a command buffer) from the factory
    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();


    // create pipelines

    // compile shaders
    let vertex_shader = factory.create_shader_vertex(VERTEX_SHADER)?;
    let pixel_shader = factory.create_shader_pixel(PIXEL_SHADER)?;

    // link the shaders together into a program
    let program = factory.create_program(&gfx::ShaderSet::Simple(vertex_shader, pixel_shader))?;

    // simple rasterizes which rasterizes both the front and back face.
    let rasterizer = gfx::state::Rasterizer::new_fill();
    //rasterizer.method = gfx::state::RasterMethod::Line(20);

    // assemble the pipeline. note that the unwrap is because the return type of this function is rather retarded
    let pso = factory.create_pipeline_from_program(&program, gfx::Primitive::TriangleList, rasterizer, pipe::new()).unwrap();
    // frame information



    // create the actual input information for the GPU
    // let box_tex  = load_texture(&mut factory, "images/container.jpg")?;
    // let face_tex = load_texture(&mut factory, "images/awesomeface.png")?;

    let box_tex  = load_texture_embedded(&mut factory, include_bytes!("../images/container.jpg"), image::ImageFormat::JPEG)?;
    let face_tex = load_texture_embedded(&mut factory, include_bytes!("../images/awesomeface.png"), image::ImageFormat::PNG)?;

    // sampler settings
    let sampler = factory.create_sampler(gfx::texture::SamplerInfo::new(gfx::texture::FilterMethod::Scale, gfx::texture::WrapMode::Tile));



    // my own abstractions go here

    // keys
    let mut keys_pressed = HashSet::new();
    let mut first_mouse = true;

    // datastructures that will be rendered
    let box_mesh = Rc::new(object::Mesh::new(TRIANGLES.iter().cloned(), None));
    // world to contain it all
    let mut world = object::World::new();

    // add a camera
    let mut camera_angle = m::Euler::new(m::Rad(0.0), m::Rad(0.0), m::Rad(0.0));
    let (mut width, mut height) = window.get_inner_size_pixels().unwrap();
    world.add_item(
        object::Item::new_animation(
            Box::new(move |pos, _, state| {
                use m::{Rotation, Rotation3};
                const VELOCITY: f32 = 2.0;
                const ANGULAR_VELOCITY: m::Rad<f32> = m::Rad(0.06);
                let dt = state.dt;

                camera_angle.x += -ANGULAR_VELOCITY * state.dt * state.mouse_movement.0 as f32;
                camera_angle.y += ANGULAR_VELOCITY * state.dt * state.mouse_movement.1 as f32;
                if camera_angle.y > m::Deg(90.0).into() {
                    camera_angle.y = m::Deg(90.0).into()
                } else if camera_angle.y < m::Deg(-90.0).into() {
                    camera_angle.y = m::Deg(-90.0).into()
                }

                // figure out the rotation (we can't use euler angles naturally as the coordinate system is bad for them)
                let rotation = m::Basis3::from_angle_y(camera_angle.x) * m::Basis3::from_angle_x(camera_angle.y);
                pos.rot = m::Quaternion::from(rotation);

                // note: front is in the OPPOSITE direction of the camera
                let front = rotation.rotate_vector(m::Vector3::new(0.0, 0.0, 1.0));
                let up    = rotation.rotate_vector(m::Vector3::new(0.0, 1.0, 0.0));
                let right = rotation.rotate_vector(m::Vector3::new(1.0, 0.0, 0.0));

                // update the position
                if state.keys_pressed.contains(&VirtualKeyCode::W) {
                    pos.disp -= front * VELOCITY * dt;
                } else if state.keys_pressed.contains(&VirtualKeyCode::S) {
                    pos.disp += front * VELOCITY * dt;
                }

                if state.keys_pressed.contains(&VirtualKeyCode::A) {
                    pos.disp -= right * VELOCITY * dt;
                } else if state.keys_pressed.contains(&VirtualKeyCode::D) {
                    pos.disp += right * VELOCITY * dt;
                }

                if state.keys_pressed.contains(&VirtualKeyCode::Space) {
                    pos.disp += up * VELOCITY * dt;
                } else if state.keys_pressed.contains(&VirtualKeyCode::LControl) {
                    pos.disp -= up * VELOCITY * dt;
                }
            }),
            {
                let mut world = object::World::new();
                world.add_item(object::Item::new_camera("cam1", m::perspective(
                    m::Deg(90.0), width as f32 / height as f32, 0.1, 100.0
                )));
                world.add_item(
                    object::Item::new_static(box_mesh.clone())
                    .at(0.0, 0.0, -10.0)
                );

                object::Item::new_subdomain(world)
            }
        )
        .at(0.0, 0.0, 3.0)
    );

    // hook some fancy shit up
    for (i, point) in CUBEPOSITIONS.iter().enumerate() {
        world.add_item(
            object::Item::new_animation(
                Box::new(|pos, _, state| {
                    use m::Rotation3;
                    let dt = state.dt;

                    if state.keys_pressed.contains(&VirtualKeyCode::I) {
                        pos.rot = m::Quaternion::from_angle_x(m::Deg(-90.0 * dt)) * pos.rot;
                    } else if state.keys_pressed.contains(&VirtualKeyCode::K) {
                        pos.rot = m::Quaternion::from_angle_x(m::Deg(90.0 * dt)) * pos.rot;
                    }

                    if state.keys_pressed.contains(&VirtualKeyCode::J) {
                        pos.rot = m::Quaternion::from_angle_y(m::Deg(-90.0 * dt)) * pos.rot;
                    } else if state.keys_pressed.contains(&VirtualKeyCode::L) {
                        pos.rot = m::Quaternion::from_angle_y(m::Deg(90.0 * dt)) * pos.rot;
                    }
                }),
                object::Item::new_static(box_mesh.clone())
            )
            .at(point.x, point.y, point.z)
            .rotate([1.0, 0.3, 0.5].into(), m::Deg(20.0 * i as f32))
        );
    }

    // rendering abstraction
    let mut renderer = object::RenderContext::new();

    // create the vertex buffer
    world.render(&mut renderer);
    let (vertex_buffer, _) = renderer.submit(&mut factory);

    // pipeline and global data
    let mut data = pipe::Data {
        projection: m::Matrix4::zero().into(),
        blend: 0.0,
        face_texture: (face_tex, sampler.clone()),
        box_texture: (box_tex, sampler),
        vbuf: vertex_buffer,
        out_color: render_target,
        out_depth: depth_stencil
    };


    // main loop
    'main: loop {
        let mut mouse_movement = (0, 0);

        // check for any events
        for event in window.poll_events() {
            match event {
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::Closed => break 'main,
                glutin::Event::Resized(new_width, new_height) => {
                    width = new_width;
                    height = new_height;
                    gfx_window_glutin::update_views(&window, &mut data.out_color, &mut data.out_depth);
                },
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed, _, Some(keycode)) => {
                    keys_pressed.insert(keycode);
                },
                glutin::Event::KeyboardInput(glutin::ElementState::Released, _, Some(keycode)) => {
                    keys_pressed.remove(&keycode);
                },
                glutin::Event::MouseEntered => {
                    first_mouse = true;
                    mouse_movement = (0, 0);
                },
                glutin::Event::MouseMoved(x, y) => {
                    // println!("set {} {}", x, y);
                    if first_mouse {
                        first_mouse = false;
                    } else {
                        mouse_movement = (x - width as i32 / 2, height as i32 / 2 - y);
                    }
                    window.set_cursor_position(width as i32 / 2, height as i32 / 2).expect("setting cursor failed?");
                }
                _ => () // a => println!("{:?}", a)
            }
        }

        // update
        let mut state = object::UpdateState::new(1.0/60.0, &mut keys_pressed, mouse_movement);
        world.update(&mut state);

        // rebuild the vertices
        renderer.clear();
        world.render(&mut renderer);
        let (vertex_buffer, slice) = renderer.submit(&mut factory);
        data.vbuf = vertex_buffer;

        // camera
        data.projection = {
            let &(ref camera_pos, ref camera) = renderer.cameras.get("cam1").unwrap();
            camera.perspective * m::Matrix4::from(camera_pos.rot.invert()) * m::Matrix4::from_translation(-camera_pos.disp)
        }.into();


        // clear the window
        encoder.clear(&data.out_color, CLEAR_COLOR);
        encoder.clear_depth(&data.out_depth, 1.0);
        // draw to the commandbuffer with the given vertex data, the pipeline and the input/output for the pipeline
        encoder.draw(&slice, &pso, &data);
        // flush the command buffer to the GPU to actually do work
        encoder.flush(&mut device);
        // swap framebuffers
        window.swap_buffers().unwrap();
        // clean up resources
        device.cleanup();
    }

    Ok(())
}

const TRIANGLES: &'static [Vertex] = &[
    Vertex {pos: [-0.5, -0.5, -0.5], tex: [0.0, 0.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], tex: [1.0, 1.0] },
    Vertex {pos: [-0.5,  0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], tex: [0.0, 0.0] },

    Vertex {pos: [-0.5, -0.5,  0.5], tex: [0.0, 0.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], tex: [1.0, 1.0] },
    Vertex {pos: [-0.5,  0.5,  0.5], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5,  0.5], tex: [0.0, 0.0] },

    Vertex {pos: [-0.5,  0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [-0.5,  0.5, -0.5], tex: [1.0, 1.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5,  0.5], tex: [0.0, 0.0] },
    Vertex {pos: [-0.5,  0.5,  0.5], tex: [1.0, 0.0] },

    Vertex {pos: [ 0.5,  0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], tex: [0.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], tex: [1.0, 0.0] },

    Vertex {pos: [-0.5, -0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [-0.5, -0.5,  0.5], tex: [0.0, 0.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], tex: [0.0, 1.0] },

    Vertex {pos: [-0.5,  0.5, -0.5], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], tex: [1.0, 0.0] },
    Vertex {pos: [-0.5,  0.5,  0.5], tex: [0.0, 0.0] },
    Vertex {pos: [-0.5,  0.5, -0.5], tex: [0.0, 1.0] },
];

const CUBEPOSITIONS: &'static [m::Point3<f32>] = &[
    m::Point3 {x:  0.0, y:  0.0, z:  0.0 },
    m::Point3 {x:  2.0, y:  5.0, z: -15.0 },
    m::Point3 {x: -1.5, y: -2.2, z: -2.5 },
    m::Point3 {x: -3.8, y: -2.0, z: -12.3 },
    m::Point3 {x:  2.4, y: -0.4, z: -3.5 },
    m::Point3 {x: -1.7, y:  3.0, z: -7.5 },
    m::Point3 {x:  1.3, y: -2.0, z: -2.5 },
    m::Point3 {x:  1.5, y:  2.0, z: -2.5 },
    m::Point3 {x:  1.5, y:  0.2, z: -1.5 },
    m::Point3 {x: -1.3, y:  1.0, z: -1.5 }
];