#![allow(dead_code)]

#[macro_use]
extern crate gfx;
extern crate gfx_window_glutin;
extern crate glutin;
extern crate image;
extern crate cgmath;
extern crate env_logger;

use cgmath as m;
use cgmath::{Rotation};

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
        pos:    [f32; 3] = "vertex_Pos",
        normal: [f32; 3] = "vertex_normal",
        tex:    [f32; 2] = "vertex_TexPos",
    }

    constant DirectionalLight {
        direction: [f32; 4] = "direction",
        ambient:   [f32; 4] = "ambient",
        diffuse:   [f32; 4] = "diffuse",
        specular:  [f32; 4] = "specular",
    }

    constant PointLight {
        position: [f32; 4] = "pos",
        ambient:  [f32; 4] = "ambient",
        diffuse:  [f32; 4] = "diffuse",
        specular: [f32; 4] = "specular",
        falloff:  [f32; 4] = "falloff",
    }

    constant SpotLight {
        position: [f32; 4] = "pos",
        direction:[f32; 4] = "direction",
        ambient:  [f32; 4] = "ambient",
        diffuse:  [f32; 4] = "diffuse",
        specular: [f32; 4] = "specular",
        falloff:  [f32; 4] = "falloff",
        cone:     [f32; 4] = "cone",
    }

    pipeline shaded_pipeline {
        // projection
        projection: gfx::Global<[[f32; 4]; 4]> = "projection",
        // globals
        material_diffuse:   gfx::TextureSampler<[f32; 4]> = "material.diffuse",
        material_specular:  gfx::TextureSampler<[f32; 4]> = "material.specular",
        material_shininess: gfx::Global<f32>              = "material.shininess",

        // lighting stuff
        light_directional_count: gfx::Global<i32>   = "directional_count",
        light_point_count:       gfx::Global<i32>   = "point_count",
        light_spot_count:        gfx::Global<i32>   = "spot_count",

        light_directionals: gfx::ConstantBuffer<DirectionalLight> = "Directionals",
        light_points:       gfx::ConstantBuffer<PointLight>       = "Points",
        light_spots:        gfx::ConstantBuffer<SpotLight>        = "Spots",

        // camera pos
        viewpos: gfx::Global<[f32; 3]> = "viewpos",
        // textures
        // in/output
        vbuf: gfx::VertexBuffer<Vertex> = (),
        out_color: gfx::RenderTarget<gfx::format::Rgba8> = "Target0",
        out_depth: gfx::DepthTarget<gfx::format::DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }

    pipeline unshaded_pipeline {
        // projection
        projection: gfx::Global<[[f32; 4]; 4]> = "projection",
        // globals
        // textures
        // in/output
        vbuf: gfx::VertexBuffer<Vertex> = (),
        out_color: gfx::RenderTarget<gfx::format::Rgba8> = "Target0",
        out_depth: gfx::DepthTarget<gfx::format::DepthStencil> = gfx::preset::depth::LESS_EQUAL_WRITE,
    }
}


const SHADED_VERTEX_SHADER: &'static [u8] = b"
    #version 330 core

    in vec3 vertex_Pos;
    in vec3 vertex_normal;
    in vec2 vertex_TexPos;

    uniform mat4 projection;

    out vec3 pos;
    out vec3 normal;
    out vec2 texpos;

    void main() {
        texpos = vertex_TexPos;
        pos = vertex_Pos;
        normal = vertex_normal;
        gl_Position = projection * vec4(pos, 1.0);
    }
";

const SHADED_PIXEL_SHADER: &'static [u8] = b"
    #version 330 core

    const int MAX_NUM_LIGHTS = 16;

    in vec3 pos;
    in vec3 normal;
    in vec2 texpos;
    uniform vec3 viewpos;

    struct Material {
        sampler2D diffuse;
        sampler2D specular;
        float shininess;
    };

    uniform Material material;

    struct DirectionalLight {
        vec4 direction;
        vec4 ambient;
        vec4 diffuse;
        vec4 specular;
    };

    struct PointLight {
        vec4 pos;
        vec4 ambient;
        vec4 diffuse;
        vec4 specular;
        vec4 falloff;
    };

    struct SpotLight {
        vec4 pos;
        vec4 direction;
        vec4 ambient;
        vec4 diffuse;
        vec4 specular;
        vec4 falloff;
        vec4 cone;
    };

    uniform int directional_count;
    uniform int point_count;
    uniform int spot_count;

    layout(std140)
    uniform Directionals {
        DirectionalLight directionals[MAX_NUM_LIGHTS];
    };

    uniform Points {
        PointLight       points[MAX_NUM_LIGHTS];
    };

    uniform Spots {
        SpotLight        spots[MAX_NUM_LIGHTS];
    };

    out vec4 Target0;

    vec3 CalcDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewdir) {
        // vec3 tex_diffuse = texture(material.diffuse, texpos).rgb;
        // vec3 tex_specular = texture(material.specular, texpos).rgb;
        // vec3 lightdir = normalize(-light.direction.xyz);
        // vec3 reflectdir = reflect(-lightdir, normal);

        // vec3 ambient = tex_diffuse * light.ambient;
        // vec3 diffuse = tex_diffuse * max(0.0, dot(normal, lightdir)) * light.diffuse.rgb;
        // vec3 specular = tex_specular * pow(max(0.0, dot(viewdir, reflectdir)), material.shininess) * light.specular.rgb;

        return vec3(1.0); // ambient + diffuse + specular;
    }

    vec3 CalcPointLight(PointLight light, vec3 normal, vec3 pos, vec3 viewdir) {
        vec3 tex_diffuse  = texture(material.diffuse, texpos).rgb;
        vec3 tex_specular = texture(material.specular, texpos).rgb;
        vec3 lightdir     = normalize(light.pos.xyz - pos);
        vec3 reflectdir   = reflect(-lightdir, normal);

        vec3 ambient = tex_diffuse * light.ambient.rgb;
        vec3 diffuse = tex_diffuse * max(0.0, dot(normal, lightdir)) * light.diffuse.rgb;
        vec3 specular = tex_specular * pow(max(0.0, dot(viewdir, reflectdir)), material.shininess) * light.specular.rgb;

        float distance = length(light.pos.xyz - pos);
        float attenuation = 1.0 / (light.falloff.x + light.falloff.y * distance + light.falloff.z * (distance * distance));

        return (ambient + diffuse + specular) * attenuation;
    }

    vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 pos, vec3 viewdir) {
        vec3 tex_diffuse  = texture(material.diffuse, texpos).rgb;
        vec3 tex_specular = texture(material.specular, texpos).rgb;
        vec3 lightdir     = normalize(light.pos.xyz - pos);
        vec3 reflectdir   = reflect(-lightdir, normal);

        vec3 ambient = tex_diffuse * light.ambient.xyz;

        float theta = dot(lightdir, normalize(-light.direction.xyz));
        float distance = length(light.pos.xyz - pos);
        float attenuation = 1.0 / (light.falloff.x + light.falloff.y * distance + light.falloff.z * (distance * distance));

        if (theta < light.cone.y) {
            return ambient * attenuation;

        } else {
            float epsilon = light.cone.x - light.cone.y;
            float intensity = clamp((theta - light.cone.y) / epsilon, 0.0, 1.0);

            vec3 diffuse = tex_diffuse * max(0.0, dot(normal, lightdir)) * light.diffuse.rgb;
            vec3 specular = tex_specular * pow(max(0.0, dot(viewdir, reflectdir)), material.shininess) * light.specular.rgb;

            vec3 spot = (diffuse + specular) * intensity + ambient; 

            return spot * attenuation;

        }
    }

    void main() {
        vec3 viewdir = normalize(viewpos - pos);
        vec3 norm = normalize(normal);

        vec3 result = vec3(0.0);
        for (int i = 0; i < directional_count; i++) {
            result += CalcDirectionalLight(directionals[i], norm, viewdir);
        }
        for (int i = 0; i < point_count; i++) {
            result += CalcPointLight(points[i], norm, pos, viewdir);
        }
        for (int i = 0; i < spot_count; i++) {
            result += CalcSpotLight(spots[i], norm, pos, viewdir);
        }

        Target0 = pow(vec4(result, 1.0), vec4(2.2));
    }
";

const UNSHADED_VERTEX_SHADER: &'static [u8] = b"
    #version 330 core

    in vec3 vertex_Pos;
    in vec2 vertex_TexPos;

    uniform mat4 projection;

    out vec4 pos;

    void main() {
        pos = projection * vec4(vertex_Pos.xyz, 1.0);
        gl_Position = pos;
    }
";

const UNSHADED_PIXEL_SHADER: &'static [u8] = b"
    #version 330 core

    out vec4 Target0;

    void main() {
        Target0 = vec4(1.0, 0.2, 0.2, 1.0);
    }
";


const CLEAR_COLOR: [f32; 4] = [0.01, 0.01, 0.01, 1.0];

fn main() {
    env_logger::init().unwrap();
    if let Err(e) = main_loop() {
        println!("{:#?}", e);
        writeln!(&mut std::io::stderr(), "An exception occurred: {}", e).unwrap();
    }
}

fn load_texture<R, F, P>(factory: &mut F, path: P) -> 
               Result<gfx::handle::ShaderResourceView<R, [f32; 4]>, Box<Error>> where
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
    let (window, mut device, mut factory, mut render_target, mut depth_stencil) = 
        gfx_window_glutin::init::<gfx::format::Rgba8, gfx::format::DepthStencil>(builder);

    // this cursor is mine
    window.set_cursor_state(glutin::CursorState::Grab)?;

    // create an encoder (thing that serializes actions into a command buffer) from the factory
    let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();


    // create pipelines

    // compile shaders
    let shaded_vertex_shader = factory.create_shader_vertex(SHADED_VERTEX_SHADER)?;
    let shaded_pixel_shader = factory.create_shader_pixel(SHADED_PIXEL_SHADER)?;
    let unshaded_vertex_shader = factory.create_shader_vertex(UNSHADED_VERTEX_SHADER)?;
    let unshaded_pixel_shader = factory.create_shader_pixel(UNSHADED_PIXEL_SHADER)?;

    // simple rasterizes which rasterizes both the front and back face.
    let rasterizer = gfx::state::Rasterizer::new_fill();

    // first pipeline
    let program = factory.create_program(&gfx::ShaderSet::Simple(shaded_vertex_shader, shaded_pixel_shader))?;
    let shaded_pso = factory.create_pipeline_from_program(&program, gfx::Primitive::TriangleList, rasterizer.clone(), shaded_pipeline::new()).map_err(gfx::PipelineStateError::<String>::from)?;

    // second pipeline
    let program = factory.create_program(&gfx::ShaderSet::Simple(unshaded_vertex_shader, unshaded_pixel_shader))?;
    let unshaded_pso = factory.create_pipeline_from_program(&program, gfx::Primitive::TriangleList, rasterizer, unshaded_pipeline::new()).map_err(gfx::PipelineStateError::<String>::from)?;

    // sampler settings
    let diffuse = load_texture(&mut factory, "images/container2.png")?;
    let specular = load_texture(&mut factory, "images/container2_specular.png")?;
    let sampler = factory.create_sampler(gfx::texture::SamplerInfo::new(gfx::texture::FilterMethod::Bilinear, gfx::texture::WrapMode::Tile));

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
                // world.add_item(
                //     object::Item::new_static(box_mesh.clone())
                //     .at(0.0, 0.0, -10.0)
                // );
                world.add_item(
                    object::Item::new_spot_light(
                        object::LightIntensity {
                            ambient: [0.1, 0.1, 0.1].into(),
                            diffuse: [1.0, 1.0, 1.0].into(),
                            specular: [2.0, 2.0, 2.0].into()
                        },
                        object::LightFalloff {
                            constant: 1.0,
                            linear: 0.09,
                            quadratic: 0.032
                        },
                        m::Deg(12.5).into(),
                        m::Deg(17.5).into()
                    )
                    .rotate([0.0, 1.0, 0.0].into(), m::Deg(180.0))
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

    // add light
    let lights = [m::Vector3::new(0.7, 0.2, 2.0), 
                  m::Vector3::new(2.3, -3.3, -4.0),
                  m::Vector3::new(-4.0, 2.0, -12.0),
                  m::Vector3::new(0.0, 0.0, -3.0)];
    for pos in lights.iter().cloned() {
        world.add_item(object::Item::new_static_unshaded(box_mesh.clone())
                                    .at(pos.x, pos.y, pos.z).scale(0.2));
        world.add_item(object::Item::new_point_light(
            object::LightIntensity {
                ambient: [0.1, 0.1, 0.1].into(),
                diffuse: [0.5, 0.1, 0.1].into(),
                specular: [1.0, 0.2, 0.2].into()
            },
            object::LightFalloff {
                constant: 1.0,
                linear: 0.09,
                quadratic: 0.032
            }
        ).at(pos.x, pos.y, pos.z));
    }

    // rendering abstraction
    let mut renderer = object::RenderContext::new();

    let directional_light_buffer = factory.create_constant_buffer::<DirectionalLight>(256);
    let point_light_buffer       = factory.create_constant_buffer::<PointLight>(256);
    let spot_light_buffer        = factory.create_constant_buffer::<SpotLight>(256);


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
                    gfx_window_glutin::update_views(&window, &mut render_target, &mut depth_stencil);
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

        // camera
        let (ref camera_pos, ref camera) = *renderer.cameras.get("cam1").unwrap();
        let projection = (camera.perspective * m::Matrix4::from(camera_pos.rot.invert()) * m::Matrix4::from_translation(-camera_pos.disp)).into();

        // queue lighting data updates
        encoder.update_buffer(&directional_light_buffer, &renderer.directional_lights, 0)?;
        encoder.update_buffer(&point_light_buffer, &renderer.point_lights, 0)?;
        encoder.update_buffer(&spot_light_buffer,  &renderer.spot_lights, 0)?;

        // clear the window
        encoder.clear(&render_target, CLEAR_COLOR);
        encoder.clear_depth(&depth_stencil, 1.0);
        // draw to the commandbuffer with the given vertex data, the pipeline and the input/output for the pipeline
        let (shaded_vertices, slice)   = factory.create_vertex_buffer_with_slice(&renderer.shaded_geometry.0, &renderer.shaded_geometry.1[..]);
        encoder.draw(&slice, &shaded_pso, &shaded_pipeline::Data {
            projection: projection,
            material_diffuse:  (diffuse.clone(), sampler.clone()),
            material_specular: (specular.clone(), sampler.clone()),
            material_shininess: 32.0,
            light_directional_count: renderer.directional_lights.len() as i32,
            light_point_count:       renderer.point_lights.len() as i32,
            light_spot_count:        renderer.spot_lights.len() as i32,
            light_directionals:      directional_light_buffer.clone(),
            light_points:            point_light_buffer.clone(),
            light_spots:             spot_light_buffer.clone(),
            viewpos: camera_pos.disp.into(),
            vbuf: shaded_vertices,
            out_color: render_target.clone(),
            out_depth: depth_stencil.clone(),
        });
        // second draw
        let (unshaded_vertices, slice) = factory.create_vertex_buffer_with_slice(&renderer.unshaded_geometry.0, &renderer.unshaded_geometry.1[..]);
        encoder.draw(&slice, &unshaded_pso, &unshaded_pipeline::Data {
            projection: projection,
            vbuf: unshaded_vertices,
            out_color: render_target.clone(),
            out_depth: depth_stencil.clone(),
        });
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
    Vertex {pos: [-0.5, -0.5, -0.5], normal: [ 0.0,  0.0, -1.0], tex: [0.0, 0.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], normal: [ 0.0,  0.0, -1.0], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], normal: [ 0.0,  0.0, -1.0], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], normal: [ 0.0,  0.0, -1.0], tex: [1.0, 1.0] },
    Vertex {pos: [-0.5,  0.5, -0.5], normal: [ 0.0,  0.0, -1.0], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], normal: [ 0.0,  0.0, -1.0], tex: [0.0, 0.0] },

    Vertex {pos: [-0.5, -0.5,  0.5], normal: [ 0.0,  0.0,  1.0], tex: [0.0, 0.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], normal: [ 0.0,  0.0,  1.0], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], normal: [ 0.0,  0.0,  1.0], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], normal: [ 0.0,  0.0,  1.0], tex: [1.0, 1.0] },
    Vertex {pos: [-0.5,  0.5,  0.5], normal: [ 0.0,  0.0,  1.0], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5,  0.5], normal: [ 0.0,  0.0,  1.0], tex: [0.0, 0.0] },

    Vertex {pos: [-0.5,  0.5,  0.5], normal: [-1.0,  0.0,  0.0], tex: [1.0, 0.0] },
    Vertex {pos: [-0.5,  0.5, -0.5], normal: [-1.0,  0.0,  0.0], tex: [1.0, 1.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], normal: [-1.0,  0.0,  0.0], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], normal: [-1.0,  0.0,  0.0], tex: [0.0, 1.0] },
    Vertex {pos: [-0.5, -0.5,  0.5], normal: [-1.0,  0.0,  0.0], tex: [0.0, 0.0] },
    Vertex {pos: [-0.5,  0.5,  0.5], normal: [-1.0,  0.0,  0.0], tex: [1.0, 0.0] },

    Vertex {pos: [ 0.5,  0.5,  0.5], normal: [ 1.0,  0.0,  0.0], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], normal: [ 1.0,  0.0,  0.0], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], normal: [ 1.0,  0.0,  0.0], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], normal: [ 1.0,  0.0,  0.0], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], normal: [ 1.0,  0.0,  0.0], tex: [0.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], normal: [ 1.0,  0.0,  0.0], tex: [1.0, 0.0] },

    Vertex {pos: [-0.5, -0.5, -0.5], normal: [ 0.0, -1.0,  0.0], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5, -0.5], normal: [ 0.0, -1.0,  0.0], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], normal: [ 0.0, -1.0,  0.0], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5, -0.5,  0.5], normal: [ 0.0, -1.0,  0.0], tex: [1.0, 0.0] },
    Vertex {pos: [-0.5, -0.5,  0.5], normal: [ 0.0, -1.0,  0.0], tex: [0.0, 0.0] },
    Vertex {pos: [-0.5, -0.5, -0.5], normal: [ 0.0, -1.0,  0.0], tex: [0.0, 1.0] },

    Vertex {pos: [-0.5,  0.5, -0.5], normal: [ 0.0,  1.0,  0.0], tex: [0.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5, -0.5], normal: [ 0.0,  1.0,  0.0], tex: [1.0, 1.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], normal: [ 0.0,  1.0,  0.0], tex: [1.0, 0.0] },
    Vertex {pos: [ 0.5,  0.5,  0.5], normal: [ 0.0,  1.0,  0.0], tex: [1.0, 0.0] },
    Vertex {pos: [-0.5,  0.5,  0.5], normal: [ 0.0,  1.0,  0.0], tex: [0.0, 0.0] },
    Vertex {pos: [-0.5,  0.5, -0.5], normal: [ 0.0,  1.0,  0.0], tex: [0.0, 1.0] },
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