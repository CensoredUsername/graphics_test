
use gfx::{Resources, Factory, Slice};
use gfx::traits::FactoryExt;
use gfx::handle::Buffer;
use cgmath as m;
use cgmath::{Rotation3, Transform, InnerSpace};
use glutin::VirtualKeyCode;

use std::collections::{HashSet, HashMap};
use std::rc::Rc;

use ::Vertex;


#[derive(Debug)]
pub struct UpdateState<'a> {
    pub dt: f32,
    pub mouse_movement: (i32, i32),
    pub keys_pressed: &'a HashSet<VirtualKeyCode>
}

impl<'a> UpdateState<'a> {
    pub fn new(dt: f32, keys_pressed: &'a HashSet<VirtualKeyCode>, mouse_movement: (i32, i32)) -> UpdateState<'a> {
        UpdateState {
            dt: dt,
            mouse_movement: mouse_movement,
            keys_pressed: keys_pressed,
        }
    }
}

#[derive(Debug)]
pub struct RenderContext {
    pub geometry: (Vec<Vertex>, Vec<u32>),
    pub transform_stack: Vec<WorldPos>,
    pub cameras: HashMap<&'static str, (WorldPos, Camera)>
}

impl RenderContext {
    pub fn new() -> RenderContext {
        RenderContext {
            geometry: (Vec::new(), Vec::new()),
            transform_stack: Vec::new(),
            cameras: HashMap::new(),
        }
    }

    pub fn with_transform<F: Fn(&mut RenderContext)>(&mut self, pos: &WorldPos, f: F) {
        let new = if let Some(current_transform) = self.transform_stack.last() {
            current_transform.concat(pos)
        } else {
            pos.clone()
        };
        self.transform_stack.push(new);
        f(self);
        self.transform_stack.pop();
    }

    pub fn clear(&mut self) {
        let (ref mut vertices, ref mut indices) = self.geometry;
        vertices.clear();
        indices.clear();
        self.cameras.clear()
    }

    pub fn submit<F, R: Resources>(&mut self, factory: &mut F) -> (Buffer<R, Vertex>, Slice<R>) where F: Factory<R> {
        let (ref vertices, ref indices) = self.geometry;
        factory.create_vertex_buffer_with_slice(&vertices[..], &indices[..])
    }

    fn transform_pos(stack: &Vec<WorldPos>, pos: m::Point3<f32>) -> m::Point3<f32> {
        if let Some(transform) = stack.last() {
            transform.transform_point(pos)
        } else {
            pos
        }
    }

    fn transform_worldpos(&self, pos: &WorldPos) -> WorldPos {
        if let Some(transform) = self.transform_stack.last() {
            transform.concat(pos)
        } else {
            pos.clone()
        }
    }

    pub fn add_mesh(&mut self, mesh: &Mesh<Vertex>) {
        let (ref mut vertices, ref mut indices) = self.geometry;
        let stack = &self.transform_stack;

        let current_offset = vertices.len() as u32;
        if !mesh.indices.is_empty() {
            indices.extend(mesh.indices.iter().cloned().map(|i| i + current_offset));
        } else {
            indices.extend((0..mesh.vertices.len()).map(|i| i as u32 + current_offset));
        }

        // transform vertices
        vertices.extend(
            mesh.vertices.iter().cloned().map(|mut vertex| {
                vertex.pos = Self::transform_pos(stack, vertex.pos.into()).into();
                vertex
            })
        );
    }

    pub fn set_camera(&mut self, camera: Camera) {
        let pos = if let Some(&pos) = self.transform_stack.last() {
            pos
        } else {
            Transform::one()
        };
        self.cameras.insert(camera.id, (pos, camera));
    }
}

// render tree

pub type WorldPos = m::Decomposed<m::Vector3<f32>, m::Quaternion<f32>>;

pub struct World {
    pub objects: Vec<Item>
}

impl World {
    pub fn new() -> World {
        World {
            objects: Vec::new()
        }
    }

    pub fn add_item(&mut self, item: Item) {
        self.objects.push(item);
    }

    pub fn update(&mut self, state: &mut UpdateState) {
        for object in self.objects.iter_mut() {
            object.update(state);
        }
    }

    pub fn render(&self, ctx: &mut RenderContext) {
        for object in self.objects.iter() {
            object.render(ctx);
        }
    }
}

pub struct Item {
    pub position: WorldPos,
    pub kind: ItemKind
}

pub enum ItemKind {
    Null,
    StaticMesh(Rc<Mesh<Vertex>>),
    Subdomain(World),
    Animation(Box<FnMut(&mut WorldPos, &mut Item, &mut UpdateState)>, Box<Item>),
    Camera(Camera),
}

impl Item {
    pub fn new_null() -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::Null
        }
    }

    pub fn new_subdomain(domain: World) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::Subdomain(domain)
        }
    }

    pub fn new_static(mesh: Rc<Mesh<Vertex>>) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::StaticMesh(mesh)
        }
    }

    pub fn new_animation(function: Box<FnMut(&mut WorldPos, &mut Item, &mut UpdateState)>, item: Item) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::Animation(
                function,
                Box::new(item)
            )
        }
    }

    pub fn new_camera(id: &'static str, perspective: m::Matrix4<f32>) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::Camera(Camera {
                perspective: perspective,
                id: id
            })
        }
    }

    pub fn at(mut self, x: f32, y: f32, z: f32) -> Item {
        self.position.disp = m::Vector3::new(x, y, z);
        self
    }

    pub fn rotate<I: Into<m::Rad<f32>>>(mut self, axis: m::Vector3<f32>, angle: I) -> Item {
        self.position.rot = m::Quaternion::from_axis_angle(axis, angle).normalize();
        self
    }

    pub fn scale(mut self, scale: f32) -> Item {
        self.position.scale = scale;
        self
    }

    pub fn update(&mut self, state: &mut UpdateState) {
        match self.kind {
            ItemKind::Subdomain(ref mut world) => world.update(state),
            ItemKind::Animation(ref mut func, ref mut item) => {
                func(&mut self.position, item, state);
                item.update(state);
            },
            ItemKind::StaticMesh(_) => (),
            ItemKind::Camera(_) => (),
            ItemKind::Null => ()
        }
    }

    pub fn render(&self, ctx: &mut RenderContext) {
        ctx.with_transform(&self.position, |ctx| {
            match self.kind {
                ItemKind::Subdomain(ref world) => {
                    world.render(ctx);
                },
                ItemKind::Animation(_, ref item) => {
                    item.render(ctx);
                },
                ItemKind::StaticMesh(ref mesh) => {
                    ctx.add_mesh(mesh);
                },
                ItemKind::Camera(ref camera) => {
                    ctx.set_camera(camera.clone());
                }
                ItemKind::Null => ()
            }
        });
    }
}

#[derive(Debug, Clone)]
pub struct Mesh<V> {
    pub vertices: Vec<V>,
    pub indices: Vec<u32>
}

impl<V> Mesh<V> {
    pub fn new<I: IntoIterator<Item=V>, J: IntoIterator<Item=u32>>(vertices: I, indices: J) -> Mesh<V> {
        Mesh {
            vertices: vertices.into_iter().collect(),
            indices: indices.into_iter().collect()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub id: &'static str,
    pub perspective: m::Matrix4<f32>
}