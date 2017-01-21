use cgmath as m;
use cgmath::{Angle, Rotation3, Transform, InnerSpace};
use glutin::VirtualKeyCode;

use std::collections::{HashSet, HashMap};
use std::rc::Rc;

use ::{Vertex, DirectionalLight, PointLight, SpotLight};


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
    pub shaded_geometry: (Vec<Vertex>, Vec<u32>),
    pub unshaded_geometry: (Vec<Vertex>, Vec<u32>),
    pub transform_stack: Vec<WorldPos>,
    pub cameras: HashMap<&'static str, (WorldPos, Camera)>,
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
}

impl RenderContext {
    pub fn new() -> RenderContext {
        RenderContext {
            shaded_geometry: (Vec::new(), Vec::new()),
            unshaded_geometry: (Vec::new(), Vec::new()),
            transform_stack: Vec::new(),
            cameras: HashMap::new(),
            directional_lights: Vec::new(),
            point_lights: Vec::new(),
            spot_lights: Vec::new()
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
        let (ref mut vertices, ref mut indices) = self.shaded_geometry;
        vertices.clear();
        indices.clear();
        let (ref mut vertices, ref mut indices) = self.unshaded_geometry;
        vertices.clear();
        indices.clear();
        self.cameras.clear();
        self.directional_lights.clear();
        self.point_lights.clear();
        self.spot_lights.clear();
    }

    fn get_pos(&self) -> WorldPos {
        if let Some(transform) = self.transform_stack.last() {
            transform.clone()
        } else {
            WorldPos::one()
        }
    }

    fn transform_pos(stack: &[WorldPos], pos: m::Point3<f32>) -> m::Point3<f32> {
        if let Some(transform) = stack.last() {
            transform.transform_point(pos)
        } else {
            pos
        }
    }

    fn transform_dir(stack: &[WorldPos], dir: m::Vector3<f32>) -> m::Vector3<f32> {
        if let Some(transform) = stack.last() {
            transform.transform_vector(dir)
        } else {
            dir
        }
    }

    fn transform_worldpos(&self, pos: &WorldPos) -> WorldPos {
        if let Some(transform) = self.transform_stack.last() {
            transform.concat(pos)
        } else {
            pos.clone()
        }
    }

    pub fn add_vertices(mesh: &Mesh<Vertex>, geometry: &mut (Vec<Vertex>, Vec<u32>), stack: &[WorldPos]) {
        let (ref mut vertices, ref mut indices) = *geometry;
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
                vertex.normal = Self::transform_dir(stack, vertex.normal.into()).into();
                vertex
            })
        );
    }

    pub fn add_mesh(&mut self, mesh: &Mesh<Vertex>) {
        Self::add_vertices(mesh, &mut self.shaded_geometry, &self.transform_stack);
    }

    pub fn add_unshaded_mesh(&mut self, mesh: &Mesh<Vertex>) {
        Self::add_vertices(mesh, &mut self.unshaded_geometry, &self.transform_stack);
    }

    pub fn add_directional_light(&mut self, intensity: LightIntensity) {
        let pos = self.get_pos();
        self.directional_lights.push(DirectionalLight {
            direction: pos.transform_vector([0.0, 0.0, 1.0].into()).extend(0.0).into(),
            ambient: intensity.ambient.extend(0.0).into(),
            diffuse: intensity.diffuse.extend(0.0).into(),
            specular: intensity.specular.extend(0.0).into(),
        });
    }

    pub fn add_point_light(&mut self, intensity: LightIntensity, falloff: LightFalloff) {
        let pos = self.get_pos();
        self.point_lights.push(PointLight {
            position: pos.disp.extend(0.0).into(),
            ambient: intensity.ambient.extend(0.0).into(),
            diffuse: intensity.diffuse.extend(0.0).into(),
            specular: intensity.specular.extend(0.0).into(),
            falloff: [falloff.constant, falloff.linear, falloff.quadratic, 0.0]
        });
    }

    pub fn add_spot_light(&mut self, intensity: LightIntensity, falloff: LightFalloff, cone: m::Rad<f32>, outercone: m::Rad<f32>) {
        let pos = self.get_pos();
        self.spot_lights.push(SpotLight {
            position: pos.disp.extend(0.0).into(),
            direction: pos.transform_vector([0.0, 0.0, 1.0].into()).extend(0.0).into(),
            ambient: intensity.ambient.extend(0.0).into(),
            diffuse: intensity.diffuse.extend(0.0).into(),
            specular: intensity.specular.extend(0.0).into(),
            falloff: [falloff.constant, falloff.linear, falloff.quadratic, 0.0],
            cone: [cone.cos(), outercone.cos(), 0.0, 0.0]
        });
    }

    pub fn set_camera(&mut self, camera: Camera) {
        let pos = self.get_pos();
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
    StaticUnshadedMesh(Rc<Mesh<Vertex>>),
    Subdomain(World),
    Animation(Box<FnMut(&mut WorldPos, &mut Item, &mut UpdateState)>, Box<Item>),
    Camera(Camera),
    DirectionalLight(LightIntensity),
    PointLight(LightIntensity, LightFalloff),
    SpotLight(LightIntensity, LightFalloff, m::Rad<f32>, m::Rad<f32>)
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

    pub fn new_static_unshaded(mesh: Rc<Mesh<Vertex>>) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::StaticUnshadedMesh(mesh)
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

    pub fn new_directional_light(intensity: LightIntensity) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::DirectionalLight(intensity)
        }
    }

    pub fn new_point_light(intensity: LightIntensity, falloff: LightFalloff) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::PointLight(intensity, falloff)
        }
    }

    pub fn new_spot_light(intensity: LightIntensity, falloff: LightFalloff, cone: m::Rad<f32>, outercone: m::Rad<f32>) -> Item {
        Item {
            position: WorldPos::one(),
            kind: ItemKind::SpotLight(intensity, falloff, cone, outercone)
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
            ItemKind::StaticUnshadedMesh(_) => (),
            ItemKind::Camera(_) => (),
            ItemKind::DirectionalLight(_) => (),
            ItemKind::PointLight(_, _) => (),
            ItemKind::SpotLight(_, _, _, _) => (),
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
                ItemKind::StaticUnshadedMesh(ref mesh) => {
                    ctx.add_unshaded_mesh(mesh);
                },
                ItemKind::Camera(ref camera) => {
                    ctx.set_camera(camera.clone());
                },
                ItemKind::DirectionalLight(ref intensity) => {
                    ctx.add_directional_light(intensity.clone());
                },
                ItemKind::PointLight(ref intensity, falloff) => {
                    ctx.add_point_light(intensity.clone(), falloff)
                },
                ItemKind::SpotLight(ref intensity, falloff, cone, outercone) => {
                    ctx.add_spot_light(intensity.clone(), falloff, cone, outercone);
                },
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

#[derive(Debug, Clone)]
pub struct LightIntensity {
    pub ambient: m::Vector3<f32>,
    pub diffuse: m::Vector3<f32>,
    pub specular: m::Vector3<f32>
}

#[derive(Debug, Clone, Copy)]
pub struct LightFalloff {
    pub constant: f32,
    pub linear: f32,
    pub quadratic: f32
}