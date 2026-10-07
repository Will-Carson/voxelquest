//! Voxel Quest structures: buildings assembled from primitive templates.
//!
//! A primitive is a hollow superellipsoid box ("rounded box" with an
//! adjustable corner power) clipped by a visibility box. Voxel Quest's
//! `primTemplates.js` defines towers, walls, round and pointed roofs and
//! portals; [`VqPrimTemplate`] exposes them, and you can define your own.
//!
//! ```no_run
//! # use bevy::prelude::*;
//! # use bevy_voxelquest::prelude::*;
//! fn spawn(mut commands: Commands) {
//!     commands.spawn((
//!         VqStructure::new(vec![
//!             VqPrim::new(VqPrimTemplate::tower(), Vec3::ZERO),
//!             VqPrim::new(VqPrimTemplate::roof_pointed_tower(), Vec3::new(0.0, 24.0, 0.0)),
//!         ]),
//!         Transform::from_xyz(10.0, 50.0, 0.0),
//!     ));
//! }
//! ```

use std::sync::OnceLock;

use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        render_resource::{
            AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
        },
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
};
use serde_json::Value;

use crate::{
    VqShading, from_vq,
    palette::VqPalette,
    raymarch::{VqShaded, VqShadingUniform, specialize_box, sync_shading},
    shader_path, to_vq,
};

/// Renders [`VqStructure`]s.
pub struct VqStructurePlugin;

impl Plugin for VqStructurePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<VqStructureMaterial>::default())
            .init_resource::<VqShading>()
            .register_type::<VqStructure>()
            .add_systems(
                PostUpdate,
                (
                    build_structures,
                    sync_transforms,
                    sync_shading::<VqStructureMaterial>,
                )
                    .chain()
                    .after(TransformSystems::Propagate),
            );
    }
}

/// Wall/roof style of a primitive (VQ `matParams.x`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Reflect)]
pub enum VqPrimStyle {
    /// Plaster with exposed brick and a timber frame (Tudor style).
    TimberFrame = 0,
    /// Brick with recessed plaster.
    Brick = 1,
    /// Fish-scale shingles over wooden boards (roofs).
    Roof = 2,
}

/// A primitive template, in Voxel Quest's Z-up coordinates (the same numbers
/// as `primTemplates.js`). Use the constructors for VQ's built-in templates.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct VqPrimTemplate {
    /// Bounds of the rounded box, relative to the primitive's position.
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    /// Visible part of the box, as fractions of the bounds (VQ `visMin`/`visMax`;
    /// a roof keeps only the top half of a sphere with `vis_min.z = 0`).
    pub vis_min: Vec3,
    pub vis_max: Vec3,
    /// Corner radius.
    pub corner_radius: f32,
    /// Wall thickness of the hollow shell.
    pub wall_thickness: f32,
    /// Superellipse power across the horizontal plane (2 = round, 10 ≈ square).
    pub power_xy: f32,
    /// Superellipse power in the vertical direction.
    pub power_z: f32,
    pub style: VqPrimStyle,
}

#[derive(Clone)]
struct NamedTemplate {
    name: String,
    template: VqPrimTemplate,
}

const DEFAULT_TEMPLATES: &str = include_str!("data/prim_templates.json");

fn templates() -> &'static [NamedTemplate] {
    static T: OnceLock<Vec<NamedTemplate>> = OnceLock::new();
    T.get_or_init(|| {
        parse_templates(DEFAULT_TEMPLATES).expect("built-in primTemplates.js is valid")
    })
}

fn parse_templates(json: &str) -> Result<Vec<NamedTemplate>, String> {
    let root: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let list = root["primTemplates"]
        .as_array()
        .ok_or("missing primTemplates")?;
    let v = |t: &Value, k: &str| -> Vec4 {
        let a = |i: usize| t[k][i].as_f64().unwrap_or(0.0) as f32;
        Vec4::new(a(0), a(1), a(2), a(3))
    };
    Ok(list
        .iter()
        .map(|t| {
            let corner = v(t, "04_cornerDis");
            let style = match v(t, "05_matParams").x as i32 {
                0 => VqPrimStyle::TimberFrame,
                1 => VqPrimStyle::Brick,
                _ => VqPrimStyle::Roof,
            };
            NamedTemplate {
                name: t["06_comment"].as_str().unwrap_or("").to_string(),
                template: VqPrimTemplate {
                    bounds_min: v(t, "02_bndMin").truncate(),
                    bounds_max: v(t, "03_bndMax").truncate(),
                    vis_min: v(t, "00_visMin").truncate(),
                    vis_max: v(t, "01_visMax").truncate(),
                    corner_radius: corner.x,
                    wall_thickness: corner.y,
                    power_xy: corner.z,
                    power_z: corner.w,
                    style,
                },
            }
        })
        .collect())
}

impl VqPrimTemplate {
    /// Looks up one of Voxel Quest's templates by its `primTemplates.js`
    /// comment, e.g. `"tower"` or `"roof pointed tower"`.
    pub fn by_name(name: &str) -> Option<Self> {
        templates()
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.template)
    }

    /// Names of all built-in templates.
    pub fn names() -> impl Iterator<Item = &'static str> {
        templates().iter().skip(1).map(|t| t.name.as_str())
    }

    fn builtin(name: &str) -> Self {
        Self::by_name(name).unwrap_or_else(|| panic!("missing built-in template {name}"))
    }

    /// 16×16×32 round-cornered brick tower (walls up to 3/4 of its height).
    pub fn tower() -> Self {
        Self::builtin("tower")
    }
    /// Thick 20×32 curtain wall running along Z (VQ "wall X").
    pub fn wall_along_z() -> Self {
        Self::builtin("wall X")
    }
    /// Thick 32×20 curtain wall running along X (VQ "wall Y").
    pub fn wall_along_x() -> Self {
        Self::builtin("wall Y")
    }
    /// Dome roof for a tower (VQ "roof sphere tower").
    pub fn roof_sphere_tower() -> Self {
        Self::builtin("roof sphere tower")
    }
    /// Barrel roof running along X (VQ "roof sphere X").
    pub fn roof_barrel_x() -> Self {
        Self::builtin("roof sphere X")
    }
    /// Barrel roof running along Z (VQ "roof sphere Y").
    pub fn roof_barrel_z() -> Self {
        Self::builtin("roof sphere Y")
    }
    /// Pointed (conical) tower roof.
    pub fn roof_pointed_tower() -> Self {
        Self::builtin("roof pointed tower")
    }
    /// Arched doorway through a wall that runs along X (VQ "portal X").
    pub fn portal_x() -> Self {
        Self::builtin("portal X")
    }
    /// Arched doorway through a wall that runs along Z (VQ "portal Y").
    pub fn portal_z() -> Self {
        Self::builtin("portal Y")
    }

    /// A custom template, given in Bevy (Y-up) space.
    pub fn custom(
        bounds_min: Vec3,
        bounds_max: Vec3,
        corner_radius: f32,
        wall_thickness: f32,
        style: VqPrimStyle,
    ) -> Self {
        let (a, b) = (to_vq(bounds_min), to_vq(bounds_max));
        Self {
            bounds_min: a.min(b),
            bounds_max: a.max(b),
            vis_min: Vec3::ONE,
            vis_max: Vec3::ONE,
            corner_radius,
            wall_thickness,
            power_xy: 2.0,
            power_z: 2.0,
            style,
        }
    }

    /// Visible region (VQ space, relative to the primitive position).
    fn vis_box(&self) -> (Vec3, Vec3) {
        (
            self.bounds_min * self.vis_min,
            self.bounds_max * self.vis_max,
        )
    }
}

/// One primitive instance in a [`VqStructure`].
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct VqPrim {
    pub template: VqPrimTemplate,
    /// Position relative to the structure (Bevy space).
    pub position: Vec3,
}

impl VqPrim {
    pub fn new(template: VqPrimTemplate, position: Vec3) -> Self {
        Self { template, position }
    }
}

/// A building made of [`VqPrim`]s, rendered as a single ray-marched volume.
/// Primitives that touch are merged (their interiors are carved out of each
/// other's walls). The entity's `Transform` moves, rotates and scales it.
#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component)]
#[require(Transform, Visibility)]
pub struct VqStructure {
    pub prims: Vec<VqPrim>,
    /// Maximum ray-march steps per pixel.
    pub max_steps: u32,
}

impl VqStructure {
    pub fn new(prims: Vec<VqPrim>) -> Self {
        Self {
            prims,
            max_steps: 96,
        }
    }

    /// Local-space (Bevy) bounding box of all primitives.
    pub fn local_aabb(&self) -> Option<(Vec3, Vec3)> {
        self.prims
            .iter()
            .map(|p| {
                let (a, b) = p.template.vis_box();
                let (a, b) = (from_vq(a), from_vq(b));
                (a.min(b) + p.position, a.max(b) + p.position)
            })
            .reduce(|(a0, a1), (b0, b1)| (a0.min(b0), a1.max(b1)))
    }

    /// Signed distance to the structure's walls at a local-space (Bevy) point,
    /// without surface detail. Used for physics.
    pub fn distance(&self, p: Vec3) -> f32 {
        let p = to_vq(p);
        let gpu = self.gpu_prims();
        let (mut sub, mut res) = (f32::MAX, f32::MAX);
        for g in &gpu {
            let vis = sd_box(p - g.vis_center.truncate(), g.vis_half.truncate());
            let b = ud_round_box(
                p - g.box_center.truncate(),
                g.box_dim,
                g.params.x,
                g.params.y,
                g.params.z,
            );
            sub = sub.min((b.z * 0.5).max(vis));
            res = res.min((b.x.max(b.y) * 0.5).max(vis));
        }
        if sub < f32::MAX {
            res = res.max(-(sub - 0.01));
        }
        res
    }

    fn gpu_prims(&self) -> Vec<PrimGpu> {
        self.prims
            .iter()
            .map(|p| {
                let t = &p.template;
                let pos = to_vq(p.position);
                let (vmin, vmax) = t.vis_box();
                let r = t.corner_radius;
                PrimGpu {
                    vis_center: (pos + (vmin + vmax) * 0.5).extend(0.0),
                    vis_half: ((vmax - vmin) * 0.5).abs().extend(0.0),
                    box_center: (pos + (t.bounds_min + t.bounds_max) * 0.5).extend(0.0),
                    box_dim: (((t.bounds_max - r) - (t.bounds_min + r)) * 0.5)
                        .max(Vec3::ZERO)
                        .extend(r),
                    params: Vec4::new(
                        t.power_xy,
                        t.power_z,
                        t.wall_thickness,
                        t.style as u32 as f32,
                    ),
                }
            })
            .collect()
    }
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let d = p.abs() - b;
    d.max_element().min(0.0) + d.max(Vec3::ZERO).length()
}

fn ud_round_box(v: Vec3, dim: Vec4, pxy: f32, pz: f32, wall: f32) -> Vec3 {
    let n = (v.abs() - dim.truncate()).max(Vec3::ZERO) + 1.0e-7;
    let a = (n.x.powf(pxy) + n.y.powf(pxy)).powf(1.0 / pxy);
    let b = (a.powf(pz) + n.z.powf(pz)).powf(1.0 / pz);
    Vec3::new(b - dim.w, (dim.w - wall) - b, b - (dim.w - wall))
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
struct PrimGpu {
    vis_center: Vec4,
    vis_half: Vec4,
    box_center: Vec4,
    box_dim: Vec4,
    params: Vec4,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct StructureParams {
    pub world_from_local: Mat4,
    pub local_from_world: Mat4,
    pub box_min: Vec4,
    pub box_max: Vec4,
}

/// Ray-marches a [`VqStructure`] inside its bounding box.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VqStructureMaterial {
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub palette: Handle<Image>,
    #[uniform(2)]
    pub shading: VqShadingUniform,
    #[uniform(3)]
    pub params: StructureParams,
    #[storage(4, read_only)]
    pub prims: Handle<ShaderBuffer>,
}

impl Material for VqStructureMaterial {
    fn fragment_shader() -> ShaderRef {
        shader_path("structure.wgsl")
    }

    fn prepass_fragment_shader() -> ShaderRef {
        shader_path("structure.wgsl")
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(0.5)
    }

    fn specialize(
        pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        specialize_box(pipeline, descriptor, layout, key)
    }
}

impl VqShaded for VqStructureMaterial {
    fn shading_mut(&mut self) -> &mut VqShadingUniform {
        &mut self.shading
    }
}

fn build_structures(
    mut commands: Commands,
    structures: Query<(Entity, &VqStructure, &GlobalTransform), Changed<VqStructure>>,
    palette: Res<VqPalette>,
    shading: Res<VqShading>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<VqStructureMaterial>>,
) {
    for (entity, structure, transform) in &structures {
        let Some((min, max)) = structure.local_aabb() else {
            commands
                .entity(entity)
                .remove::<(Mesh3d, MeshMaterial3d<VqStructureMaterial>)>();
            continue;
        };
        let (min, max) = (min - 0.5, max + 0.5);
        let gpu = structure.gpu_prims();
        let count = gpu.len();
        let buffer = buffers.add(ShaderBuffer::from(gpu));
        let world_from_local = transform.affine().into();
        let material = materials.add(VqStructureMaterial {
            palette: palette.image.clone(),
            shading: VqShadingUniform::new(&shading, &palette),
            params: StructureParams {
                world_from_local,
                local_from_world: Mat4::inverse(&world_from_local),
                box_min: min.extend(count as f32),
                box_max: max.extend(structure.max_steps.max(8) as f32),
            },
            prims: buffer,
        });
        let size = max - min;
        let mesh = Mesh::from(Cuboid::new(size.x, size.y, size.z)).translated_by((min + max) * 0.5);
        commands
            .entity(entity)
            .insert((Mesh3d(meshes.add(mesh)), MeshMaterial3d(material)));
    }
}

type MovedStructure = Or<(
    Changed<GlobalTransform>,
    Changed<MeshMaterial3d<VqStructureMaterial>>,
)>;

fn sync_transforms(
    moved: Query<(&GlobalTransform, &MeshMaterial3d<VqStructureMaterial>), MovedStructure>,
    mut materials: ResMut<Assets<VqStructureMaterial>>,
) {
    for (transform, handle) in &moved {
        if let Some(mut m) = materials.get_mut(&handle.0) {
            let world_from_local: Mat4 = transform.affine().into();
            m.params.world_from_local = world_from_local;
            m.params.local_from_world = world_from_local.inverse();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_templates_parse() {
        let names: Vec<_> = VqPrimTemplate::names().collect();
        assert!(names.contains(&"tower"));
        assert!(names.contains(&"roof pointed tower"));
        let t = VqPrimTemplate::tower();
        assert_eq!(t.bounds_max, Vec3::new(8.0, 8.0, 16.0));
    }

    #[test]
    fn tower_is_hollow() {
        let s = VqStructure::new(vec![VqPrim::new(VqPrimTemplate::tower(), Vec3::ZERO)]);
        // Centre of the tower is empty, the wall is solid, outside is empty.
        assert!(s.distance(Vec3::ZERO) > 0.0);
        assert!(s.distance(Vec3::new(7.0, 0.0, 0.0)) < 0.0);
        assert!(s.distance(Vec3::new(12.0, 0.0, 0.0)) > 0.0);
    }
}
