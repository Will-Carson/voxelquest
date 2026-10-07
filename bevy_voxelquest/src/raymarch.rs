//! Plumbing shared by the ray-marched materials.

use bevy::{
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        Face, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    },
};

use crate::{VqShading, VqShadingMode, palette::VqPalette};

/// GPU mirror of [`VqShading`] (`VqShading` in `common.wgsl`).
#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct VqShadingUniform {
    pub mode: u32,
    pub palette_len: f32,
    pub time_of_day: f32,
    pub ao_strength: f32,
    pub rim_strength: f32,
    pub fallback_light_x: f32,
    pub fallback_light_y: f32,
    pub fallback_light_z: f32,
}

impl VqShadingUniform {
    pub fn new(shading: &VqShading, palette: &VqPalette) -> Self {
        let l = shading.fallback_light_direction.normalize_or(Vec3::Y);
        Self {
            mode: match shading.mode {
                VqShadingMode::Palette => 0,
                VqShadingMode::Pbr => 1,
            },
            palette_len: palette.len() as f32,
            time_of_day: shading.time_of_day,
            ao_strength: shading.ao_strength,
            rim_strength: shading.rim_strength,
            fallback_light_x: l.x,
            fallback_light_y: l.y,
            fallback_light_z: l.z,
        }
    }
}

/// Materials that carry the shared palette + shading bindings (0..=2).
pub trait VqShaded: Asset {
    fn shading_mut(&mut self) -> &mut VqShadingUniform;
}

/// Pushes [`VqShading`] changes into every material of type `M`.
pub fn sync_shading<M: VqShaded>(
    shading: Res<VqShading>,
    palette: Res<VqPalette>,
    mut materials: ResMut<Assets<M>>,
) {
    if !shading.is_changed() {
        return;
    }
    let uniform = VqShadingUniform::new(&shading, &palette);
    let ids: Vec<_> = materials.ids().collect();
    for id in ids {
        if let Some(mut m) = materials.get_mut(id) {
            *m.shading_mut() = uniform;
        }
    }
}

/// Pipeline tweak for ray-marched bounding boxes: draw back faces only, so
/// every pixel covered by the box runs the shader exactly once, even when the
/// camera is inside the box.
pub fn specialize_box<M: Material>(
    _pipeline: &MaterialPipeline,
    descriptor: &mut RenderPipelineDescriptor,
    _layout: &MeshVertexBufferLayoutRef,
    _key: MaterialPipelineKey<M>,
) -> Result<(), SpecializedMeshPipelineError> {
    descriptor.primitive.cull_mode = Some(Face::Front);
    Ok(())
}
