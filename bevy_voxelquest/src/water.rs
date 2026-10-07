//! Voxel Quest's sea: an animated plane at sea level.

use bevy::{
    light::NotShadowCaster,
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

use crate::{
    VqShading, VqWorldSettings,
    palette::VqPalette,
    raymarch::{VqShaded, VqShadingUniform, sync_shading},
    shader_path,
    terrain::VqTerrainFocus,
};

/// Spawns a sea plane at [`VqWorldSettings::sea_height`] that follows the
/// [`VqTerrainFocus`].
pub struct VqWaterPlugin;

impl Plugin for VqWaterPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<VqWaterMaterial>::default())
            .init_resource::<VqShading>()
            .add_systems(Startup, spawn_water)
            .add_systems(Update, (follow_focus, sync_shading::<VqWaterMaterial>));
    }
}

/// The sea plane entity.
#[derive(Component)]
pub struct VqWater;

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct WaterParams {
    pub wave_scale: f32,
    pub height: f32,
    pub time_scale: f32,
    pub alpha_min: f32,
    pub alpha_max: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            wave_scale: 1.0,
            height: 1.0,
            time_scale: 0.4,
            alpha_min: 0.6,
            alpha_max: 0.95,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VqWaterMaterial {
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub palette: Handle<Image>,
    #[uniform(2)]
    pub shading: VqShadingUniform,
    #[uniform(3)]
    pub params: WaterParams,
}

impl Material for VqWaterMaterial {
    fn fragment_shader() -> ShaderRef {
        shader_path("water.wgsl")
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }
}

impl VqShaded for VqWaterMaterial {
    fn shading_mut(&mut self) -> &mut VqShadingUniform {
        &mut self.shading
    }
}

fn spawn_water(
    mut commands: Commands,
    settings: Res<VqWorldSettings>,
    palette: Res<VqPalette>,
    shading: Res<VqShading>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VqWaterMaterial>>,
) {
    let size = settings.tile_size * (settings.view_radius_tiles as f32 * 2.0 + 3.0);
    // VQ samples waves at a quarter of the world position, in a 4096-cell world.
    let k = settings.height_max / 4096.0;
    commands.spawn((
        Name::new("VQ sea"),
        VqWater,
        Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size))),
        MeshMaterial3d(materials.add(VqWaterMaterial {
            palette: palette.image.clone(),
            shading: VqShadingUniform::new(&shading, &palette),
            params: WaterParams {
                wave_scale: 4.0 * k.max(0.05),
                ..default()
            },
        })),
        Transform::from_xyz(0.0, settings.sea_height(), 0.0),
        NotShadowCaster,
    ));
}

fn follow_focus(
    settings: Res<VqWorldSettings>,
    focus: Query<&GlobalTransform, With<VqTerrainFocus>>,
    mut water: Query<&mut Transform, With<VqWater>>,
) {
    let Some(f) = focus.iter().next() else { return };
    let snap = settings.tile_size;
    for mut t in &mut water {
        let p = f.translation();
        t.translation.x = (p.x / snap).round() * snap;
        t.translation.z = (p.z / snap).round() * snap;
        t.translation.y = settings.sea_height();
    }
}
