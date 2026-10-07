//! Voxel Quest's palette sky.

use bevy::{
    camera::visibility::NoFrustumCulling,
    light::NotShadowCaster,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

use crate::{
    VqShading,
    palette::VqPalette,
    raymarch::{VqShaded, VqShadingUniform, specialize_box, sync_shading},
    shader_path,
};

/// Adds a sky dome to every camera carrying [`VqSky`].
pub struct VqSkyPlugin;

impl Plugin for VqSkyPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<VqSkyMaterial>::default())
            .init_resource::<VqShading>()
            .add_systems(
                PostUpdate,
                (spawn_sky, follow_camera, sync_shading::<VqSkyMaterial>)
                    .chain()
                    .before(TransformSystems::Propagate),
            );
    }
}

/// Put this on a camera to surround it with Voxel Quest's sky. The sky is
/// driven by the first `DirectionalLight` and [`VqShading::time_of_day`].
#[derive(Component, Clone, Copy, Debug, Default, Reflect)]
#[reflect(Component)]
pub struct VqSky;

#[derive(Component)]
struct SkyDome(Entity);

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VqSkyMaterial {
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub palette: Handle<Image>,
    #[uniform(2)]
    pub shading: VqShadingUniform,
}

impl Material for VqSkyMaterial {
    fn fragment_shader() -> ShaderRef {
        shader_path("sky.wgsl")
    }

    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
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

impl VqShaded for VqSkyMaterial {
    fn shading_mut(&mut self) -> &mut VqShadingUniform {
        &mut self.shading
    }
}

fn spawn_sky(
    mut commands: Commands,
    cameras: Query<Entity, (With<VqSky>, Without<SkyDome>)>,
    palette: Res<VqPalette>,
    shading: Res<VqShading>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VqSkyMaterial>>,
) {
    for camera in &cameras {
        let dome = commands
            .spawn((
                Name::new("VQ sky"),
                Mesh3d(meshes.add(Sphere::new(1.0).mesh().ico(3).unwrap())),
                MeshMaterial3d(materials.add(VqSkyMaterial {
                    palette: palette.image.clone(),
                    shading: VqShadingUniform::new(&shading, &palette),
                })),
                Transform::default(),
                NotShadowCaster,
                NoFrustumCulling,
            ))
            .id();
        commands.entity(camera).insert(SkyDome(dome));
    }
}

fn follow_camera(
    cameras: Query<(&Transform, &SkyDome, &Projection), Without<MeshMaterial3d<VqSkyMaterial>>>,
    mut domes: Query<&mut Transform, With<MeshMaterial3d<VqSkyMaterial>>>,
) {
    for (camera, dome, projection) in &cameras {
        if let Ok(mut t) = domes.get_mut(dome.0) {
            // Big enough to contain the near plane, small enough for any far plane.
            let near = match projection {
                Projection::Perspective(p) => p.near,
                Projection::Orthographic(o) => o.near.abs(),
                _ => 0.1,
            };
            t.translation = camera.translation;
            t.scale = Vec3::splat((near * 10.0).max(1.0));
        }
    }
}
