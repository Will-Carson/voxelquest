//! Voxel Quest's chunky low-resolution look.
//!
//! VQ ray-marched its G-buffer at a quarter of the window resolution and
//! upscaled it with nearest-neighbour sampling, which is a big part of its
//! pixel-art feel. [`VqPixelate`] does the same for any `Camera3d`: the camera
//! renders into a small image that a full-screen UI node shows upscaled.

use bevy::{
    camera::RenderTarget,
    image::ImageSampler,
    prelude::*,
    render::render_resource::TextureFormat,
    window::{PrimaryWindow, WindowRef},
};

/// Manages [`VqPixelate`] cameras.
pub struct VqPixelatePlugin;

impl Plugin for VqPixelatePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<VqPixelate>()
            .add_systems(PostUpdate, update_pixelated_cameras);
    }
}

/// Render this camera at `1 / factor` of the window resolution and upscale
/// it without filtering. VQ used a factor of 4 relative to its G-buffer
/// (8 relative to the window, with its default 2× render scale).
///
/// The camera must render to the primary window (its default). The plugin
/// takes over its [`RenderTarget`] and spawns a 2D camera + UI node to
/// present the result.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct VqPixelate {
    pub factor: u32,
}

impl Default for VqPixelate {
    fn default() -> Self {
        Self { factor: 4 }
    }
}

#[derive(Component)]
struct PixelateState {
    image: Handle<Image>,
    size: UVec2,
}

fn update_pixelated_cameras(
    mut commands: Commands,
    window: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<(Entity, &VqPixelate, &Camera, Option<&mut PixelateState>)>,
    mut images: ResMut<Assets<Image>>,
) {
    let Ok(window) = window.single() else { return };
    let physical = UVec2::new(window.physical_width(), window.physical_height());
    if physical.x == 0 || physical.y == 0 {
        return;
    }
    for (entity, pixelate, camera, state) in &mut cameras {
        let size = (physical / pixelate.factor.max(1)).max(UVec2::ONE);
        match state {
            Some(state) if state.size == size => {}
            Some(mut state) => {
                if let Some(mut image) = images.get_mut(&state.image) {
                    image.resize(bevy::render::render_resource::Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    });
                }
                state.size = size;
            }
            None => {
                let mut image =
                    Image::new_target_texture(size.x, size.y, TextureFormat::Rgba8UnormSrgb, None);
                image.sampler = ImageSampler::nearest();
                let image = images.add(image);

                let presenter = commands
                    .spawn((
                        Name::new("VQ pixelate presenter"),
                        Camera2d,
                        Camera {
                            order: camera.order + 1,
                            ..default()
                        },
                        RenderTarget::Window(WindowRef::Primary),
                    ))
                    .id();
                commands.spawn((
                    Node {
                        width: percent(100),
                        height: percent(100),
                        ..default()
                    },
                    ImageNode::new(image.clone()),
                    UiTargetCamera(presenter),
                ));
                commands.entity(entity).insert((
                    RenderTarget::Image(image.clone().into()),
                    PixelateState { image, size },
                ));
            }
        }
    }
}
