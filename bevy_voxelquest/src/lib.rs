//! Voxel Quest's procedural, ray-marched voxel rendering as a set of Bevy plugins.
//!
//! The original Voxel Quest renderer (Gavan Woolery, 2014-2016, zlib) draws the
//! world by ray-marching signed distance fields on the GPU: a heightmap terrain
//! roughened by Voronoi "rocks", and buildings made of hollow superellipsoid
//! shells with procedural brick, timber framing and shingles. Lighting is
//! mapped through hand-authored per-material colour ramps (`materials.js`),
//! which is what gives the game its painterly look.
//!
//! This crate ports that to Bevy. Instead of VQ's bespoke full-screen G-buffer
//! pipeline, every ray-marched object is an ordinary Bevy mesh (its bounding
//! box) with a custom [`Material`](bevy::pbr::Material) that marches the SDF
//! inside the box and writes real depth. That means the voxel world:
//!
//! * depth-tests against, and is lit alongside, regular Bevy meshes,
//! * casts and receives Bevy shadow maps (the prepass shader marches too),
//! * works with Bevy fog, multiple cameras, frustum culling, etc.
//!
//! # Plugins
//!
//! [`VoxelQuestPlugins`] adds everything; each sub-plugin can also be used on
//! its own (they all depend on [`VqCorePlugin`]):
//!
//! | Plugin | What it does |
//! |---|---|
//! | [`VqCorePlugin`] | Shared shaders, [`VqWorldSettings`], the material palette |
//! | [`terrain::VqTerrainPlugin`] | Generates the heightmap + Voronoi volume, renders streamed terrain tiles |
//! | [`structure::VqStructurePlugin`] | Buildings made from VQ primitive templates ([`structure::VqStructure`]) |
//! | [`water::VqWaterPlugin`] | Animated sea plane using VQ's 8-wave function |
//! | [`sky::VqSkyPlugin`] | VQ's palette-driven sky dome with sun/moon glow |
//! | [`pixelate::VqPixelatePlugin`] | VQ's chunky low-resolution look ([`pixelate::VqPixelate`]) |
//! | `physics::VqPhysicsPlugin` | (feature `physics`) avian3d colliders built from the same SDFs |
//!
//! # Coordinates
//!
//! Bevy is Y-up; Voxel Quest is Z-up. All public APIs use Bevy coordinates.
//! Internally the SDFs are evaluated in "VQ space", `(x, -z, y)`, so the
//! original formulas port unchanged (see [`to_vq`]).

use bevy::{asset::embedded_asset, prelude::*, shader::load_shader_library};

pub mod noise;
pub mod palette;
#[cfg(feature = "pixelate")]
pub mod pixelate;
mod raymarch;
pub mod settings;
pub mod sky;
pub mod structure;
pub mod terrain;
pub mod water;

#[cfg(feature = "physics")]
pub mod physics;

pub use settings::*;

/// Convenient imports.
pub mod prelude {
    pub use crate::{
        VoxelQuestPlugins, VqCorePlugin,
        palette::{VqMat, VqPalette},
        settings::*,
        sky::{VqSky, VqSkyPlugin},
        structure::{VqPrim, VqPrimTemplate, VqStructure, VqStructurePlugin},
        terrain::{VqTerrain, VqTerrainFocus, VqTerrainPlugin},
        water::VqWaterPlugin,
    };

    #[cfg(feature = "physics")]
    pub use crate::physics::VqPhysicsPlugin;
    #[cfg(feature = "pixelate")]
    pub use crate::pixelate::{VqPixelate, VqPixelatePlugin};
}

/// All Voxel Quest plugins.
///
/// With the `physics` feature this includes `VqPhysicsPlugin`, which expects
/// avian3d's `PhysicsPlugins` to be added by the app.
pub struct VoxelQuestPlugins;

impl PluginGroup for VoxelQuestPlugins {
    fn build(self) -> bevy::app::PluginGroupBuilder {
        let group = bevy::app::PluginGroupBuilder::start::<Self>()
            .add(VqCorePlugin)
            .add(terrain::VqTerrainPlugin)
            .add(structure::VqStructurePlugin)
            .add(water::VqWaterPlugin)
            .add(sky::VqSkyPlugin);
        #[cfg(feature = "pixelate")]
        let group = group.add(pixelate::VqPixelatePlugin);
        #[cfg(feature = "physics")]
        let group = group.add(physics::VqPhysicsPlugin);
        group
    }
}

/// Shared resources used by every other Voxel Quest plugin: the WGSL shader
/// library, [`VqWorldSettings`] and the [`palette::VqPalette`] built from
/// Voxel Quest's `materials.js`.
pub struct VqCorePlugin;

impl Plugin for VqCorePlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/noise.wgsl");
        load_shader_library!(app, "shaders/common.wgsl");
        load_shader_library!(app, "shaders/terrain_sdf.wgsl");
        embedded_asset!(app, "shaders/terrain.wgsl");
        embedded_asset!(app, "shaders/structure.wgsl");
        embedded_asset!(app, "shaders/water.wgsl");
        embedded_asset!(app, "shaders/sky.wgsl");
        embedded_asset!(app, "shaders/shadow_proxy.wgsl");

        app.init_resource::<VqWorldSettings>()
            .register_type::<VqWorldSettings>()
            .add_plugins(palette::VqPalettePlugin);
    }
}

/// Shader asset path of one of this crate's embedded shaders.
pub(crate) fn shader_path(name: &str) -> bevy::shader::ShaderRef {
    bevy::shader::ShaderRef::Path(
        bevy::asset::AssetPath::parse(&format!("embedded://bevy_voxelquest/shaders/{name}"))
            .into_owned(),
    )
}

/// Converts a Bevy (Y-up) position into Voxel Quest's Z-up space.
#[inline]
pub fn to_vq(p: Vec3) -> Vec3 {
    Vec3::new(p.x, -p.z, p.y)
}

/// Converts a Voxel Quest (Z-up) position into Bevy's Y-up space.
#[inline]
pub fn from_vq(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y)
}
