use bevy::prelude::*;

/// Global parameters of the generated Voxel Quest world.
///
/// Distances are in world units. One unit corresponds to one Voxel Quest
/// "cell"; the original game used roughly half a metre per cell, so scale the
/// entities you put into the world accordingly (or scale the structures).
///
/// Changing terrain parameters after startup is not supported yet: they are
/// read once when the terrain is generated.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct VqWorldSettings {
    /// Seed for every procedural step.
    pub seed: u32,
    /// Period of the (tiling) heightmap in world units (VQ: `cellsPerWorld`).
    pub world_size: f32,
    /// Resolution of the generated heightmap (it is square).
    pub heightmap_resolution: u32,
    /// Height of a heightmap value of 1.0 (VQ: `heightMapMaxInCells`).
    pub height_max: f32,
    /// Sea level as a fraction of [`Self::height_max`] (VQ: `seaLevel`, 100/255).
    pub sea_level: f32,
    /// Frequencies of the 4 heightmap octaves (VQ: `mapFreqs`).
    pub map_freqs: Vec4,
    /// Amplitudes of the 4 heightmap octaves (VQ: `mapAmps`).
    pub map_amps: Vec4,
    /// How strongly the higher octaves are sheared with altitude. VQ samples
    /// them at `(xy + z)`, which slants cliffs and creates overhangs.
    pub octave_shear: f32,
    /// Large boulders carved along Voronoi cell borders.
    pub rocks_large: RockLayer,
    /// Medium rocks.
    pub rocks_medium: RockLayer,
    /// Pebbles; only rendered close to the camera.
    pub rocks_small: RockLayer,
    /// Fine bumps sampled from the heightmap itself.
    pub bump_depth: f32,
    /// Where the heightmap comes from.
    pub heightmap_source: HeightmapSource,
    /// Terrain is drawn as square tiles of this size (one ray-marched box each).
    pub tile_size: f32,
    /// Tiles are kept loaded within this many tiles of the [`crate::terrain::VqTerrainFocus`].
    pub view_radius_tiles: i32,
    /// Maximum ray-march steps per pixel per tile.
    pub max_steps: u32,
    /// Grass grows on ground whose normal's vertical component exceeds this.
    /// Voxel Quest's final build only had grass on a narrow altitude band; set
    /// to 0 to use its rules unchanged.
    pub grass_flatness: f32,
}

/// One scale of Voronoi rock displacement (VQ `MapLand.c`).
#[derive(Clone, Copy, Debug, Reflect)]
pub struct RockLayer {
    /// Size of one Voronoi cell in world units.
    pub cell_size: f32,
    /// Maximum displacement into the surface, in world units.
    pub depth: f32,
    /// Exponent applied to `1 - centreness`; higher values give sharper cracks.
    pub sharpness: f32,
    /// Distance from the camera at which this layer has faded out (rendering only).
    pub fade_distance: f32,
}

/// Source of the base heightmap.
#[derive(Clone, Debug, Default, Reflect)]
pub enum HeightmapSource {
    /// Fully procedural: tiling ridged noise stands in for Voxel Quest's
    /// real-world heightmaps, then goes through VQ's `TerrainMix` step.
    #[default]
    Procedural,
    /// Voxel Quest's original `data/hm0.bmp` and `data/hm1.bmp` (2048² 24-bit
    /// BMPs, three heightmaps per file), mixed exactly like `TerrainMix.c`.
    /// Paths are filesystem paths, read synchronously at startup. Falls back
    /// to [`HeightmapSource::Procedural`] if they can't be read.
    VoxelQuestBmp { hm0: String, hm1: String },
}

impl Default for VqWorldSettings {
    fn default() -> Self {
        Self {
            seed: 1,
            world_size: 4096.0,
            heightmap_resolution: 1024,
            height_max: 256.0,
            sea_level: 100.0 / 255.0,
            map_freqs: Vec4::new(1.0, 1.0, 2.0, 4.0),
            map_amps: Vec4::new(1.0, 0.25, 0.125, 0.0625),
            octave_shear: 1.0,
            rocks_large: RockLayer {
                cell_size: 48.0,
                depth: 18.0,
                sharpness: 8.0,
                fade_distance: 4000.0,
            },
            rocks_medium: RockLayer {
                cell_size: 8.0,
                depth: 2.5,
                sharpness: 4.0,
                fade_distance: 600.0,
            },
            rocks_small: RockLayer {
                cell_size: 1.5,
                depth: 0.35,
                sharpness: 4.0,
                fade_distance: 90.0,
            },
            bump_depth: 6.0,
            heightmap_source: HeightmapSource::Procedural,
            tile_size: 256.0,
            view_radius_tiles: 6,
            max_steps: 160,
            grass_flatness: 0.8,
        }
    }
}

impl VqWorldSettings {
    /// World-space height of the sea surface (before waves).
    pub fn sea_height(&self) -> f32 {
        self.sea_level * self.height_max
    }
}

/// How ray-marched surfaces are lit. Changes apply to all Voxel Quest
/// materials on the next frame.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct VqShading {
    /// Lighting model.
    pub mode: VqShadingMode,
    /// 0 = night (moon high), 1 = day (sun high). Tints VQ's bounce lighting
    /// and darkens the palette at night, like VQ's `timeOfDay`.
    pub time_of_day: f32,
    /// Strength of the SDF ambient occlusion, 0..1.
    pub ao_strength: f32,
    /// Strength of VQ's cyan/orange rim lighting, 0..1.
    pub rim_strength: f32,
    /// Light direction (pointing *towards* the light) used when the scene has
    /// no `DirectionalLight`.
    pub fallback_light_direction: Vec3,
}

/// Lighting model for [`VqShading`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Reflect)]
pub enum VqShadingMode {
    /// Voxel Quest's own model: the first `DirectionalLight` (direction, hue
    /// and shadow map) plus coloured bounce terms, mapped through the
    /// per-material colour ramps of the palette, channel by channel.
    /// Point/spot lights are ignored. Looks best with `Tonemapping::None`.
    #[default]
    Palette,
    /// Bevy's standard PBR lighting (all light types, environment maps,
    /// SSAO...) with the palette providing the albedo.
    Pbr,
}

impl Default for VqShading {
    fn default() -> Self {
        Self {
            mode: VqShadingMode::Palette,
            time_of_day: 1.0,
            ao_strength: 1.0,
            rim_strength: 1.0,
            fallback_light_direction: Vec3::new(0.4, 0.8, 0.3),
        }
    }
}
