# bevy_voxelquest

Voxel Quest's ray-marched voxel world ported to [Bevy](https://bevy.org) 0.19 as
a set of plugins, with optional [avian3d](https://github.com/Jondolf/avian)
physics.

Voxel Quest (Gavan Woolery, 2014–2016, zlib) drew its world by ray-marching
signed distance fields on the GPU:

- **Terrain:** a heightmap terrain carved by Voronoi rocks.
- **Buildings:** hollow superellipsoid shells with procedural brick, timber
  framing and shingles.
- **Lighting:** shaded through hand-authored per-material colour ramps.

This crate ports those shaders (`src/glsl/PrimShader.c`, `MapLand.c`,
`TerHeightFunc.c`, `WaveFuncs.c`, `PreLightingShader.c`,
`PostLightingShader.c`) and the data they use (`materials.js`,
`primTemplates.js`, `hm0.bmp`/`hm1.bmp`).

The VQ-specific pipeline (a G-buffer of 8 render targets plus a chain of full-screen
passes) is not ported. Instead, every ray-marched object is a normal Bevy entity:

- The mesh is the object's bounding box.
- A custom `Material` marches the SDF inside that box and writes real depth.

So voxel terrain and buildings depth-sort with your regular meshes. They cast
and receive Bevy shadow maps, and they write correct motion vectors, so TAA
and motion blur work. They also get frustum culling, fog and multiple cameras.

## Plugins

`VoxelQuestPlugins` adds everything. Each sub-plugin also works on its own on
top of `VqCorePlugin`.

| Plugin | What it does | Ported from |
|---|---|---|
| `VqCorePlugin` | Shader library, `VqWorldSettings`, `VqShading`, the material palette (`VqPalette`) | `Singleton::updateMatVol` |
| `VqTerrainPlugin` | Generates heightmap + Voronoi rock volume; streams ray-marched terrain tiles around `VqTerrainFocus` | `initMap`, `TerrainMix.c`, `TerHeightFunc.c`, `MapLand.c`, `PrimShader.c` (DOTER) |
| `VqStructurePlugin` | Buildings from VQ primitive templates (`VqStructure`) | `primTemplates.js`, `PrimShader.c` (DOPRIM: `mapSolid`, `udRoundBox`, `getUVW`, bricks, timber, shingles, wood grain) |
| `VqWaterPlugin` | Animated sea plane at sea level | `WaveFuncs.c` |
| `VqSkyPlugin` | Palette sky dome with sun/moon glow (`VqSky` on a camera) | `getFogColor` |
| `VqPixelatePlugin` | VQ's chunky low-res look (`VqPixelate` on a camera) | VQ's quarter-resolution G-buffer |
| `VqPhysicsPlugin` (feature `physics`, on by default) | avian3d heightfield colliders for nearby terrain tiles; voxel colliders for structures | Replaces VQ's GPU-readback voxel grid |

AI, characters, fluids and the editor are not ported.

## Usage

```rust
use bevy::{core_pipeline::tonemapping::Tonemapping, prelude::*};
use bevy_voxelquest::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(VoxelQuestPlugins)
        .add_plugins(avian3d::prelude::PhysicsPlugins::default()) // if using physics
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, terrain: Res<VqTerrain>) {
    let ground = terrain.field.height_at(0.0, 0.0);
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-60.0, ground + 40.0, 80.0).looking_at(Vec3::new(0.0, ground, 0.0), Vec3::Y),
        VqTerrainFocus,          // stream terrain around this entity
        VqSky,                   // VQ sky dome
        Tonemapping::None,       // the palette already holds final colours
    ));
    commands.spawn((
        DirectionalLight { shadow_maps_enabled: true, ..default() },
        Transform::default().looking_to(Vec3::new(-0.5, -0.7, -0.35), Vec3::Y),
    ));
    commands.spawn((
        VqStructure::new(vec![
            VqPrim::new(VqPrimTemplate::tower(), Vec3::ZERO),
            VqPrim::new(VqPrimTemplate::roof_pointed_tower(), Vec3::new(0.0, 8.0, 0.0)),
        ]),
        Transform::from_xyz(0.0, ground + 14.0, 0.0),
    ));
}
```

Run the example from this directory. It uses VQ's original heightmaps from
`../data` when they are present:

```sh
cargo run --release --example world
```

Example controls:
- WASD / Space / Shift to fly; Ctrl to fly faster.
- Hold the right mouse button to look.
- F throws a crate.
- P switches on the pixelated look.
- M switches between palette and PBR lighting.
- `[` and `]` change the time of day.

Environment variables:
- `VQ_PIXELATE=4` starts pixelated.
- `VQ_DROP_CRATES=1` drops crates onto the castle at startup.
- `VQ_PBR=1` starts with PBR lighting.
- `VQ_NO_TAA=1` / `VQ_NO_SHADOWS=1` turn off TAA / sun shadows.
- `VQ_BENCH=N` prints the average frame time over N frames and exits.
- `VQ_SCREENSHOT=out.png` saves a screenshot and exits.

## How it fits together

- **`VqWorldSettings`** holds the world generation parameters.
  - Heightmap: size, resolution, height, sea level, VQ's `mapFreqs`/`mapAmps`.
  - Rocks: the three Voronoi rock layers.
  - Tiles: tile size and view radius.
  - Insert it before adding the plugins. It is read once, at startup.
- **`HeightmapSource`**:
  - `Procedural` (the default) uses tiling ridged noise in place of VQ's
    heightmaps.
  - `VoxelQuestBmp { hm0, hm1 }` mixes VQ's original 2048² heightmaps exactly
    like `TerrainMix.c`.
- **`VqTerrain`** (resource) holds the `TerrainField`. It is the CPU mirror of
  the terrain shader, with `distance`, `height_at` and `normal_at` for
  gameplay and physics.
- **`VqShading`** (resource) selects the lighting model:
  - `Palette` is VQ's own model. It takes the first `DirectionalLight` (its
    direction, hue and shadow map), adds VQ's coloured bounce and rim terms,
    and maps the result per channel through each material's colour ramp. Use
    it with `Tonemapping::None`.
  - `Pbr` uses Bevy's full PBR lighting, with palette albedo.
  - It also sets `time_of_day`, AO strength and rim strength.
- **`VqStructure`** is a list of `VqPrim`s (template + position).
  - Primitives in one structure merge: their interiors are carved out of each
    other's walls, which is how VQ joins towers and walls.
  - The entity's `Transform` moves, rotates and scales the whole building.
  - Templates:
    - `tower`
    - `wall_along_x` / `wall_along_z`
    - `roof_sphere_tower`
    - `roof_barrel_x` / `roof_barrel_z`
    - `roof_pointed_tower`
    - `portal_x` / `portal_z`
    - `VqPrimTemplate::custom(...)` for your own.
- **`VqPalette`** holds VQ's 30 materials (`VqMat`). Use `palette.color(mat,
  variation, light)` to tint regular meshes so they match.
  `VqPaletteSource(json)` loads your own `materials.js`.

### Your own heightmap

`HeightmapSource::Custom(Arc<CustomHeightmap>)` renders a heightmap from
another generator (a world map, an atlas) instead of VQ's:

- **Heights:** real heights in world units, with the sea at y = 0.
- **Layout:** any width × height, placed by `origin` and `cell_size`. It does
  not tile; the edges are clamped.
- **Colours (optional):** per-texel sRGB colours replace the palette's terrain
  materials. Steep ground fades to bare rock, because map texels are coarser
  than cliffs.
- **Rockiness (optional):** per-texel 0..1 values scale the Voronoi rock
  layers, so crags appear on mountains but not on farmland.

`HeightmapSource::Manual` generates nothing at startup. Build a field on any
thread with `TerrainField::generate` or `TerrainField::from_custom`. Insert it
with `VqTerrain::new(field, &mut images, &mut buffers)`. Removing the resource
clears the tiles, and replacing it swaps terrains. The arcs lab's "far land"
uses this to draw a 128 km atlas crop around its island.

To embed only the renderer, use `default-features = false`. That drops avian
physics and the UI-based pixelation, and Bevy's features stay trimmed to
`3d_bevy_render`.

Coordinates: Bevy is Y-up and Voxel Quest is Z-up. Public APIs use Bevy
space. The shaders work in VQ space, `(x, -z, y)`, so the original formulas
port line for line.

## Performance notes

- **Ray-marched boxes:** each terrain tile and structure is a box whose pixels
  each run a ray march. Away from the surface, the terrain march steps with a
  cheap lower bound: one heightmap lookup, then the bare heightfield. Rock
  detail is only evaluated within the band it can affect.
- **Terrain shadows:** terrain does **not** ray-march into shadow maps. Each
  tile has an invisible heightfield proxy mesh that casts its shadows
  (`shadow_proxy_bias` lowers it to avoid acne). A short SDF soft-shadow march
  in the main pass adds back the small-scale self-shadowing of rocks.
  Ray-marching every shadow-map texel used to be over 90% of the frame.
  Structures still ray-march into shadow maps; they are small in light space.
- **Depth prepass reuse:** with a depth prepass (TAA and SSAO add one), the
  prepass does the marching. The main pass reads the hit back from the
  prepass depth and only shades, so TAA costs little extra and can even be
  faster than without it.
- **Tuning:** use `view_radius_tiles`, `max_steps` and the rock layers'
  `fade_distance`.
- **Pixelated look:** `VqPixelate { factor: 4 }` makes everything about 16×
  cheaper and matches VQ's look. VQ itself marched at a quarter of its
  G-buffer resolution.
- **Measured** on a CPU software renderer (lavapipe) at 640×360, with a
  3-cascade shadowed sun. These are relative numbers only; no real-GPU numbers
  yet.

  | Version | Frame time |
  |---|---|
  | Initial port | 15.2 s |
  | Cheaper marching | 12.9 s |
  | Shadow proxies | 1.24 s |
  | Shadow proxies + TAA | 1.06 s |

  `VQ_BENCH=N cargo run --release --example world` prints your own.

## Differences from Voxel Quest

- **Lighting:** large-scale shadows come from Bevy shadow maps (terrain via
  proxy meshes). Only the fine self-shadowing uses VQ-style in-shader soft
  shadows, and only in palette mode; PBR mode uses shadow maps alone. AO is a
  short SDF march along the normal instead of SSAO.
- **Not ported:** radiosity, the water refraction pass and the median filter.
  Bevy's fog stands in for VQ's fog shader.
- **Missing data:** VQ's `voro.bmp`, which caps mountains into mesas, is
  missing from its repo. A procedural Voronoi map replaces it.
- **Terrain materials:** grass also grows on flat ground
  (`VqWorldSettings::grass_flatness`). VQ's final rules only put it on a narrow
  altitude band; set the value to 0 to restore them.
- **Destruction and limbs:** the local destruction/water volume and character
  limbs (`volIdPrim`, `limbTBO`) are not ported.
- **Overhangs:** like VQ, the higher heightmap octaves are sheared with
  altitude (`octave_shear`). That gives overhangs and some floating rock
  shards; set it to 0 for a pure heightfield.

## License

zlib, like Voxel Quest.
