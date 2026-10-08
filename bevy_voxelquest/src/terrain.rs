//! Voxel Quest terrain: generation, a CPU mirror of the distance field, and
//! streamed ray-marched tiles.

use std::sync::Arc;

use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowReceiver,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::{
        render_resource::{
            AsBindGroup, Extent3d, RenderPipelineDescriptor, ShaderType,
            SpecializedMeshPipelineError, TextureDimension, TextureFormat,
        },
        storage::ShaderBuffer,
    },
    shader::ShaderRef,
};

use crate::{
    CustomHeightmap, HeightmapSource, RockLayer, VqShading, VqWorldSettings, from_vq,
    noise::{calc_noise, ridged_fbm, tiled_noise, voronoi_map, voronoi_volume},
    palette::VqPalette,
    raymarch::{VqShaded, VqShadingUniform, specialize_box, sync_shading},
    shader_path, to_vq,
};

/// Resolution of the tiling Voronoi rock volume.
pub const VORO_RES: u32 = 64;
/// Voronoi cells per side of the rock volume.
pub const VORO_CELLS: u32 = 8;

/// Generates the terrain and streams ray-marched terrain tiles around the
/// entity marked [`VqTerrainFocus`] (or the origin if there is none).
pub struct VqTerrainPlugin;

impl Plugin for VqTerrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MaterialPlugin::<VqTerrainMaterial>::default(),
            MaterialPlugin::<VqShadowProxyMaterial>::default(),
        ))
            .init_resource::<VqShading>()
            .init_resource::<LoadedTiles>()
            .add_systems(
                Update,
                (stream_tiles, sync_shading::<VqTerrainMaterial>).chain(),
            );
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let proxy = world
            .resource_mut::<Assets<VqShadowProxyMaterial>>()
            .add(VqShadowProxyMaterial {});
        world.insert_resource(ShadowProxyMaterial(proxy));

        let settings = world.resource::<VqWorldSettings>().clone();
        if matches!(settings.heightmap_source, HeightmapSource::Manual) {
            return;
        }
        let start = std::time::Instant::now();
        let field = TerrainField::generate(&settings);
        info!("bevy_voxelquest: generated terrain in {:.2?}", start.elapsed());
        world.resource_scope(|world, mut images: Mut<Assets<Image>>| {
            let mut buffers = world.resource_mut::<Assets<ShaderBuffer>>();
            let terrain = VqTerrain::new(field, &mut images, &mut buffers);
            world.insert_resource(terrain);
        });
    }
}

/// The generated terrain. Insert one (see [`VqTerrain::new`]) to show
/// terrain, replace it to swap terrains, remove it to clear the tiles.
#[derive(Resource, Clone)]
pub struct VqTerrain {
    /// CPU copy of the distance field, for gameplay queries and physics.
    pub field: Arc<TerrainField>,
    /// GPU heightmap: `(height, mesa cap or rockiness)` per texel.
    pub heightmap: Handle<ShaderBuffer>,
    /// GPU Voronoi rock volume.
    pub voro: Handle<Image>,
    /// GPU per-texel albedo of a custom map (one empty texel otherwise).
    pub albedo: Handle<ShaderBuffer>,
}

impl VqTerrain {
    /// Uploads a generated field. Building the field (`TerrainField::generate`)
    /// is the slow part and can run on another thread; this is cheap.
    pub fn new(
        field: TerrainField,
        images: &mut Assets<Image>,
        buffers: &mut Assets<ShaderBuffer>,
    ) -> Self {
        let heightmap = buffers.add(ShaderBuffer::from(field.heightmap.clone()));
        let albedo = if field.albedo.is_empty() {
            vec![0u32]
        } else {
            field.albedo.clone()
        };
        let albedo = buffers.add(ShaderBuffer::from(albedo));
        let n = field.voro_res as u32;
        let mut voro = Image::new(
            Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: n,
            },
            TextureDimension::D3,
            field.voro.clone(),
            TextureFormat::R8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        voro.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::Repeat,
            address_mode_v: ImageAddressMode::Repeat,
            address_mode_w: ImageAddressMode::Repeat,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            ..default()
        });
        VqTerrain {
            field: Arc::new(field),
            heightmap,
            voro: images.add(voro),
            albedo,
        }
    }
}

/// Marks the entity (usually the camera or player) that terrain tiles are
/// streamed around.
#[derive(Component, Default, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct VqTerrainFocus;

/// A spawned terrain tile. `coord` is in tile units: the tile covers
/// `x ∈ [coord.x, coord.x + 1) * tile_size`, likewise for `z` with `coord.y`.
#[derive(Component, Clone, Copy, Debug, Reflect)]
#[reflect(Component)]
pub struct VqTerrainTile {
    pub coord: IVec2,
    /// Vertical extent of the tile's bounding box.
    pub min_height: f32,
    pub max_height: f32,
}

#[derive(Resource, Default)]
struct LoadedTiles(HashMap<IVec2, Entity>);

// --- Material ----------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct TerrainParams {
    pub map_freqs: Vec4,
    pub map_amps: Vec4,
    pub rock_large: Vec4,
    pub rock_medium: Vec4,
    pub rock_small: Vec4,
    pub tile_min: Vec4,
    pub tile_max: Vec4,
    pub world_size: f32,
    pub height_max: f32,
    pub sea_level: f32,
    pub hm_res: f32,
    pub octave_shear: f32,
    pub bump_depth: f32,
    pub voro_res: f32,
    pub grass_flatness: f32,
    pub map_rect: Vec4,
    pub map_info: Vec4,
}

fn rock_uniform(r: &RockLayer) -> Vec4 {
    Vec4::new(
        1.0 / (r.cell_size * VORO_CELLS as f32),
        r.depth,
        r.sharpness,
        r.fade_distance,
    )
}

impl TerrainParams {
    pub fn new(field: &TerrainField) -> Self {
        let s = &field.settings;
        let m = &field.map;
        Self {
            map_rect: Vec4::new(m.origin.x, m.origin.y, m.extent.x, m.extent.y),
            map_info: Vec4::new(
                m.size.x as f32,
                m.size.y as f32,
                s.base_height,
                if m.custom { 1.0 } else { 0.0 },
            ),
            map_freqs: s.map_freqs,
            map_amps: s.map_amps,
            rock_large: rock_uniform(&s.rocks_large),
            rock_medium: rock_uniform(&s.rocks_medium),
            rock_small: rock_uniform(&s.rocks_small),
            tile_min: Vec4::ZERO,
            tile_max: Vec4::ZERO,
            world_size: s.world_size,
            height_max: s.height_max,
            sea_level: s.sea_level,
            hm_res: s.heightmap_resolution as f32,
            octave_shear: s.octave_shear,
            bump_depth: s.bump_depth,
            voro_res: VORO_RES as f32,
            grass_flatness: s.grass_flatness,
        }
    }
}

/// Ray-marches one terrain tile inside its bounding box.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VqTerrainMaterial {
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub palette: Handle<Image>,
    #[uniform(2)]
    pub shading: VqShadingUniform,
    #[uniform(3)]
    pub params: TerrainParams,
    #[storage(4, read_only)]
    pub heightmap: Handle<ShaderBuffer>,
    #[texture(5, dimension = "3d")]
    #[sampler(6)]
    pub voro: Handle<Image>,
    #[storage(7, read_only)]
    pub albedo: Handle<ShaderBuffer>,
}

impl Material for VqTerrainMaterial {
    // Shadow maps are drawn from a cheap proxy mesh instead (see
    // `VqShadowProxyMaterial`): ray-marching every shadow-map texel cost more
    // than the rest of the frame put together.
    fn enable_shadows() -> bool {
        false
    }

    fn fragment_shader() -> ShaderRef {
        shader_path("terrain.wgsl")
    }

    fn prepass_fragment_shader() -> ShaderRef {
        shader_path("terrain.wgsl")
    }

    // `Mask` forces the prepass/shadow pipelines to run the fragment shader,
    // which is where the depth of the ray-marched surface is computed.
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

impl VqShaded for VqTerrainMaterial {
    fn shading_mut(&mut self) -> &mut VqShadingUniform {
        &mut self.shading
    }
}

/// Invisible stand-in that casts each terrain tile's shadows: a heightfield
/// mesh rasterised into the shadow maps only. It never draws in the main pass
/// (the fragment shader discards) and is absent from camera prepasses.
#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct VqShadowProxyMaterial {}

impl Material for VqShadowProxyMaterial {
    fn fragment_shader() -> ShaderRef {
        shader_path("shadow_proxy.wgsl")
    }

    fn enable_prepass() -> bool {
        false
    }
}

#[derive(Resource)]
struct ShadowProxyMaterial(Handle<VqShadowProxyMaterial>);

/// Heightfield mesh over one tile, in tile-local XZ (origin at the tile's
/// centre) and world-space Y, lowered by `bias` so the caster never rises
/// above the ray-marched surface between samples (no shadow acne).
fn shadow_proxy_mesh(samples: &TileSamples, tile: f32, bias: f32) -> Mesh {
    let n = samples.n;
    let step = tile / (n - 1) as f32;
    let mut positions = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            positions.push([
                i as f32 * step - tile * 0.5,
                samples.heights[j * n + i] - bias,
                j as f32 * step - tile * 0.5,
            ]);
        }
    }
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    Mesh::new(
        bevy::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0f32, 1.0, 0.0]; n * n])
    .with_inserted_indices(bevy::mesh::Indices::U32(indices))
}

// --- Tile streaming ------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn stream_tiles(
    mut commands: Commands,
    terrain: Option<Res<VqTerrain>>,
    palette: Res<VqPalette>,
    shading: Res<VqShading>,
    focus: Query<&GlobalTransform, With<VqTerrainFocus>>,
    mut loaded: ResMut<LoadedTiles>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VqTerrainMaterial>>,
    proxy_material: Res<ShadowProxyMaterial>,
) {
    // No terrain (any more): clear the tiles. A new terrain: start over.
    let Some(terrain) = terrain else {
        for (_, e) in loaded.0.drain() {
            commands.entity(e).despawn();
        }
        return;
    };
    if terrain.is_changed() {
        for (_, e) in loaded.0.drain() {
            commands.entity(e).despawn();
        }
    }
    let field = &terrain.field;
    let s = &field.settings;
    let tile = s.tile_size;
    let center = focus
        .iter()
        .next()
        .map(|t| t.translation())
        .unwrap_or_default();
    let center_tile = IVec2::new(
        (center.x / tile).floor() as i32,
        (center.z / tile).floor() as i32,
    );

    let r = s.view_radius_tiles;
    let mut wanted = HashSet::new();
    for dz in -r..=r {
        for dx in -r..=r {
            if dx * dx + dz * dz <= r * r + r {
                wanted.insert(center_tile + IVec2::new(dx, dz));
            }
        }
    }

    loaded.0.retain(|coord, entity| {
        let keep = wanted.contains(coord);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });

    let shading = VqShadingUniform::new(&shading, &palette);
    let new: Vec<IVec2> = wanted
        .into_iter()
        .filter(|c| !loaded.0.contains_key(c))
        .collect();
    let samples = parallel_map(&new, |c| field.tile_samples(*c, TILE_SAMPLES));
    for (coord, samples) in new.into_iter().zip(samples) {
        let (min_h, max_h) = (samples.min_height, samples.max_height);
        let min = Vec3::new(coord.x as f32 * tile, min_h, coord.y as f32 * tile);
        let max = Vec3::new(min.x + tile, max_h, min.z + tile);

        let mut params = TerrainParams::new(field);
        params.tile_min = min.extend(s.max_steps as f32);
        params.tile_max = max.extend(0.0);
        let material = materials.add(VqTerrainMaterial {
            palette: palette.image.clone(),
            shading,
            params,
            heightmap: terrain.heightmap.clone(),
            voro: terrain.voro.clone(),
            albedo: terrain.albedo.clone(),
        });
        let size = max - min;
        let entity = commands
            .spawn((
                Name::new(format!("VQ terrain tile {coord}")),
                VqTerrainTile {
                    coord,
                    min_height: min_h,
                    max_height: max_h,
                },
                Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
                MeshMaterial3d(material),
                Transform::from_translation((min + max) * 0.5),
            ))
            .with_child((
                Name::new("VQ terrain shadow proxy"),
                Mesh3d(meshes.add(shadow_proxy_mesh(&samples, tile, s.shadow_proxy_bias))),
                MeshMaterial3d(proxy_material.0.clone()),
                // The proxy mesh is in tile-local XZ with world heights.
                Transform::from_translation(Vec3::new(0.0, -(min.y + max.y) * 0.5, 0.0)),
                NotShadowReceiver,
            ))
            .id();
        loaded.0.insert(coord, entity);
    }
}

// --- CPU terrain field ---------------------------------------------------------

/// CPU mirror of the terrain distance field in `terrain_sdf.wgsl`.
///
/// All public methods take Bevy (Y-up) coordinates and evaluate the field at
/// full detail (no distance fade), which is what physics and gameplay want.
pub struct TerrainField {
    /// The settings the field was built with. For a custom heightmap the
    /// height range, sea level and octave settings are filled in from it.
    pub settings: VqWorldSettings,
    /// `(height 0..1, mesa cap or rockiness 0..1)` per texel, row-major in
    /// VQ order (row = VQ y).
    pub heightmap: Vec<Vec2>,
    /// Packed sRGB albedo per texel (custom maps with colours), else empty.
    pub albedo: Vec<u32>,
    /// Where the heightmap lies and how it is sampled.
    pub map: MapLayout,
    /// Voronoi centreness, `voro_res³`, tiling.
    pub voro: Vec<u8>,
    pub voro_res: usize,
}

/// Placement of the heightmap in VQ space.
#[derive(Clone, Copy, Debug)]
pub struct MapLayout {
    /// VQ-space corner of texel (0, 0) and the size the map covers.
    pub origin: Vec2,
    pub extent: Vec2,
    /// Texels across and down.
    pub size: UVec2,
    /// Custom maps clamp at the edges (no tiling), have no mesa cap and
    /// carry rockiness in the second channel.
    pub custom: bool,
}

impl TerrainField {
    /// Runs the full generation (heightmap, Voronoi volume). With
    /// [`HeightmapSource::Manual`] this generates the procedural terrain.
    pub fn generate(settings: &VqWorldSettings) -> Self {
        if let HeightmapSource::Custom(map) = &settings.heightmap_source {
            return Self::from_custom(settings, map);
        }
        let res = settings.heightmap_resolution.max(16) as usize;
        let heights = match &settings.heightmap_source {
            HeightmapSource::VoxelQuestBmp { hm0, hm1 } => match (read_bmp(hm0), read_bmp(hm1)) {
                (Ok(a), Ok(b)) => mix_heightmap(res, settings.seed, Some([a, b])),
                (a, b) => {
                    let err = a.err().or(b.err()).unwrap_or_default();
                    warn!(
                        "bevy_voxelquest: can't read VQ heightmaps ({err}); using procedural terrain"
                    );
                    mix_heightmap(res, settings.seed, None)
                }
            },
            _ => mix_heightmap(res, settings.seed, None),
        };
        let caps = voronoi_map(res as u32, 12, settings.seed ^ 0x51ed);
        let heightmap = heights
            .into_iter()
            .zip(caps)
            .map(|(h, c)| Vec2::new(h, c))
            .collect();
        Self {
            settings: settings.clone(),
            heightmap,
            albedo: Vec::new(),
            map: MapLayout {
                origin: Vec2::ZERO,
                extent: Vec2::splat(settings.world_size),
                size: UVec2::splat(res as u32),
                custom: false,
            },
            voro: voronoi_volume(VORO_RES, VORO_CELLS, settings.seed ^ 0xa11c),
            voro_res: VORO_RES as usize,
        }
    }

    /// Builds a field from an app-supplied heightmap. The sea is at y = 0.
    pub fn from_custom(settings: &VqWorldSettings, map: &CustomHeightmap) -> Self {
        let (w, h) = (map.width.max(2), map.height.max(2));
        assert_eq!(map.heights.len(), w * h, "CustomHeightmap: heights must be width × height");
        let (lo, hi) = map
            .heights
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let range = (hi - lo).max(1.0);

        let mut s = settings.clone();
        s.base_height = lo;
        s.height_max = range;
        s.sea_level = (0.0 - lo) / range;
        // One octave of the real map: the higher octaves and VQ's bumps
        // would sample unrelated parts of it. Detail comes from the rocks.
        s.map_amps = Vec4::new(1.0, 0.0, 0.0, 0.0);
        s.octave_shear = 0.0;
        s.bump_depth = 0.0;
        s.world_size = (w.max(h) as f32) * map.cell_size;
        s.heightmap_resolution = w.max(h) as u32;

        // VQ y is -Z, so VQ row r is the map's row h - 1 - r.
        let mut heightmap = Vec::with_capacity(w * h);
        let mut albedo = Vec::new();
        for r in 0..h {
            let row = h - 1 - r;
            for x in 0..w {
                let i = row * w + x;
                let rock = map.rockiness.get(i).copied().unwrap_or(1.0);
                heightmap.push(Vec2::new((map.heights[i] - lo) / range, rock.clamp(0.0, 1.0)));
                if let Some(c) = map.colors.get(i) {
                    albedo.push(u32::from_le_bytes([c[0], c[1], c[2], 255]));
                }
            }
        }
        if albedo.len() != w * h {
            albedo.clear();
        }
        let extent = Vec2::new(w as f32, h as f32) * map.cell_size;
        Self {
            map: MapLayout {
                origin: Vec2::new(map.origin.x, -(map.origin.y + extent.y)),
                extent,
                size: UVec2::new(w as u32, h as u32),
                custom: true,
            },
            settings: s,
            heightmap,
            albedo,
            voro: voronoi_volume(VORO_RES, VORO_CELLS, settings.seed ^ 0xa11c),
            voro_res: VORO_RES as usize,
        }
    }

    fn texel_index(&self, x: i32, y: i32) -> usize {
        let (w, h) = (self.map.size.x as i32, self.map.size.y as i32);
        let (x, y) = if self.map.custom {
            (x.clamp(0, w - 1), y.clamp(0, h - 1))
        } else {
            (x.rem_euclid(w), y.rem_euclid(h))
        };
        (x + y * w) as usize
    }

    fn hm_texel(&self, x: i32, y: i32) -> Vec2 {
        self.heightmap[self.texel_index(x, y)]
    }

    fn map_uv(&self, xy: Vec2) -> Vec2 {
        (xy - self.map.origin) / self.map.extent
    }

    fn hm_bilin(&self, uv: Vec2) -> Vec2 {
        let c = uv * self.map.size.as_vec2() - 0.5;
        let i = c.floor();
        let f = c - i;
        let (x, y) = (i.x as i32, i.y as i32);
        let a = self.hm_texel(x, y);
        let b = self.hm_texel(x + 1, y);
        let c2 = self.hm_texel(x, y + 1);
        let d = self.hm_texel(x + 1, y + 1);
        a.lerp(b, f.x).lerp(c2.lerp(d, f.x), f.y)
    }

    fn voro_texel(&self, x: i32, y: i32, z: i32) -> f32 {
        let n = self.voro_res as i32;
        self.voro[(x.rem_euclid(n) + y.rem_euclid(n) * n + z.rem_euclid(n) * n * n) as usize] as f32
            / 255.0
    }

    fn voro_sample(&self, uvw: Vec3) -> f32 {
        let c = uvw * self.voro_res as f32 - 0.5;
        let i = c.floor();
        let f = c - i;
        let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
        let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let v = |dx, dy, dz| self.voro_texel(x + dx, y + dy, z + dz);
        l(
            l(
                l(v(0, 0, 0), v(1, 0, 0), f.x),
                l(v(0, 1, 0), v(1, 1, 0), f.x),
                f.y,
            ),
            l(
                l(v(0, 0, 1), v(1, 0, 1), f.x),
                l(v(0, 1, 1), v(1, 1, 1), f.x),
                f.y,
            ),
            f.z,
        )
    }

    /// VQ `getTerHeight` in VQ space: (vertical distance, height 0..1).
    fn ter_height(&self, p: Vec3, h0: Vec2) -> Vec2 {
        let s = &self.settings;
        let xy = Vec2::new(p.x, p.y);
        let tc = self.map_uv(xy);
        let tc2 = self.map_uv(xy + p.z * s.octave_shear);
        let hm = Vec4::new(
            h0.x,
            self.hm_bilin(tc2 * s.map_freqs.y).x,
            self.hm_bilin(tc2 * s.map_freqs.z).x,
            self.hm_bilin(tc2 * s.map_freqs.w).x,
        );
        let cap = if self.map.custom {
            1.0
        } else {
            let v2 = self.hm_bilin(tc * 8.0).y;
            ((0.5 + (0.95 - 0.5) * h0.y) + v2 * 0.05).clamp(0.0, 1.0)
        };
        let d = hm.dot(s.map_amps).min(cap);
        Vec2::new(p.z - (s.base_height + d * s.height_max), d)
    }

    /// Terrain distance in VQ space at full detail (`ter_val` in WGSL).
    pub fn distance_vq(&self, p: Vec3) -> f32 {
        let s = &self.settings;
        let xy = Vec2::new(p.x, p.y);
        let h0 = self.hm_bilin(self.map_uv(xy) * s.map_freqs.x);
        let mut res = self.ter_height(p, h0).x;
        if s.bump_depth > 0.0 {
            res += self.hm_bilin(self.map_uv(xy) * 32.0 + 0.74).x * s.bump_depth;
        }
        let rock = if self.map.custom { h0.y } else { 1.0 };

        let rl = rock_uniform(&s.rocks_large);
        let patch =
            ((p.x * rl.x * 6.0).sin() * (p.y * rl.x * 6.0).sin() * (p.z * rl.x * 6.0).sin()).abs();
        let patchy = 0.35 + 0.65 * patch.sqrt();
        let v = self.voro_sample(p * Vec3::new(rl.x, rl.x, rl.x * 0.5));
        res += (1.0 - v).powf(rl.z).clamp(0.0, 1.0) * rl.y * patchy * rock;

        let rm = rock_uniform(&s.rocks_medium);
        res += (1.0 - self.voro_sample(p * rm.x + 0.37)).powf(rm.z) * rm.y * rock;
        let rs = rock_uniform(&s.rocks_small);
        res += (1.0 - self.voro_sample(p * rs.x + 0.71)).powf(rs.z) * rs.y * rock;
        res
    }

    /// Signed (approximately vertical) distance to the terrain surface.
    /// Negative inside the ground.
    pub fn distance(&self, p: Vec3) -> f32 {
        self.distance_vq(to_vq(p))
    }

    /// Height of the topmost terrain surface at `(x, z)`.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let mut p = to_vq(Vec3::new(x, 0.0, z));
        // Start above the highest possible terrain and walk down; the field's
        // vertical slope is at most ~2.5 (the octave shear tilts it), so steps
        // of 0.4·d never skip a surface.
        p.z = self.settings.base_height + self.settings.height_max + self.settings.bump_depth + 1.0;
        let mut last_outside = p.z;
        for _ in 0..256 {
            let d = self.distance_vq(p);
            if d < 0.0 {
                break;
            }
            if d < 1.0e-3 {
                return p.z;
            }
            last_outside = p.z;
            p.z -= (d * 0.4).max(1.0e-3);
        }
        // Bisect between the last point above ground and the first below.
        let (mut lo, mut hi) = (p.z, last_outside);
        for _ in 0..24 {
            let mid = (lo + hi) * 0.5;
            if self.distance_vq(Vec3::new(p.x, p.y, mid)) < 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    }

    /// Surface normal (Bevy space) at a point near the surface.
    pub fn normal_at(&self, p: Vec3) -> Vec3 {
        let e = 0.05;
        let q = to_vq(p);
        let d = |o: Vec3| self.distance_vq(q + o);
        let n = Vec3::new(
            d(Vec3::X * e) - d(-Vec3::X * e),
            d(Vec3::Y * e) - d(-Vec3::Y * e),
            d(Vec3::Z * e) - d(-Vec3::Z * e),
        );
        from_vq(n).normalize_or(Vec3::Y)
    }

    /// Height range covered by a tile (with margins).
    pub fn tile_height_range(&self, coord: IVec2) -> (f32, f32) {
        let samples = self.tile_samples(coord, TILE_SAMPLES);
        (samples.min_height, samples.max_height)
    }

    /// Samples the topmost surface on an `n × n` grid over a tile
    /// (`heights[z * n + x]`) and derives the tile's vertical bounds.
    pub fn tile_samples(&self, coord: IVec2, n: usize) -> TileSamples {
        let s = &self.settings;
        let n = n.max(2);
        let step = s.tile_size / (n - 1) as f32;
        let (x0, z0) = (coord.x as f32 * s.tile_size, coord.y as f32 * s.tile_size);
        let mut heights = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                heights.push(self.height_at(x0 + i as f32 * step, z0 + j as f32 * step));
            }
        }
        let (lo, hi) = heights
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
        // Sampling can miss peaks and overhangs, and distant tiles render with
        // less rock detail (a higher surface), hence the margins.
        let rock_depth = s.rocks_large.depth + s.rocks_medium.depth + s.rocks_small.depth;
        let margin = s.height_max * 0.04 + 2.0;
        TileSamples {
            n,
            heights,
            min_height: lo - margin - rock_depth,
            max_height: hi + margin + rock_depth,
        }
    }
}

/// Surface heights sampled over one tile.
#[derive(Clone, Debug, Default)]
pub struct TileSamples {
    pub n: usize,
    /// `heights[z * n + x]`, world-space heights.
    pub heights: Vec<f32>,
    /// Vertical bounds of everything the tile can render (with margins).
    pub min_height: f32,
    pub max_height: f32,
}

/// Grid used for tile bounds and the shadow proxy mesh.
const TILE_SAMPLES: usize = 33;

/// Runs `f` over `items` on all cores.
pub(crate) fn parallel_map<T: Sync, R: Send + Default + Clone>(
    items: &[T],
    f: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = items.len().div_ceil(threads).max(1);
    let mut out = vec![R::default(); items.len()];
    std::thread::scope(|scope| {
        for (src, dst) in items.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let f = &f;
            scope.spawn(move || {
                for (s, d) in src.iter().zip(dst) {
                    *d = f(s);
                }
            });
        }
    });
    out
}

/// Port of `GameWorld::initMap` + `TerrainMix.c`: three source heightmaps
/// are masked by low-frequency simplex noise and the maximum is kept, then
/// normalised to 0..1.
fn mix_heightmap(res: usize, seed: u32, bmp: Option<[Bmp; 2]>) -> Vec<f32> {
    let rand = |i: u32| crate::noise::hash3([i as i32, 3, 9], seed)[0];
    // VQ picks three of its six heightmap channels at random.
    let mut channels = [0usize, 1, 2, 3, 4, 5];
    for i in 0..30u32 {
        let a = (rand(i * 2) * 6.0) as usize % 6;
        let b = (rand(i * 2 + 1) * 6.0) as usize % 6;
        channels.swap(a, b);
    }
    let offsets: Vec<Vec2> = (0..3)
        .map(|i| Vec2::new(rand(100 + i), rand(200 + i)))
        .collect();
    let time = rand(300) * 100.0;

    let rows: Vec<usize> = (0..res).collect();
    let raw: Vec<Vec<f32>> = parallel_map(&rows, |&y| {
        (0..res)
            .map(|x| {
                let uv = Vec2::new((x as f32 + 0.5) / res as f32, (y as f32 + 0.5) / res as f32);
                // Simplex2D.c: tiling 2-octave noise in three channels.
                let m = Vec3::new(
                    tiled_noise(uv.x, uv.y, time, 1.0, calc_noise),
                    tiled_noise(uv.x, uv.y, time + 37.0, 1.0, calc_noise),
                    tiled_noise(uv.x, uv.y, time + 58.0, 1.0, calc_noise),
                )
                .normalize_or_zero()
                    * 0.5
                    + 0.5;
                let sv = |i: usize| -> f32 {
                    let uv = (uv + offsets[i]).fract();
                    let ch = channels[i];
                    match &bmp {
                        Some(maps) => maps[ch / 3].sample(uv, ch % 3),
                        None => ridged_fbm(uv, 4.0, 6, ch as f32 * 13.7 + seed as f32 * 0.37),
                    }
                };
                (sv(0) * m.x)
                    .max(sv(1) * m.y)
                    .max(sv(2) * m.z)
                    .clamp(0.0, 1.0)
            })
            .collect()
    });
    let mut heights: Vec<f32> = raw.into_iter().flatten().collect();
    let (lo, hi) = heights
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
    let range = (hi - lo).max(1.0e-6);
    for h in &mut heights {
        *h = (*h - lo) / range;
    }
    heights
}

/// A decoded 24-bit BMP.
#[derive(Clone, Default)]
struct Bmp {
    width: usize,
    height: usize,
    /// RGB, top row first.
    rgb: Vec<[u8; 3]>,
}

impl Bmp {
    fn sample(&self, uv: Vec2, channel: usize) -> f32 {
        let x = ((uv.x * self.width as f32) as usize).min(self.width - 1);
        let y = ((uv.y * self.height as f32) as usize).min(self.height - 1);
        self.rgb[x + y * self.width][channel] as f32 / 255.0
    }
}

fn read_bmp(path: &str) -> Result<Bmp, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    if bytes.len() < 54 || &bytes[0..2] != b"BM" {
        return Err(format!("{path}: not a BMP"));
    }
    let offset = u32_at(10) as usize;
    let width = u32_at(18) as usize;
    let raw_height = u32_at(22) as i32;
    let bpp = u16::from_le_bytes([bytes[28], bytes[29]]);
    if bpp != 24 {
        return Err(format!("{path}: expected 24-bit BMP, got {bpp}-bit"));
    }
    let height = raw_height.unsigned_abs() as usize;
    let stride = (width * 3).div_ceil(4) * 4;
    if bytes.len() < offset + stride * height {
        return Err(format!("{path}: truncated"));
    }
    let mut rgb = Vec::with_capacity(width * height);
    for row in 0..height {
        // Positive height = bottom-up rows.
        let src = if raw_height > 0 {
            height - 1 - row
        } else {
            row
        };
        let line = &bytes[offset + src * stride..];
        for x in 0..width {
            let b = &line[x * 3..x * 3 + 3];
            rgb.push([b[2], b[1], b[0]]);
        }
    }
    Ok(Bmp { width, height, rgb })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_field() -> TerrainField {
        TerrainField::generate(&VqWorldSettings {
            heightmap_resolution: 128,
            world_size: 1024.0,
            tile_size: 256.0,
            ..default()
        })
    }

    #[test]
    fn height_at_lies_on_the_surface() {
        let field = small_field();
        for (x, z) in [(0.0, 0.0), (130.5, -77.0), (900.0, 400.0), (-512.0, 1300.0)] {
            let h = field.height_at(x, z);
            let d = field.distance(Vec3::new(x, h, z));
            assert!(d.abs() < 0.05, "distance {d} at surface ({x}, {h}, {z})");
            assert!(field.distance(Vec3::new(x, h + 5.0, z)) > 0.0);
        }
    }

    #[test]
    fn custom_heightmap_is_placed_in_world_units() {
        // 8×8 texels of 10 m from (-40, -40): flat at -5 m (sea floor),
        // with one 100 m column at texel (6, 1), i.e. x 20..30, z -30..-20.
        let (w, h, c) = (8, 8, 10.0);
        let mut heights = vec![-5.0; w * h];
        heights[1 * w + 6] = 100.0;
        let map = CustomHeightmap {
            width: w,
            height: h,
            heights,
            rockiness: vec![0.0; w * h], // no rocks: heights exactly as given
            colors: Vec::new(),
            origin: Vec2::new(-40.0, -40.0),
            cell_size: c,
        };
        let field = TerrainField::from_custom(&VqWorldSettings::default(), &map);
        // Texel centres are exact (bilinear filtering reproduces them).
        let peak = field.height_at(25.0, -25.0);
        assert!((peak - 100.0).abs() < 0.01, "peak {peak}");
        let flat = field.height_at(-25.0, 25.0);
        assert!((flat + 5.0).abs() < 0.01, "flat {flat}");
        // Mirrored position (wrong Z flip) must be low.
        assert!(field.height_at(25.0, 25.0) < 0.0);
        // Sea at y = 0, and no tiling: beyond the map the edge is clamped.
        assert!(field.settings.sea_height().abs() < 1.0e-3);
        assert!((field.height_at(1000.0, 1000.0) + 5.0).abs() < 0.01);
    }

    #[test]
    fn tile_ranges_contain_surface() {
        let field = small_field();
        let (lo, hi) = field.tile_height_range(IVec2::new(1, 2));
        for i in 0..8 {
            let x = 256.0 + i as f32 * 31.0;
            let z = 512.0 + i as f32 * 29.0;
            let h = field.height_at(x, z);
            assert!(h > lo && h < hi, "{h} outside {lo}..{hi}");
        }
    }
}
