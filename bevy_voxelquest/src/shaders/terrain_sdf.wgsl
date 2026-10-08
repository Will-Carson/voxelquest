// Voxel Quest terrain distance field (TerHeightFunc.c + MapLand.c).
//
// Evaluated in VQ space (Z up). Mirrored on the CPU by `terrain::TerrainField`;
// keep the two in sync.
#define_import_path bevy_voxelquest::terrain_sdf

struct TerrainParams {
    map_freqs: vec4<f32>,
    map_amps: vec4<f32>,
    // (1 / texture period, depth, sharpness, fade distance)
    rock_large: vec4<f32>,
    rock_medium: vec4<f32>,
    rock_small: vec4<f32>,
    // Tile bounds in Bevy space; tile_min.w = max march steps.
    tile_min: vec4<f32>,
    tile_max: vec4<f32>,
    world_size: f32,
    height_max: f32,
    sea_level: f32,
    hm_res: f32,
    octave_shear: f32,
    bump_depth: f32,
    voro_res: f32,
    grass_flatness: f32,
    // VQ-space rectangle the heightmap covers: (origin.xy, extent.xy).
    map_rect: vec4<f32>,
    // (texels across, texels down, world height of a 0 sample, custom map flag).
    // A custom map clamps at its edges instead of tiling, has no mesa cap,
    // carries per-texel rockiness in `.y` and albedo in `albedo`.
    map_info: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> terrain: TerrainParams;
// (height 0..1, mesa cap 0..1) per texel, row-major, tiling.
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<storage, read> heightmap: array<vec2<f32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var voro_texture: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var voro_sampler: sampler;
// Custom maps only: sRGB albedo per texel, packed RGBA8 (alpha 0 = none).
@group(#{MATERIAL_BIND_GROUP}) @binding(7) var<storage, read> albedo: array<u32>;

fn is_custom() -> bool {
    return terrain.map_info.w > 0.5;
}

// World height of a heightmap sample of 0.
fn base_height() -> f32 {
    return terrain.map_info.z;
}

// Heightmap coordinates (in map periods) of a VQ-space position.
fn map_uv(xy: vec2<f32>) -> vec2<f32> {
    return (xy - terrain.map_rect.xy) / terrain.map_rect.zw;
}

fn texel_index(x: i32, y: i32) -> i32 {
    let w = i32(terrain.map_info.x);
    let h = i32(terrain.map_info.y);
    if is_custom() {
        return clamp(x, 0, w - 1) + clamp(y, 0, h - 1) * w;
    }
    return (((x % w) + w) % w) + (((y % h) + h) % h) * w;
}

fn hm_texel(x: i32, y: i32) -> vec2<f32> {
    return heightmap[texel_index(x, y)];
}

// VQ `bilin`: manual bilinear filtering (wrapping or clamped); uv in map periods.
fn hm_bilin(uv: vec2<f32>) -> vec2<f32> {
    let c = uv * terrain.map_info.xy - 0.5;
    let i = floor(c);
    let f = c - i;
    let x = i32(i.x);
    let y = i32(i.y);
    let a = hm_texel(x, y);
    let b = hm_texel(x + 1, y);
    let cc = hm_texel(x, y + 1);
    let d = hm_texel(x + 1, y + 1);
    return mix(mix(a, b, f.x), mix(cc, d, f.x), f.y);
}

fn albedo_texel(x: i32, y: i32) -> vec4<f32> {
    return unpack4x8unorm(albedo[texel_index(x, y)]);
}

// Bilinear per-texel albedo of a custom map (sRGB, alpha 0 = none).
fn albedo_at(xy: vec2<f32>) -> vec4<f32> {
    if !is_custom() {
        return vec4(0.0);
    }
    let c = map_uv(xy) * terrain.map_info.xy - 0.5;
    let i = floor(c);
    let f = c - i;
    let x = i32(i.x);
    let y = i32(i.y);
    return mix(
        mix(albedo_texel(x, y), albedo_texel(x + 1, y), f.x),
        mix(albedo_texel(x, y + 1), albedo_texel(x + 1, y + 1), f.x),
        f.y,
    );
}

// Mesa cap from VQ's voro map; custom maps are uncapped.
fn cap_from(h0y: f32, v2: f32) -> f32 {
    if is_custom() {
        return 1.0;
    }
    return clamp(mix(0.5, 0.95, h0y) + v2 * 0.05, 0.0, 1.0);
}

// Upper bound of the cap from the first sample alone.
fn cap_max_from(h0y: f32) -> f32 {
    if is_custom() {
        return 1.0;
    }
    return clamp(mix(0.5, 0.95, h0y) + 0.05, 0.0, 1.0);
}

// How strongly the rock layers apply: per texel on custom maps.
fn rockiness_from(h0y: f32) -> f32 {
    return select(1.0, h0y, is_custom());
}

// Voronoi centreness (1 at a cell centre, 0 at its border); tiling.
fn voro(uvw: vec3<f32>) -> f32 {
    return textureSampleLevel(voro_texture, voro_sampler, uvw, 0.0).r;
}

// VQ `getTerHeight`: returns (signed vertical distance, height 0..1).
// `h0` is the first octave sample, `hm_bilin(tc * map_freqs.x)`.
fn ter_height_from(p: vec3<f32>, h0: vec2<f32>) -> vec2<f32> {
    let tc = map_uv(p.xy);
    let tc2 = map_uv(p.xy + p.z * terrain.octave_shear);
    let hm = vec4(
        h0.x,
        hm_bilin(tc2 * terrain.map_freqs.y).x,
        hm_bilin(tc2 * terrain.map_freqs.z).x,
        hm_bilin(tc2 * terrain.map_freqs.w).x,
    );
    var dot_val = dot(hm, terrain.map_amps);
    var v2 = 0.0;
    if !is_custom() {
        v2 = hm_bilin(tc * 8.0).y;
    }
    dot_val = min(dot_val, cap_from(h0.y, v2));
    return vec2(p.z - (base_height() + dot_val * terrain.height_max), dot_val);
}

fn first_octave(p: vec3<f32>) -> vec2<f32> {
    return hm_bilin(map_uv(p.xy) * terrain.map_freqs.x);
}

fn ter_height(p: vec3<f32>) -> vec2<f32> {
    return ter_height_from(p, first_octave(p));
}

fn fade(cam_dist: f32, fade_distance: f32) -> f32 {
    return 1.0 - smoothstep(fade_distance * 0.5, fade_distance, cam_dist);
}

struct TerVal {
    dist: f32,
    height_frac: f32,
    // Fine bump sample, used for snow placement.
    bump: f32,
}

// VQ `getTerVal`: heightfield + bumps + three scales of Voronoi rocks.
// `cam_dist` fades fine detail out with distance (pass 0 for full detail).
fn ter_val(p: vec3<f32>, cam_dist: f32) -> TerVal {
    let h0 = first_octave(p);
    let th = ter_height_from(p, h0);
    var res = th.x;
    var bump = 0.0;
    if terrain.bump_depth > 0.0 {
        bump = hm_bilin(map_uv(p.xy) * 32.0 + 0.74).x;
        res += bump * terrain.bump_depth;
    }
    let rock = rockiness_from(h0.y);

    let rl = terrain.rock_large;
    let fl = fade(cam_dist, rl.w) * rock;
    if fl > 0.0 {
        let patch_v = abs(sin(p.x * rl.x * 6.0) * sin(p.y * rl.x * 6.0) * sin(p.z * rl.x * 6.0));
        let patchy = 0.35 + 0.65 * sqrt(patch_v);
        let v = voro(p * vec3(rl.x, rl.x, rl.x * 0.5));
        res += clamp(pow(1.0 - v, rl.z), 0.0, 1.0) * rl.y * fl * patchy;
    }
    let rm = terrain.rock_medium;
    let fm = fade(cam_dist, rm.w) * rock;
    if fm > 0.0 {
        res += pow(1.0 - voro(p * rm.x + 0.37), rm.z) * rm.y * fm;
    }
    let rs = terrain.rock_small;
    let fs = fade(cam_dist, rs.w) * rock;
    if fs > 0.0 {
        res += pow(1.0 - voro(p * rs.x + 0.71), rs.z) * rs.y * fs;
    }

    var out: TerVal;
    out.dist = res;
    out.height_frac = th.y;
    out.bump = bump;
    return out;
}

fn ter_dist(p: vec3<f32>, cam_dist: f32) -> f32 {
    return ter_val(p, cam_dist).dist;
}

// Most the bump and rock layers can add at this distance. They only ever
// push the surface *down* (increase the distance), so the bare heightfield
// is a lower bound of the full field that is at most this far off.
fn detail_band(cam_dist: f32) -> f32 {
    return terrain.bump_depth
        + terrain.rock_large.y * fade(cam_dist, terrain.rock_large.w)
        + terrain.rock_medium.y * fade(cam_dist, terrain.rock_medium.w)
        + terrain.rock_small.y * fade(cam_dist, terrain.rock_small.w)
        + 1.0;
}

// Distance for marching: exact near the surface, a cheap lower bound away
// from it. Never returns less than `ter_dist`, so it never reports a false
// hit, and stepping by it is as safe as stepping by the full field.
fn ter_march_dist(p: vec3<f32>, cam_dist: f32) -> f32 {
    let band = detail_band(cam_dist);
    // Tier 0: one bilinear lookup. The higher octaves add at most the sum of
    // their amplitudes, and the mesa cap is bounded by the first sample too.
    let h0 = first_octave(p);
    let a = terrain.map_amps;
    let upper = min(h0.x * a.x + a.y + a.z + a.w, cap_max_from(h0.y));
    let bound0 = p.z - (base_height() + upper * terrain.height_max);
    if bound0 > band {
        return bound0;
    }
    // Tier 1: the bare heightfield (no rocks).
    let base = ter_height_from(p, h0).x;
    if base > band {
        return base;
    }
    return ter_dist(p, cam_dist);
}

// Central-difference normal, returned in VQ space.
fn ter_normal(p: vec3<f32>, cam_dist: f32) -> vec3<f32> {
    let e = clamp(cam_dist * 0.002, 0.05, 4.0);
    let dx = ter_dist(p + vec3(e, 0.0, 0.0), cam_dist) - ter_dist(p - vec3(e, 0.0, 0.0), cam_dist);
    let dy = ter_dist(p + vec3(0.0, e, 0.0), cam_dist) - ter_dist(p - vec3(0.0, e, 0.0), cam_dist);
    let dz = ter_dist(p + vec3(0.0, 0.0, e), cam_dist) - ter_dist(p - vec3(0.0, 0.0, e), cam_dist);
    return normalize(vec3(dx, dy, dz));
}
