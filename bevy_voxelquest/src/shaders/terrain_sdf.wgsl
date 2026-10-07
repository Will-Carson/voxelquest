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
}

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> terrain: TerrainParams;
// (height 0..1, mesa cap 0..1) per texel, row-major, tiling.
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<storage, read> heightmap: array<vec2<f32>>;
@group(#{MATERIAL_BIND_GROUP}) @binding(5) var voro_texture: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(6) var voro_sampler: sampler;

fn hm_texel(x: i32, y: i32) -> vec2<f32> {
    let n = i32(terrain.hm_res);
    let xi = ((x % n) + n) % n;
    let yi = ((y % n) + n) % n;
    return heightmap[xi + yi * n];
}

// VQ `bilin`: manual bilinear filtering with wrap; uv in texture periods.
fn hm_bilin(uv: vec2<f32>) -> vec2<f32> {
    let c = uv * terrain.hm_res - 0.5;
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

// Voronoi centreness (1 at a cell centre, 0 at its border); tiling.
fn voro(uvw: vec3<f32>) -> f32 {
    return textureSampleLevel(voro_texture, voro_sampler, uvw, 0.0).r;
}

// VQ `getTerHeight`: returns (signed vertical distance, height 0..1).
fn ter_height(p: vec3<f32>) -> vec2<f32> {
    let tc = p.xy / terrain.world_size;
    let tc2 = (p.xy + p.z * terrain.octave_shear) / terrain.world_size;
    let h0 = hm_bilin(tc * terrain.map_freqs.x);
    let hm = vec4(
        h0.x,
        hm_bilin(tc2 * terrain.map_freqs.y).x,
        hm_bilin(tc2 * terrain.map_freqs.z).x,
        hm_bilin(tc2 * terrain.map_freqs.w).x,
    );
    var dot_val = dot(hm, terrain.map_amps);
    let v2 = hm_bilin(tc * 8.0).y;
    let cap = clamp(mix(0.5, 0.95, h0.y) + v2 * 0.05, 0.0, 1.0);
    dot_val = min(dot_val, cap);
    return vec2(p.z - dot_val * terrain.height_max, dot_val);
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
    let th = ter_height(p);
    var res = th.x;
    let tc = p.xy / terrain.world_size;
    let bump = hm_bilin(tc * 32.0 + 0.74).x;
    res += bump * terrain.bump_depth;

    let rl = terrain.rock_large;
    let fl = fade(cam_dist, rl.w);
    if fl > 0.0 {
        let patch_v = abs(sin(p.x * rl.x * 6.0) * sin(p.y * rl.x * 6.0) * sin(p.z * rl.x * 6.0));
        let patchy = 0.35 + 0.65 * sqrt(patch_v);
        let v = voro(p * vec3(rl.x, rl.x, rl.x * 0.5));
        res += clamp(pow(1.0 - v, rl.z), 0.0, 1.0) * rl.y * fl * patchy;
    }
    let rm = terrain.rock_medium;
    let fm = fade(cam_dist, rm.w);
    if fm > 0.0 {
        res += pow(1.0 - voro(p * rm.x + 0.37), rm.z) * rm.y * fm;
    }
    let rs = terrain.rock_small;
    let fs = fade(cam_dist, rs.w);
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

// Central-difference normal, returned in VQ space.
fn ter_normal(p: vec3<f32>, cam_dist: f32) -> vec3<f32> {
    let e = clamp(cam_dist * 0.002, 0.05, 4.0);
    let dx = ter_dist(p + vec3(e, 0.0, 0.0), cam_dist) - ter_dist(p - vec3(e, 0.0, 0.0), cam_dist);
    let dy = ter_dist(p + vec3(0.0, e, 0.0), cam_dist) - ter_dist(p - vec3(0.0, e, 0.0), cam_dist);
    let dz = ter_dist(p + vec3(0.0, 0.0, e), cam_dist) - ter_dist(p - vec3(0.0, 0.0, e), cam_dist);
    return normalize(vec3(dx, dy, dz));
}
