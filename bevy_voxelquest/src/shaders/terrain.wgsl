// Ray-marched Voxel Quest terrain tile (PrimShader.c, DOTER pass).
//
// Rasterises the back faces of the tile's bounding box, marches the terrain
// SDF inside the box and writes real depth, so the same shader serves the
// main pass, the depth/normal prepass and shadow maps.

#import bevy_voxelquest::common::{
    view_ray, ray_box, to_vq, from_vq, Surface, VqFragmentOutput, is_orthographic, depth_at,
}
#import bevy_voxelquest::terrain_sdf::{terrain, ter_val, ter_dist, ter_normal}
#import bevy_voxelquest::noise::hash_vec
#import bevy_pbr::mesh_view_bindings::view

#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::VertexOutput
#import bevy_voxelquest::common::prepass_output
#else
#import bevy_pbr::forward_io::VertexOutput
#import bevy_voxelquest::common::shade
#endif

const MAT_SAND: u32 = 3u;
const MAT_SNOW: u32 = 5u;
const MAT_GRASS: u32 = 6u;
const MAT_EARTH: u32 = 12u;

struct Hit {
    pos: vec3<f32>, // VQ space
    cam_dist: f32,
}

fn march(in_world: vec3<f32>) -> Hit {
    let ray = view_ray(in_world);
    let tb = ray_box(ray.origin, ray.dir, terrain.tile_min.xyz, terrain.tile_max.xyz);
    var t = max(tb.x, ray.t_min);
    let t_end = tb.y;
    if t > t_end {
        discard;
    }

    let o = to_vq(ray.origin);
    let d = to_vq(ray.dir);
    // Detail fades with distance from the *viewer*; shadow views use the
    // distance along their own ray, which keeps casters coarse but stable.
    let ortho = is_orthographic();
    let cam = to_vq(view.world_position);
    let max_steps = i32(terrain.tile_min.w);

    var prev_t = t;
    var hit = false;
    var cam_dist = 0.0;
    for (var i = 0; i < max_steps; i++) {
        let p = o + d * t;
        cam_dist = select(distance(p, cam), 0.0, ortho);
        let dist = ter_dist(p, cam_dist);
        let tol = 0.01 + cam_dist * 0.0015;
        if dist < tol {
            hit = true;
            break;
        }
        prev_t = t;
        // The heightfield "distance" is vertical, so relax the step.
        t += max(dist * 0.5, tol * 0.5);
        if t > t_end {
            break;
        }
    }
    if !hit {
        discard;
    }

    // Bisection between the last outside sample and the hit.
    var lo = prev_t;
    var hi = t;
    for (var i = 0; i < 6; i++) {
        let mid = (lo + hi) * 0.5;
        if ter_dist(o + d * mid, cam_dist) < 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    var out: Hit;
    out.pos = o + d * hi;
    out.cam_dist = cam_dist;
    return out;
}

fn randf3(p: vec3<f32>) -> f32 {
    return hash_vec(p * 113.17);
}

// Terrain material rules from PrimShader.c `castLand`, with distances scaled
// from VQ's 4096-cell height range to `terrain.height_max`.
fn classify(p: vec3<f32>, n: vec3<f32>, cam_dist: f32, out_mat: ptr<function, u32>, out_var: ptr<function, f32>) {
    let H = terrain.height_max;
    let k = H / 4096.0;
    let S = 512.0 * k;
    let sea = terrain.sea_level;
    let tv = ter_val(p, cam_dist);

    let cam01 = clamp(cam_dist * 4.0 / (terrain.world_size * 4.0), 0.0, 1.0);
    let speckle = randf3(floor(p * 32.0) / 32.0) * clamp(1.0 - cam01 * 4.0, 0.0, 1.0);
    let hv = clamp(1.0 - (H - p.z) / H, 0.0, 1.0) * 0.3
        + abs(sin(p.x / S) * sin(p.y / S) * sin(p.z / S)) * 0.01;
    let snow_source = pow(abs(tv.bump - 0.5) * 2.0, 8.0) * 0.05;
    var snow = hv + snow_source * 0.4 - cam01 * 0.02 - 0.15;
    snow += n.z * 0.05 * f32(hv > 0.1);
    let is_grass = snow - speckle * 0.01 - clamp(hv - 0.12, 0.0, 0.1) * 2.0;

    var mat = MAT_EARTH;
    var variation = clamp((sin(p.z / S) + 1.0) * 0.5, 0.0, 1.0);

    // Extra rule (not in VQ's final build): grass on flat ground above the shore.
    let flat_grass = terrain.grass_flatness > 0.0 && n.z > terrain.grass_flatness && snow < 0.0;
    if (is_grass > 0.001 || flat_grass) && (p.z - 100.0 * k) > (sea + speckle * 0.01) * H {
        mat = MAT_GRASS;
        variation = clamp(speckle + (1.0 - n.z) * 2.0, 0.0, 1.0);
    }
    if (p.z - 20.0 * k) < (sea + speckle * 0.005) * H && n.z > 0.5 {
        mat = MAT_SAND;
    }
    if snow > 0.04 {
        mat = MAT_SNOW;
        variation = clamp(snow * 4.0, 0.0, 1.0);
    }
    *out_mat = mat;
    *out_var = variation;
}

// SDF ambient occlusion: how much the terrain closes in along the normal.
fn terrain_ao(p: vec3<f32>, n: vec3<f32>, cam_dist: f32) -> f32 {
    var occ = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 5; i++) {
        let h = 0.6 * f32(i) * f32(i);
        let d = ter_dist(p + n * h, cam_dist);
        occ += (h - max(d, 0.0)) / h * w;
        w *= 0.6;
    }
    return clamp(1.0 - occ * 0.45, 0.0, 1.0);
}

@fragment
fn fragment(in: VertexOutput) -> VqFragmentOutput {
    let hit = march(in.world_position.xyz);
    let n_vq = ter_normal(hit.pos, hit.cam_dist);
    let world = from_vq(hit.pos);
    let normal = normalize(from_vq(n_vq));

#ifdef PREPASS_PIPELINE
    return prepass_output(world, normal);
#else
    var mat: u32;
    var variation: f32;
    classify(hit.pos, n_vq, hit.cam_dist, &mat, &variation);

    var s: Surface;
    s.world_position = world;
    s.normal = normal;
    s.mat = mat;
    s.variation = variation;
    s.ao = terrain_ao(hit.pos, n_vq, hit.cam_dist);
    s.specular = select(0.0, 0.5, mat == MAT_SNOW);

    var out: VqFragmentOutput;
    out.color = shade(s, in.position);
    out.frag_depth = depth_at(world);
    return out;
#endif
}
