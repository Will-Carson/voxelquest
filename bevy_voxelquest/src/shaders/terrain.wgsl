// Ray-marched Voxel Quest terrain tile (PrimShader.c, DOTER pass).
//
// Rasterises the back faces of the tile's bounding box, marches the terrain
// SDF inside the box and writes real depth, so the same shader serves the
// main pass and the camera's depth/normal/motion prepass. Shadow maps are
// drawn from a cheap proxy mesh instead (terrain.rs), with fine
// self-shadowing added here by `contact_shadow`.

#import bevy_voxelquest::common::{
    view_ray, ray_box, to_vq, from_vq, Surface, VqFragmentOutput, is_orthographic, depth_at,
}
#import bevy_voxelquest::terrain_sdf::{terrain, ter_val, ter_dist, ter_normal, ter_march_dist}
#import bevy_voxelquest::noise::hash_vec
#import bevy_pbr::mesh_view_bindings::view

#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::VertexOutput
#import bevy_voxelquest::common::prepass_output
#else
#import bevy_pbr::forward_io::VertexOutput
#import bevy_pbr::mesh_view_bindings::lights
#import bevy_voxelquest::common::{shade, shading}
#ifdef DEPTH_PREPASS
#import bevy_voxelquest::common::prepass_surface
#endif
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
    let cam = to_vq(view.world_position);

    let max_steps = i32(terrain.tile_min.w);
    var prev_t = t;
    var hit = false;
    var cam_dist = 0.0;
    for (var i = 0; i < max_steps; i++) {
        let p = o + d * t;
        cam_dist = distance(p, cam);
        let dist = ter_march_dist(p, cam_dist);
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

#ifndef PREPASS_PIPELINE
// Short soft-shadow march towards the light (VQ's `softShadow`) for the
// small-scale self-shadowing of rocks and cracks that the coarse shadow
// proxy can't resolve. Large-scale shadows come from the shadow maps.
fn contact_shadow(p: vec3<f32>, n: vec3<f32>, cam_dist: f32) -> f32 {
    var to_light = normalize(vec3(shading.fallback_light_x, shading.fallback_light_y, shading.fallback_light_z));
    if lights.n_directional_lights > 0u {
        to_light = lights.directional_lights[0].direction_to_light;
    }
    let l = to_vq(to_light);
    if dot(n, l) <= 0.0 {
        return 1.0; // facing away; the lighting term is already zero
    }
    let origin = p + n * (0.05 + cam_dist * 0.001);
    var res = 1.0;
    var t = 0.2;
    for (var i = 0; i < 16; i++) {
        let h = ter_dist(origin + l * t, cam_dist);
        res = min(res, 6.0 * h / t);
        t += clamp(h, 0.25, 4.0);
        if res < 0.02 || t > 40.0 {
            break;
        }
    }
    return clamp(res, 0.0, 1.0);
}
#endif

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

#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
// Main pass with a depth prepass: shade the prepass hit if it is on this
// tile's terrain, otherwise something else is in front, so discard.
fn prepass_hit(frag_coord: vec4<f32>) -> Hit {
    let s = prepass_surface(frag_coord);
    let w = s.xyz;
    let lo = terrain.tile_min.xyz - 0.01;
    let hi = terrain.tile_max.xyz + 0.01;
    if s.w <= 0.0 || any(w < lo) || any(w > hi) {
        discard;
    }
    let cam_dist = distance(w, view.world_position);
    let p = to_vq(w);
    if abs(ter_dist(p, cam_dist)) > 0.05 + cam_dist * 0.004 {
        discard;
    }
    var out: Hit;
    out.pos = p;
    out.cam_dist = cam_dist;
    return out;
}
#endif
#endif

@fragment
fn fragment(in: VertexOutput) -> VqFragmentOutput {
#ifdef PREPASS_PIPELINE
    let hit = march(in.world_position.xyz);
#else
#ifdef DEPTH_PREPASS
    let hit = prepass_hit(in.position);
#else
    let hit = march(in.world_position.xyz);
#endif
#endif
    let world = from_vq(hit.pos);

#ifdef PREPASS_PIPELINE
#ifdef NORMAL_PREPASS
    let normal = normalize(from_vq(ter_normal(hit.pos, hit.cam_dist)));
#else
    // Depth-only prepass: no normal needed.
    let normal = vec3(0.0, 1.0, 0.0);
#endif
    // Terrain is static: last frame it was in the same place.
    return prepass_output(world, world, normal);
#else
    let n_vq = ter_normal(hit.pos, hit.cam_dist);
    let normal = normalize(from_vq(n_vq));
    var mat: u32;
    var variation: f32;
    classify(hit.pos, n_vq, hit.cam_dist, &mat, &variation);

    var s: Surface;
    s.world_position = world;
    s.normal = normal;
    s.mat = mat;
    s.variation = variation;
    s.ao = terrain_ao(hit.pos, n_vq, hit.cam_dist);
    s.contact_shadow = contact_shadow(hit.pos, n_vq, hit.cam_dist);
    s.specular = select(0.0, 0.5, mat == MAT_SNOW);

    var out: VqFragmentOutput;
    out.color = shade(s, in.position);
#ifdef DEPTH_PREPASS
    // Bit-identical to the prepass, so the GreaterEqual depth test passes.
    out.frag_depth = prepass_surface(in.position).w;
#else
    out.frag_depth = depth_at(world);
#endif
    return out;
#endif
}
