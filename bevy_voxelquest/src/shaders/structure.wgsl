// Ray-marched Voxel Quest structure (PrimShader.c, DOPRIM pass).
//
// A structure is a list of primitive instances (hollow superellipsoid shells
// clipped by a visibility box). Walls that meet are merged by subtracting
// every primitive's interior from every shell, and the closest primitive gets
// VQ's procedural brick / plaster / timber / shingle surface detail.

#import bevy_voxelquest::common::{
    view_ray, ray_box, to_vq, from_vq, Surface, VqFragmentOutput, depth_at, BIG,
}
#import bevy_voxelquest::noise::{snoise_2d, hash_vec}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::prepass_io::VertexOutput
#import bevy_voxelquest::common::prepass_output
#else
#import bevy_pbr::forward_io::VertexOutput
#import bevy_voxelquest::common::shade
#ifdef DEPTH_PREPASS
#import bevy_voxelquest::common::prepass_surface
#endif
#endif

struct Prim {
    vis_center: vec4<f32>,
    vis_half: vec4<f32>,
    box_center: vec4<f32>,
    // xyz = inner box half extents, w = corner radius
    box_dim: vec4<f32>,
    // (power xy, power z, wall thickness, style)
    params: vec4<f32>,
}

struct StructureParams {
    world_from_local: mat4x4<f32>,
    local_from_world: mat4x4<f32>,
    // Last frame's transform, for motion vectors.
    previous_world_from_local: mat4x4<f32>,
    // Local-space bounds; box_min.w = prim count, box_max.w = max steps.
    box_min: vec4<f32>,
    box_max: vec4<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> structure: StructureParams;
@group(#{MATERIAL_BIND_GROUP}) @binding(4) var<storage, read> prims: array<Prim>;

const MAT_WOOD: f32 = 8.0;
const MAT_BRICK: f32 = 9.0;
const MAT_SHINGLE: f32 = 10.0;
const MAT_PLASTER: f32 = 11.0;
const MAX_CANDIDATES: u32 = 16u;
const PI: f32 = 3.14159265;

var<private> candidates: array<u32, 16>;
var<private> candidate_count: u32;

// --- SDF operators (PrimShader.c 428-567) -----------------------------------

fn fmod3(p: vec3<f32>, c: vec3<f32>) -> vec3<f32> { return p - c * floor(p / c); }
fn fmod1(p: f32, c: f32) -> f32 { return p - c * floor(p / c); }
fn op_rep(p: vec3<f32>, c: vec3<f32>) -> vec3<f32> { return fmod3(p, c) - 0.5 * c; }

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, vec3(0.0)));
}

fn sd_box_m(m: f32, p: vec3<f32>, b: vec3<f32>) -> vec2<f32> {
    return vec2(sd_box(p, b), m);
}

fn op_u(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return select(b, a, a.x < b.x); }
fn op_u3(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec2<f32> { return op_u(op_u(a, b), c); }

// Union that breaks ties by the lower `.z`.
fn op_u_tie(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    if a.x == b.x {
        return select(b, a, a.z < b.z);
    }
    return select(b, a, a.x < b.x);
}

// Superellipsoid rounded box (VQ `udRoundBox`):
// x = outer distance, y = negative inner distance, z = inner-cavity distance.
fn ud_round_box(v: vec3<f32>, dim: vec4<f32>, power: vec2<f32>, wall: f32) -> vec3<f32> {
    let n = max(abs(v) - dim.xyz, vec3(0.0)) + 1.0e-7;
    let a = pow(pow(n.x, power.x) + pow(n.y, power.x), 1.0 / power.x);
    let b = pow(pow(a, power.y) + pow(n.z, power.y), 1.0 / power.y);
    return vec3(b - dim.w, (dim.w - wall) - b, b - (dim.w - wall));
}

fn ud_min_box(v: vec3<f32>, dim: vec4<f32>) -> f32 {
    var nb = dim.xyz;
    nb.z = 0.0;
    if nb.x > nb.y { nb.y = 0.0; } else { nb.x = 0.0; }
    return length(max(abs(v) - nb, vec3(0.0)));
}

// --- Surface parameterisation and patterns -----------------------------------

// VQ `getUVW`: unwraps a rounded box into (u, v) surface coordinates.
fn get_uvw(wp: vec3<f32>, center: vec3<f32>, dim: vec4<f32>) -> vec3<f32> {
    let min_corner = center - (dim.xyz + dim.w);
    let pos = wp - min_corner;
    let c = wp - center;
    let inner = dim.xyz;
    var uvw = pos;
    let o = max(abs(c) - inner, vec3(0.0));
    let ang_mod = (2.0 / PI) * max(floor(sqrt(dim.w * dim.w * 2.0)), 1.0);
    let lxy = length(o.xy);
    if lxy > 0.0 {
        let n1 = normalize(o.xy) * sign(c.xy);
        if n1.x == 0.0 {
            uvw = vec3(pos.x, pos.z, uvw.z);
        } else if n1.y == 0.0 {
            uvw = vec3(pos.y, pos.z, uvw.z);
        } else {
            uvw = vec3(atan2(n1.y, n1.x) * ang_mod, pos.z, uvw.z);
        }
    }
    let l2 = length(vec2(lxy, o.z));
    if l2 > 0.0 {
        let n2 = vec2(lxy, o.z) / l2;
        if n2.x != 0.0 && n2.y != 0.0 {
            uvw.y = atan2(n2.y, n2.x) * ang_mod;
        }
        if n2.x == 0.0 {
            let cn = abs(c / inner);
            if cn.x > cn.y {
                uvw = vec3(pos.y, pos.x, uvw.z);
                if c.x > 0.0 { uvw.y = -uvw.y + 0.5; }
            } else {
                uvw = vec3(pos.x, pos.y, uvw.z);
                if c.y > 0.0 { uvw.y = -uvw.y + 0.5; }
            }
        }
    }
    return vec3(uvw.xy, 0.0);
}

// Running-bond brick: 0 at a brick's centre, 1 at the mortar.
fn get_brick(uvw: vec3<f32>) -> f32 {
    let mv1 = f32(fmod1(uvw.y, 2.0) < 1.0);
    let mv2 = f32(fmod1(uvw.z, 2.0) < 1.0);
    var r = fmod3(uvw + vec3(0.5 * (mv1 + mv2), 0.0, 0.0), vec3(1.0));
    r = abs(r - 0.5) * 2.0;
    return max(max(r.x, r.y), r.z);
}

// Fish-scale roof shingles.
fn get_shingle(uv: vec2<f32>) -> f32 {
    let iuv = floor(uv);
    var dis = uv - iuv;
    dis.y = 1.0 - dis.y;
    if fmod1(iuv.x + iuv.y, 2.0) >= 1.0 {
        dis.x = 1.0 - dis.x;
    }
    let dl = length(dis);
    if dl < 1.0 {
        let f = 1.0 - (dis.y * 0.5 + 0.5);
        return mix(f, dl, pow(dl, 8.0));
    }
    return 1.0 - dis.y * 0.5;
}

// Concentric growth rings (VQ `getWoodGrain`).
fn get_wood_grain(wp: vec3<f32>, wood_rad: f32, board_dir: f32, stretch: f32) -> f32 {
    var p = wp;
    if board_dir == 1.0 { p = wp.xzy; }
    if board_dir == 2.0 { p = wp.yxz; }
    if board_dir == 3.0 { p = wp.yzx; }
    if board_dir == 4.0 { p = wp.zxy; }
    if board_dir == 5.0 { p = wp.zyx; }
    let diam = wood_rad * 2.0;
    var center = floor((p.xy + wood_rad) / diam) * diam;
    let wv = normalize(p.xy - center + 1.0e-6);
    let len = p.z;
    center.y *= stretch;
    var q = p.xy;
    q.y *= stretch;
    let a = atan2(wv.y, wv.x);
    var m = sin(
        (distance(q, center) + wood_rad / 2.0)
        * ((8.0 + sin(a * 24.0) * 0.0625 + sin(a * 12.0) * 0.125 + sin(len / 16.0) * 0.5 + sin(len / 4.0) * 0.25)
            / (wood_rad / sqrt(2.0)))
    );
    if m < 0.0 {
        m = (1.0 - m) / 2.0;
    }
    return m;
}

// --- Scene -------------------------------------------------------------------

struct Solid {
    dist: f32,
    mat: f32,
    variation: f32,
}

// VQ `mapSolid`, evaluated over the candidate list. Positions in VQ space.
fn map_solid(p: vec3<f32>, want_material: bool) -> Solid {
    var sub1 = BIG;
    var sub2 = BIG;
    var r1 = vec3(BIG, -1.0, BIG);
    var r2 = r1;
    for (var c = 0u; c < candidate_count; c++) {
        let i = candidates[c];
        let pr = prims[i];
        let vis = sd_box(p - pr.vis_center.xyz, pr.vis_half.xyz);
        let q = p - pr.box_center.xyz;
        let b1 = ud_round_box(q, pr.box_dim, pr.params.xy, pr.params.z);
        let b2 = ud_round_box(q, pr.box_dim, pr.params.xy, pr.params.z * 2.0);
        let mb = ud_min_box(q, pr.box_dim);
        sub1 = min(sub1, max(b1.z * 0.5, vis));
        r1 = op_u_tie(r1, vec3(max(max(b1.x, b1.y) * 0.5, vis), f32(i), mb));
        sub2 = min(sub2, max(b2.z * 0.5, vis));
        r2 = op_u_tie(r2, vec3(max(max(b2.x, b2.y) * 0.5, vis), f32(i), mb));
    }
    if sub1 < BIG { r1.x = max(r1.x, -(sub1 - 0.01)); }
    if sub2 < BIG { r2.x = max(r2.x, -(sub2 - 0.01)); }

    var out: Solid;
    out.dist = r1.x;
    out.mat = MAT_PLASTER;
    out.variation = 0.0;
    if r2.y < 0.0 {
        return out;
    }

    var res = r2.xy;
    let orig = r1.x;
    let pr = prims[u32(r2.y)];
    let q = p - pr.box_center.xyz;
    var uvw = get_uvw(p, pr.box_center.xyz, pr.box_dim);
    uvw.z = ud_round_box(q, pr.box_dim, pr.params.xy, pr.params.z).x;
    let ms = uvw * 0.5;

    // Layered shells: 0.2, 0.4 and 0.6 units into the wall.
    var bx = res.x;
    var by = res.x + 0.4;
    var bz = res.x + 0.6;
    var bw = res.x + 0.2;
    bx = max(max(-by, bx), orig);
    by = max(max(-bz, by), orig);
    bz = max(bz, orig);
    bw = max(bw, orig);

    // Timber frame: posts, beams and diagonal braces.
    var timber = op_u3(
        sd_box_m(2.0, op_rep(ms, vec3(2.0)), vec3(0.125, 0.84, 10.0)),
        sd_box_m(5.0, op_rep(ms + vec3(1.0, 1.0, 0.0), vec3(4.0, 2.0, 2.0)), vec3(1.97, 0.125, 10.0)),
        sd_box_m(2.0, op_rep(ms + vec3(1.0, ms.x, 0.0), vec3(2.0)), vec3(0.8, 0.15, 10.0)),
    );
    let wood_dir = timber.y;
    timber.y = MAT_WOOD;

    let style = pr.params.w;
    var shingle = 0.0;
    var brick_cell = 0.0;
    if style < 2.0 {
        let my_dis = select(orig, bw, style < 1.0);
        timber.x = max(timber.x, orig);
        brick_cell = get_brick(uvw * vec3(0.5, 1.0, 0.5));
        let d1 = max(brick_cell, 0.8) - 0.8;
        let d2 = (snoise_2d(uvw.xy * 2.0) * 0.5 + 0.5) * 0.3 + f32(style > 0.0);
        let t1 = vec2(my_dis + d1, select(MAT_BRICK, MAT_PLASTER, brick_cell > 0.95));
        let t2 = vec2(max(-(my_dis + d1), orig + d2), MAT_PLASTER);
        res = op_u(t1, t2);
        if style < 1.0 {
            res.x = max(-timber.x, res.x);
            res = op_u(res, timber);
        }
    } else {
        shingle = get_shingle(abs(uvw.xy * 2.0));
        let eave = 0.25 - clamp(p.z - (pr.vis_center.z - pr.vis_half.z), 0.0, 0.25);
        let t1 = vec2(bx + shingle * 0.2 + eave * shingle, MAT_SHINGLE);
        let stagger = f32(fmod1(uvw.y, 1.0) < 0.5) * 2.0;
        let t2 = vec2(max(by, sd_box(op_rep(uvw + vec3(stagger, 0.0, 0.0), vec3(4.0, 0.5, 2.0)), vec3(1.95, 0.23, 10.0))), MAT_WOOD);
        let t3 = vec2(max(bz, timber.x), MAT_WOOD);
        res = op_u3(t1, t2, t3);
    }

    out.dist = res.x;
    out.mat = res.y;
    if want_material {
        if res.y == MAT_SHINGLE {
            out.variation = shingle * 0.3 + 0.3;
        } else if res.y == MAT_WOOD {
            out.variation = get_wood_grain(uvw * 0.5, 1.0, wood_dir, 4.0);
        } else if res.y == MAT_BRICK {
            // Slight per-brick variation (VQ used a constant here).
            out.variation = hash_vec(floor(uvw * vec3(0.5, 1.0, 0.5) + 0.5)) * 0.25;
        }
    }
    return out;
}

fn solid_dist(p: vec3<f32>) -> f32 {
    return map_solid(p, false).dist;
}

fn solid_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.025;
    return normalize(vec3(
        solid_dist(p + vec3(e, 0.0, 0.0)) - solid_dist(p - vec3(e, 0.0, 0.0)),
        solid_dist(p + vec3(0.0, e, 0.0)) - solid_dist(p - vec3(0.0, e, 0.0)),
        solid_dist(p + vec3(0.0, 0.0, e)) - solid_dist(p - vec3(0.0, 0.0, e)),
    ));
}

fn solid_ao(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occ = 0.0;
    var w = 1.0;
    for (var i = 1; i <= 4; i++) {
        let h = 0.35 * f32(i);
        occ += (h - max(solid_dist(p + n * h), 0.0)) / h * w;
        w *= 0.65;
    }
    return clamp(1.0 - occ * 0.4, 0.0, 1.0);
}

// VQ `lineStep`: keep only primitives whose visibility box the ray crosses.
fn gather_candidates(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32) -> vec2<f32> {
    candidate_count = 0u;
    var range = vec2(BIG, -BIG);
    let n = u32(structure.box_min.w);
    for (var i = 0u; i < n; i++) {
        let pr = prims[i];
        let pad = vec3(0.5);
        let hit = ray_box(o, d, pr.vis_center.xyz - pr.vis_half.xyz - pad, pr.vis_center.xyz + pr.vis_half.xyz + pad);
        let a = max(hit.x, t0);
        let b = min(hit.y, t1);
        if a <= b && candidate_count < MAX_CANDIDATES {
            candidates[candidate_count] = i;
            candidate_count += 1u;
            range = vec2(min(range.x, a), max(range.y, b));
        }
    }
    return range;
}

struct Hit {
    pos: vec3<f32>, // VQ local space
}

fn march(in_world: vec3<f32>, frag_coord: vec4<f32>) -> Hit {
    let ray = view_ray(in_world);
    // March in the structure's local space so it can be moved/rotated/scaled.
    let o_local = (structure.local_from_world * vec4(ray.origin, 1.0)).xyz;
    let d_local = normalize((structure.local_from_world * vec4(ray.dir, 0.0)).xyz);
    let tb = ray_box(o_local, d_local, structure.box_min.xyz, structure.box_max.xyz);
    // t is measured in local units from here on.
    let t_min = select(0.0, -BIG, ray.t_min < 0.0);
    var t = max(tb.x, t_min);
    var t_end = tb.y;
    if t > t_end {
        discard;
    }

    let o = to_vq(o_local);
    let d = to_vq(d_local);
    let range = gather_candidates(o, d, t, t_end);
    if candidate_count == 0u {
        discard;
    }
    t = max(t, range.x);
    t_end = min(t_end, range.y);

#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
    // Main pass with a depth prepass: reuse the prepass hit if it lies on
    // this structure, otherwise something else is in front.
    let surface = prepass_surface(frag_coord);
    if surface.w <= 0.0 {
        discard;
    }
    let pv = to_vq((structure.local_from_world * vec4(surface.xyz, 1.0)).xyz);
    if abs(solid_dist(pv)) > 0.02 {
        discard;
    }
    var reused: Hit;
    reused.pos = pv;
    return reused;
#endif
#endif

    let max_steps = i32(structure.box_max.w);
    var hit = false;
    for (var i = 0; i < max_steps; i++) {
        let dist = solid_dist(o + d * t);
        if dist < 0.002 {
            hit = true;
            break;
        }
        t += dist;
        if t > t_end {
            break;
        }
    }
    if !hit {
        discard;
    }
    // Refinement (VQ castSolid).
    t -= 0.003;
    for (var i = 0; i < 8; i++) {
        let dist = solid_dist(o + d * t);
        if dist < 0.002 {
            break;
        }
        t += dist * 0.5;
    }
    var out: Hit;
    out.pos = o + d * t;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> VqFragmentOutput {
    let hit = march(in.world_position.xyz, in.position);
    let n_vq = solid_normal(hit.pos);
    let local = vec4(from_vq(hit.pos), 1.0);
    let world = (structure.world_from_local * local).xyz;
    // Normals transform with the inverse transpose.
    let normal = normalize((transpose(structure.local_from_world) * vec4(from_vq(n_vq), 0.0)).xyz);

#ifdef PREPASS_PIPELINE
    let previous_world = (structure.previous_world_from_local * local).xyz;
    return prepass_output(world, previous_world, normal);
#else
    let solid = map_solid(hit.pos, true);
    var s: Surface;
    s.world_position = world;
    s.normal = normal;
    s.mat = u32(solid.mat);
    s.variation = solid.variation;
    s.ao = solid_ao(hit.pos, n_vq);
    s.specular = 0.0;
    s.contact_shadow = 1.0;

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
