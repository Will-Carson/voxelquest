// Shared ray-marching and Voxel Quest shading helpers.
//
// Every ray-marched material binds the palette and shading parameters at
// bindings 0..2 of the material group, so this file can declare them.
#define_import_path bevy_voxelquest::common

#import bevy_pbr::{
    mesh_view_bindings::view,
    view_transformations::{position_ndc_to_world, frag_coord_to_ndc},
}

#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
#import bevy_pbr::prepass_utils::prepass_depth
#endif
#endif

#ifndef PREPASS_PIPELINE
#import bevy_pbr::{
    mesh_view_bindings::lights,
    mesh_view_types::DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT,
    mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT,
    pbr_types,
    pbr_functions,
    shadows,
}
#endif

struct VqShading {
    // 0 = Voxel Quest palette lighting, 1 = Bevy PBR with palette albedo.
    mode: u32,
    // Number of materials (depth of the palette volume).
    palette_len: f32,
    // 0 = night (moon), 1 = day (sun high); tints VQ's bounce light.
    time_of_day: f32,
    // Strength of the SDF ambient occlusion.
    ao_strength: f32,
    // Strength of VQ's coloured rim terms.
    rim_strength: f32,
    // Fallback light direction (towards the light) when no directional light exists.
    fallback_light_x: f32,
    fallback_light_y: f32,
    fallback_light_z: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var palette_texture: texture_3d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(1) var palette_sampler: sampler;
@group(#{MATERIAL_BIND_GROUP}) @binding(2) var<uniform> shading: VqShading;

const BIG: f32 = 1.0e9;

// --- Ray setup --------------------------------------------------------------

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
    // Smallest valid t. Orthographic views (directional shadow maps) may
    // need geometry behind the near plane, so they allow negative t.
    t_min: f32,
}

fn is_orthographic() -> bool {
    return view.clip_from_view[3][3] == 1.0;
}

// Ray through the pixel that rasterised `surface_world` (a point on the
// bounding box), for the current view: the main camera or a shadow view.
fn view_ray(surface_world: vec3<f32>) -> Ray {
    let clip = view.clip_from_world * vec4(surface_world, 1.0);
    let ndc = clip.xy / clip.w;
    // Reverse-Z: NDC depth 1 is the near plane.
    let near = position_ndc_to_world(vec3(ndc, 1.0));
    var r: Ray;
    r.origin = near;
    r.dir = normalize(surface_world - near);
    r.t_min = select(0.0, -BIG, is_orthographic());
    return r;
}

// Slab test; returns (t_enter, t_exit). Miss when x > y.
fn ray_box(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / d;
    let t0 = (bmin - o) * inv;
    let t1 = (bmax - o) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

fn depth_at(world: vec3<f32>) -> f32 {
    let c = view.clip_from_world * vec4(world, 1.0);
    return clamp(c.z / c.w, 0.0, 1.0);
}

// Bevy (Y-up) <-> Voxel Quest (Z-up).
fn to_vq(p: vec3<f32>) -> vec3<f32> { return vec3(p.x, -p.z, p.y); }
fn from_vq(p: vec3<f32>) -> vec3<f32> { return vec3(p.x, p.z, -p.y); }

// --- Palette ---------------------------------------------------------------

// VQ PostLightingShader `unpackColor`: x = light, y = variation, z = material.
fn palette(mat: u32, variation: f32, light: f32) -> vec3<f32> {
    let z = (f32(mat) + 0.5) / shading.palette_len;
    return textureSampleLevel(
        palette_texture, palette_sampler,
        vec3(clamp(light, 0.0, 1.0), clamp(variation, 0.0, 1.0), z), 0.0
    ).rgb;
}

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    let q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    let d = q.x - min(q.w, q.y);
    let e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3(0.0), vec3(1.0)), c.y);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + 0.055) / 1.055, vec3(2.4));
    return select(hi, lo, c <= vec3(0.04045));
}

// --- Shading ---------------------------------------------------------------

struct Surface {
    world_position: vec3<f32>,
    normal: vec3<f32>,
    mat: u32,
    variation: f32,
    // Ambient occlusion in [0, 1] (1 = unoccluded).
    ao: f32,
    // Extra 0..1 specular weight (wet / shiny materials).
    specular: f32,
    // 0..1 multiplier on the shadow-map term for fine self-shadowing.
    contact_shadow: f32,
}

#ifndef PREPASS_PIPELINE

fn view_z_of(world: vec3<f32>) -> f32 {
    return dot(vec4(
        view.view_from_world[0].z, view.view_from_world[1].z,
        view.view_from_world[2].z, view.view_from_world[3].z,
    ), vec4(world, 1.0));
}

// Voxel Quest lighting (PreLightingShader + PostLightingShader), driven by
// the first Bevy directional light and its shadow map. Returns linear RGB.
fn shade_vq(s: Surface, frag_coord: vec4<f32>) -> vec3<f32> {
    let N = s.normal;
    let ao = mix(1.0, s.ao, shading.ao_strength);
    let tod = shading.time_of_day;

    var L = normalize(vec3(shading.fallback_light_x, shading.fallback_light_y, shading.fallback_light_z));
    var light_col = vec3(1.0);
    var shadow = 1.0;
    if lights.n_directional_lights > 0u {
        let dl = lights.directional_lights[0];
        L = dl.direction_to_light;
        // Only the hue of the light matters here; VQ's ramps encode brightness.
        light_col = dl.color.rgb / max(max(dl.color.r, max(dl.color.g, dl.color.b)), 1.0e-4);
        if (dl.flags & DIRECTIONAL_LIGHT_FLAGS_SHADOWS_ENABLED_BIT) != 0u {
            shadow = shadows::fetch_directional_shadow(
                0u, vec4(s.world_position, 1.0), N, view_z_of(s.world_position), frag_coord.xy
            );
        }
    }
    shadow *= s.contact_shadow;

    // --- PreLighting ---
    let col_amount = mix(0.0625, 0.25, tod);
    let front = clamp(dot(N, L), 0.0, 1.0);
    let bottom = clamp(dot(N, vec3(0.0, -1.0, 0.0)), 0.0, 1.0);
    let behind = clamp(dot(N, vec3(-L.x, 0.0, -L.z)), 0.0, 1.0);
    var tot = front * shadow * light_col;
    tot += vec3(0.0, 1.0, 1.0) * col_amount * bottom * 0.5;
    tot += vec3(0.9, 0.5, 0.2) * col_amount * (tod * 0.5 + 0.5) * behind * 0.5 * (1.0 - front);
    let lit = clamp(pow(
        max(mix(vec3(ao) * 0.2 + tot * 0.1, tot - (1.0 - ao), clamp(tot, vec3(0.0), vec3(1.0))), vec3(0.0))
            * (shadow * 0.5 + 0.5),
        vec3(0.5 + ao)
    ), vec3(0.0), vec3(1.0));

    // Specular (reflected view ray against the light).
    let V = normalize(view.world_position - s.world_position);
    let spec = pow(clamp(dot(reflect(-V, N), L), 0.0, 1.0), 8.0) * shadow * (0.25 + s.specular);

    // --- PostLighting ---
    let variation = s.variation * ao;
    var col = vec3(
        palette(s.mat, variation, lit.r).r,
        palette(s.mat, variation, lit.g).g,
        palette(s.mat, variation, lit.b).b,
    );
    var hsv = rgb2hsv(col);
    hsv.z = rgb2hsv(lit).z;
    col = mix(col, hsv2rgb(hsv), 0.25);
    col *= ao;

    let lmax = max(lit.r, max(lit.g, lit.b));
    let light_res = clamp(ao * lmax + lmax * 0.1, 0.0, 1.0);
    let look = -V;
    let facing = (dot(N, look) + 1.0) * 0.5;
    var rim = vec3(0.0, 0.65, 1.0) * pow(1.0 - light_res, 10.0) * facing;
    rim += vec3(1.0, 0.5, 0.0) * pow(light_res, 10.0) * (1.0 - facing);
    col += rim * 0.25 * shading.rim_strength;
    col += pow(lit, vec3(4.0)) * 0.1;

    col = mix(pow(col, vec3(2.0)) * 0.5, col, tod);
    col = pow(max(col * 0.9, vec3(0.0)), vec3(0.85));
    col += col * spec;
    return srgb_to_linear(clamp(col, vec3(0.0), vec3(1.0)));
}

fn make_pbr_input(s: Surface, frag_coord: vec4<f32>, base_color: vec3<f32>) -> pbr_types::PbrInput {
    var pbr = pbr_types::pbr_input_new();
    pbr.material.base_color = vec4(base_color, 1.0);
    pbr.material.perceptual_roughness = mix(0.9, 0.35, clamp(s.specular, 0.0, 1.0));
    pbr.material.reflectance = vec3(0.3);
    pbr.material.flags = pbr_types::STANDARD_MATERIAL_FLAGS_FOG_ENABLED_BIT
        | pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_OPAQUE;
    pbr.frag_coord = frag_coord;
    pbr.world_position = vec4(s.world_position, 1.0);
    pbr.world_normal = s.normal;
    pbr.N = s.normal;
    pbr.V = pbr_functions::calculate_view(pbr.world_position, is_orthographic());
    pbr.is_orthographic = is_orthographic();
    pbr.diffuse_occlusion = vec3(mix(1.0, s.ao, shading.ao_strength));
    pbr.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
    return pbr;
}

// Full shading entry point: lighting + Bevy fog / tonemapping.
fn shade(s: Surface, frag_coord: vec4<f32>) -> vec4<f32> {
    var color: vec4<f32>;
    let base = srgb_to_linear(palette(s.mat, s.variation, 0.8));
    let pbr = make_pbr_input(s, frag_coord, base);
    if shading.mode == 1u {
        color = pbr_functions::apply_pbr_lighting(pbr);
    } else {
        color = vec4(shade_vq(s, frag_coord), 1.0);
    }
    return pbr_functions::main_pass_post_lighting_processing(pbr, color);
}

#endif // !PREPASS_PIPELINE

// --- Prepass reuse ------------------------------------------------------------

#ifndef PREPASS_PIPELINE
#ifdef DEPTH_PREPASS
// When the camera has a depth prepass (required by TAA and SSAO), the prepass
// has already ray-marched every pixel. The main pass reads the nearest
// surface back instead of marching again: xyz = world position, w = depth
// (0 when nothing opaque covers the pixel).
fn prepass_surface(frag_coord: vec4<f32>) -> vec4<f32> {
    let depth = prepass_depth(frag_coord, 0u);
    let ndc = frag_coord_to_ndc(vec4(frag_coord.xy, depth, 1.0));
    return vec4(position_ndc_to_world(vec3(ndc.xy, depth)), depth);
}
#endif
#endif

// --- Fragment outputs -------------------------------------------------------

#ifdef PREPASS_PIPELINE
#ifdef MOTION_VECTOR_PREPASS
#import bevy_pbr::prepass_bindings::previous_view_uniforms
#endif

struct VqFragmentOutput {
#ifdef NORMAL_PREPASS
    @location(0) normal: vec4<f32>,
#endif
#ifdef MOTION_VECTOR_PREPASS
    @location(1) motion_vector: vec2<f32>,
#endif
    @builtin(frag_depth) frag_depth: f32,
}

// `previous_world` is where this surface point was last frame (equal to
// `world` for static geometry); camera motion is accounted for here.
fn prepass_output(world: vec3<f32>, previous_world: vec3<f32>, normal: vec3<f32>) -> VqFragmentOutput {
    var out: VqFragmentOutput;
#ifdef NORMAL_PREPASS
    out.normal = vec4(normal * 0.5 + 0.5, 1.0);
#endif
#ifdef MOTION_VECTOR_PREPASS
    // Same convention as Bevy's prepass.wgsl.
    let clip_t = view.unjittered_clip_from_world * vec4(world, 1.0);
    let prev_t = previous_view_uniforms.clip_from_world * vec4(previous_world, 1.0);
    out.motion_vector = (clip_t.xy / clip_t.w - prev_t.xy / prev_t.w) * vec2(0.5, -0.5);
#endif
    out.frag_depth = depth_at(world);
    return out;
}
#else
struct VqFragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) frag_depth: f32,
}
#endif
