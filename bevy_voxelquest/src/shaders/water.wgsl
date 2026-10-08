// Voxel Quest sea surface: 8 directional sine waves (WaveFuncs.c) on a
// plane, shaded with the WATER palette.

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::{view, globals},
}
#import bevy_voxelquest::common::{Surface, shade, to_vq}

struct WaterParams {
    // World units per VQ wave unit (VQ samples waves at pos * 0.25).
    wave_scale: f32,
    // Wave height multiplier for the normal.
    height: f32,
    time_scale: f32,
    // Opacity at normal incidence / grazing.
    alpha_min: f32,
    alpha_max: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(3) var<uniform> water: WaterParams;

const PI2: f32 = 6.28318;
const WAVE_SPEED: f32 = 1.5;
const WAVE_SCALE: f32 = 60.0;

fn amplitude(i: u32) -> f32 {
    var a = array<f32, 8>(0.25, 0.5, 1.0, 4.0, 16.0, 32.0, 64.0, 128.0);
    return a[i] / 256.0;
}

fn wavelength(i: u32) -> f32 {
    var w = array<f32, 8>(48.0, 40.0, 32.0, 16.0, 8.0, 4.0, 2.0, 1.0);
    return WAVE_SCALE / w[i];
}

fn speed(i: u32) -> f32 {
    var s = array<f32, 8>(5.0, 7.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0);
    return WAVE_SPEED * s[i];
}

fn direction(i: u32) -> vec2<f32> {
    var a = array<f32, 8>(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.825);
    return vec2(cos(a[i] * PI2), sin(a[i] * PI2));
}

// VQ `waveNormal`: analytic derivative of the wave sum.
fn wave_normal(p: vec2<f32>, t: f32, dist: f32) -> vec3<f32> {
    var dx = 0.0;
    var dy = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let frequency = PI2 / wavelength(i);
        let phase = speed(i) * frequency;
        let theta = dot(direction(i), p);
        let c = cos(theta * frequency + t * phase);
        // Fade out waves too small to resolve at this distance (VQ fades
        // its short waves with camera distance too).
        let resolve = clamp(2.0 - dist / (wavelength(i) * water.wave_scale * 30.0), 0.0, 1.0);
        let a = amplitude(i) * frequency * 2.0 * resolve;
        dx += a * direction(i).x * c;
        dy += a * direction(i).y * c;
    }
    return normalize(vec3(-dx * water.height, -dy * water.height, 1.0));
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let world = in.world_position.xyz;
    let p = to_vq(world).xy / water.wave_scale;
    let t = globals.time * water.time_scale;
    let n_vq = wave_normal(p, t, distance(world, view.world_position));
    let normal = normalize(vec3(n_vq.x, n_vq.z, -n_vq.y));

    let V = normalize(view.world_position - world);
    let fresnel = pow(1.0 - clamp(dot(normal, V), 0.0, 1.0), 3.0);

    var s: Surface;
    s.world_position = world;
    s.normal = normal;
    s.mat = 17u; // WATER
    s.variation = clamp(0.3 + n_vq.x * 2.0, 0.0, 1.0);
    s.ao = 1.0;
    s.specular = 1.5;
    s.contact_shadow = 1.0;
    var color = shade(s, in.position);
    color.a = mix(water.alpha_min, water.alpha_max, fresnel);
    return color;
}
