// Voxel Quest sky (PostLightingShader.c `getFogColor`): the SKY palette
// sampled by view elevation and time of day, plus sun and moon glows.

#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::{view, lights},
}
#import bevy_voxelquest::common::{palette, shading, srgb_to_linear}

struct SkyOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

const MAT_SKY: u32 = 22u;

@fragment
fn fragment(in: VertexOutput) -> SkyOutput {
    let ray = normalize(in.world_position.xyz - view.world_position);
    var to_light = normalize(vec3(shading.fallback_light_x, shading.fallback_light_y, shading.fallback_light_z));
    if lights.n_directional_lights > 0u {
        to_light = lights.directional_lights[0].direction_to_light;
    }
    let tod = shading.time_of_day;

    let zv = pow(1.0 - (ray.y + 1.0) * 0.5, 2.0);
    var col = palette(MAT_SKY, zv, tod);

    let sun = clamp(dot(to_light, ray), 0.0, 1.0);
    col += (pow(sun, 16.0) * vec3(1.0, 0.5, 0.0) + pow(sun, 64.0) * vec3(1.0))
        * pow(clamp(tod + 0.4, 0.0, 1.0), 8.0);
    let moon_dir = to_light * vec3(1.0, -1.0, 1.0);
    let moon = clamp(dot(moon_dir, ray), 0.0, 1.0);
    col += (pow(moon, 64.0) * vec3(0.5, 0.5, 1.0) + pow(moon, 256.0) * vec3(1.0))
        * pow(clamp(1.0 - (tod - 0.4), 0.0, 1.0), 8.0);

    var out: SkyOutput;
    out.color = vec4(srgb_to_linear(clamp(col, vec3(0.0), vec3(1.0))), 1.0);
    // Reverse-Z far plane: behind everything else.
    out.depth = 0.0;
    return out;
}
