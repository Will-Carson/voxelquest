// Terrain shadow proxy: only its depth (shadow maps) matters, so the main
// pass draws nothing. The default prepass shader writes the shadow depth.
@fragment
fn fragment() -> @location(0) vec4<f32> {
    discard;
}
