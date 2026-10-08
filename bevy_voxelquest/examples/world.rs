//! A Voxel Quest world: streamed terrain, a small walled village, sea, sky and
//! (with the `physics` feature) crates you can throw at it.
//!
//! Controls: WASD + Space/Shift to fly, hold the right mouse button to look,
//! F to throw a crate, P to toggle the pixelated look, M to switch between
//! VQ palette lighting and Bevy PBR, [ and ] to change the time of day.
//!
//! Set `VQ_SCREENSHOT=out.png` to save a screenshot after a few frames and exit.
//! Run from the `bevy_voxelquest` directory to use VQ's original heightmaps
//! from `../data` (falls back to procedural terrain otherwise).

use bevy::{
    anti_alias::taa::TemporalAntiAliasing,
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::AccumulatedMouseMotion,
    light::CascadeShadowConfigBuilder,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};
use bevy_voxelquest::{prelude::*, structure::VqPrimStyle, terrain::VqTerrain};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Voxel Quest in Bevy".into(),
            resolution: (env_or("VQ_WIDTH", 1280), env_or("VQ_HEIGHT", 720)).into(),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(VqWorldSettings {
        heightmap_source: if std::path::Path::new("../data/hm0.bmp").exists() {
            HeightmapSource::VoxelQuestBmp {
                hm0: "../data/hm0.bmp".into(),
                hm1: "../data/hm1.bmp".into(),
            }
        } else {
            HeightmapSource::Procedural
        },
        view_radius_tiles: env_or("VQ_VIEW_RADIUS", 6),
        ..default()
    })
    .insert_resource(VqShading {
        mode: if std::env::var("VQ_PBR").is_ok() {
            VqShadingMode::Pbr
        } else {
            VqShadingMode::Palette
        },
        ..default()
    })
    .add_plugins(VoxelQuestPlugins)
    .add_systems(Startup, setup)
    .add_systems(Update, (fly_camera, toggles, screenshot, bench));

    #[cfg(feature = "physics")]
    app.add_plugins(avian3d::prelude::PhysicsPlugins::default())
        .add_systems(Update, throw_crates)
        .add_systems(PostStartup, drop_crates);

    app.run();
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[derive(Component)]
struct FlyCamera {
    yaw: f32,
    pitch: f32,
}

fn setup(mut commands: Commands, terrain: Res<VqTerrain>, settings: Res<VqWorldSettings>) {
    let field = &terrain.field;
    // Find flat, dry lowland near the origin for the village.
    let sea = settings.sea_height();
    let mut village = Vec2::ZERO;
    let mut best = f32::MAX;
    for r in 0..48 {
        for a in 0..24 {
            let ang = a as f32 / 24.0 * std::f32::consts::TAU;
            let p = Vec2::new(ang.cos(), ang.sin()) * r as f32 * 40.0;
            let h = field.height_at(p.x, p.y);
            if h < sea + 6.0 {
                continue;
            }
            // Roughness: height spread over the keep's footprint.
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for (dx, dz) in [
                (-36.0, -36.0),
                (36.0, -36.0),
                (-36.0, 36.0),
                (36.0, 36.0),
                (0.0, 0.0),
            ] {
                let g = field.height_at(p.x + dx, p.y + dz);
                lo = lo.min(g);
                hi = hi.max(g);
            }
            let score = (hi - lo) + (h - sea) * 0.05 + r as f32 * 0.2;
            if score < best {
                best = score;
                village = p;
            }
        }
    }
    let ground = field.height_at(village.x, village.y);

    // Camera looking at the village from a hillside.
    let eye = Vec3::new(village.x - 70.0, 0.0, village.y + 95.0);
    let eye = eye.with_y(field.height_at(eye.x, eye.z).max(sea) + 45.0);
    let target = Vec3::new(village.x, ground + 8.0, village.y);
    let look = Transform::from_translation(eye).looking_at(target, Vec3::Y);
    let (yaw, pitch, _) = look.rotation.to_euler(EulerRot::YXZ);
    let pixelate: u32 = env_or("VQ_PIXELATE", 0);
    let mut camera = commands.spawn((
        Camera3d::default(),
        look,
        FlyCamera { yaw, pitch },
        VqTerrainFocus,
        VqSky,
        // VQ's palette already encodes the final colours.
        Tonemapping::None,
        DistanceFog {
            color: Color::srgb(0.55, 0.62, 0.8),
            falloff: FogFalloff::Linear {
                start: 600.0,
                end: settings.tile_size * settings.view_radius_tiles as f32,
            },
            ..default()
        },
    ));
    if pixelate > 1 {
        camera.insert(VqPixelate { factor: pixelate });
    }
    // TAA smooths the aliasing of ray-marched edges and sub-pixel rock and
    // brick detail. It adds a depth + motion-vector prepass; the VQ materials
    // reuse that depth in the main pass instead of marching twice.
    if std::env::var("VQ_NO_TAA").is_err() {
        camera.insert((TemporalAntiAliasing::default(), Msaa::Off));
    }

    commands.spawn((
        DirectionalLight {
            color: Color::srgb(1.0, 0.95, 0.85),
            illuminance: 10_000.0,
            shadow_maps_enabled: std::env::var("VQ_NO_SHADOWS").is_err(),
            ..default()
        },
        Transform::default().looking_to(Vec3::new(-0.5, -0.7, -0.35), Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            maximum_distance: 600.0,
            first_cascade_far_bound: 80.0,
            ..default()
        }
        .build(),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 600.0,
        ..default()
    });

    spawn_village(&mut commands, field, village);
}

/// A walled keep in VQ's primitive vocabulary: everything that should merge
/// (towers, walls, gate) is one `VqStructure`, so each primitive's interior
/// is carved out of the walls it touches.
fn spawn_village(
    commands: &mut Commands,
    field: &bevy_voxelquest::terrain::TerrainField,
    at: Vec2,
) {
    let ground = |x: f32, z: f32| field.height_at(at.x + x, at.y + z);
    // Lowest ground under the keep, so no wall floats.
    let mut base = f32::MAX;
    for x in [-32.0, 0.0, 32.0] {
        for z in [-32.0, 0.0, 32.0] {
            base = base.min(ground(x, z));
        }
    }

    let mut prims = Vec::new();
    for (i, (x, z)) in [(-24.0, -24.0), (24.0, -24.0), (-24.0, 24.0), (24.0, 24.0)]
        .into_iter()
        .enumerate()
    {
        // Tower bounds are y ∈ [-16, 16]; the walls are visible up to +8.
        prims.push(VqPrim::new(VqPrimTemplate::tower(), Vec3::new(x, 4.0, z)));
        let roof = if i % 3 == 0 {
            VqPrimTemplate::roof_pointed_tower()
        } else {
            VqPrimTemplate::roof_sphere_tower()
        };
        prims.push(VqPrim::new(roof, Vec3::new(x, 12.0, z)));
    }
    for x in [-24.0, 24.0] {
        prims.push(VqPrim::new(
            VqPrimTemplate::wall_along_z(),
            Vec3::new(x, -4.0, 0.0),
        ));
    }
    prims.push(VqPrim::new(
        VqPrimTemplate::wall_along_x(),
        Vec3::new(0.0, -4.0, -24.0),
    ));
    prims.push(VqPrim::new(
        VqPrimTemplate::wall_along_x(),
        Vec3::new(0.0, -4.0, 24.0),
    ));
    prims.push(VqPrim::new(
        VqPrimTemplate::portal_x(),
        Vec3::new(0.0, -10.0, 24.0),
    ));

    commands.spawn((
        Name::new("keep"),
        VqStructure::new(prims),
        Transform::from_xyz(at.x, base + 12.0, at.y),
    ));

    // A timber-framed hall with a barrel roof in the courtyard.
    let hall = VqPrimTemplate::custom(
        Vec3::new(-12.0, -6.0, -6.0),
        Vec3::new(12.0, 6.0, 6.0),
        1.0,
        1.0,
        VqPrimStyle::TimberFrame,
    );
    commands.spawn((
        Name::new("hall"),
        VqStructure::new(vec![
            VqPrim::new(hall, Vec3::ZERO),
            VqPrim::new(VqPrimTemplate::roof_barrel_x(), Vec3::new(0.0, 5.5, 0.0)),
        ]),
        Transform::from_xyz(at.x, ground(0.0, 0.0) + 5.0, at.y),
    ));
}

fn fly_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut cameras: Query<(&mut Transform, &mut FlyCamera)>,
) {
    for (mut t, mut fly) in &mut cameras {
        if buttons.pressed(MouseButton::Right) {
            fly.yaw -= motion.delta.x * 0.003;
            fly.pitch = (fly.pitch - motion.delta.y * 0.003).clamp(-1.5, 1.5);
            t.rotation = Quat::from_euler(EulerRot::YXZ, fly.yaw, fly.pitch, 0.0);
        }
        let mut dir = Vec3::ZERO;
        let (f, r) = (t.forward().as_vec3(), t.right().as_vec3());
        for (key, d) in [
            (KeyCode::KeyW, f),
            (KeyCode::KeyS, -f),
            (KeyCode::KeyD, r),
            (KeyCode::KeyA, -r),
            (KeyCode::Space, Vec3::Y),
            (KeyCode::ShiftLeft, -Vec3::Y),
        ] {
            if keys.pressed(key) {
                dir += d;
            }
        }
        let speed = if keys.pressed(KeyCode::ControlLeft) {
            300.0
        } else {
            60.0
        };
        t.translation += dir.normalize_or_zero() * speed * time.delta_secs();
    }
}

fn toggles(
    keys: Res<ButtonInput<KeyCode>>,
    mut shading: ResMut<VqShading>,
    mut commands: Commands,
    cameras: Query<(Entity, Has<VqPixelate>), With<FlyCamera>>,
    time: Res<Time>,
) {
    if keys.just_pressed(KeyCode::KeyM) {
        shading.mode = match shading.mode {
            VqShadingMode::Palette => VqShadingMode::Pbr,
            VqShadingMode::Pbr => VqShadingMode::Palette,
        };
    }
    let dt = time.delta_secs() * 0.3;
    if keys.pressed(KeyCode::BracketLeft) {
        shading.time_of_day = (shading.time_of_day - dt).max(0.0);
    }
    if keys.pressed(KeyCode::BracketRight) {
        shading.time_of_day = (shading.time_of_day + dt).min(1.0);
    }
    if keys.just_pressed(KeyCode::KeyP) {
        for (e, has) in &cameras {
            if !has {
                commands.entity(e).insert(VqPixelate::default());
            }
        }
    }
}

#[cfg(feature = "physics")]
fn throw_crates(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    cameras: Query<&Transform, With<FlyCamera>>,
    palette: Res<VqPalette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use avian3d::prelude::*;
    if !keys.just_pressed(KeyCode::KeyF) {
        return;
    }
    for t in &cameras {
        commands.spawn((
            RigidBody::Dynamic,
            Collider::cuboid(2.0, 2.0, 2.0),
            LinearVelocity(t.forward() * 60.0),
            Mesh3d(meshes.add(Cuboid::new(2.0, 2.0, 2.0))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: palette.color(VqMat::Wood, 0.5, 0.7),
                perceptual_roughness: 0.9,
                ..default()
            })),
            Transform::from_translation(t.translation + t.forward() * 4.0),
        ));
    }
}

/// With `VQ_DROP_CRATES=1`, rains crates onto the keep at startup.
#[cfg(feature = "physics")]
fn drop_crates(
    mut commands: Commands,
    keeps: Query<&Transform, With<VqStructure>>,
    palette: Res<VqPalette>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    use avian3d::prelude::*;
    if std::env::var("VQ_DROP_CRATES").is_err() {
        return;
    }
    let Some(center) = keeps.iter().next().map(|t| t.translation) else {
        return;
    };
    let mesh = meshes.add(Cuboid::new(3.0, 3.0, 3.0));
    let material = materials.add(StandardMaterial {
        base_color: palette.color(VqMat::Wood, 0.5, 0.7),
        perceptual_roughness: 0.9,
        ..default()
    });
    for i in 0..24 {
        let a = i as f32 * 2.4;
        let r = 6.0 + i as f32 * 1.6;
        commands.spawn((
            RigidBody::Dynamic,
            Collider::cuboid(3.0, 3.0, 3.0),
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(
                center + Vec3::new(a.cos() * r, 30.0 + i as f32 * 2.0, a.sin() * r),
            )
            .with_rotation(Quat::from_euler(EulerRot::XYZ, a, a * 0.7, 0.0)),
        ));
    }
}

/// `VQ_BENCH=N`: average the frame time over N frames (after warm-up) and exit.
fn bench(
    time: Res<Time<Real>>,
    mut frames: Local<Vec<f32>>,
    mut seen: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(n) = std::env::var("VQ_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) else {
        return;
    };
    *seen += 1;
    // Let terrain tiles, pipelines and colliders settle first.
    if *seen <= env_or("VQ_BENCH_WARMUP", 8) {
        return;
    }
    frames.push(time.delta_secs() * 1000.0);
    if frames.len() == n {
        let mut sorted = frames.clone();
        sorted.sort_by(f32::total_cmp);
        let mean = frames.iter().sum::<f32>() / n as f32;
        println!("VQ_BENCH frames={n} mean_ms={mean:.1} median_ms={:.1}", sorted[n / 2]);
        exit.write(AppExit::Success);
    }
}

fn screenshot(mut commands: Commands, mut frame: Local<u32>, mut exit: MessageWriter<AppExit>) {
    let Ok(path) = std::env::var("VQ_SCREENSHOT") else {
        return;
    };
    *frame += 1;
    let at: u32 = env_or("VQ_SCREENSHOT_FRAME", 20);
    if *frame == at {
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    if *frame == at + 10 {
        exit.write(AppExit::Success);
    }
}
