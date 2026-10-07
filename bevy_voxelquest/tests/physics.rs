//! Headless check that bodies come to rest on VQ terrain and structures.
#![cfg(feature = "physics")]

use std::{sync::Arc, time::Duration};

use avian3d::prelude::*;
use bevy::{ecs::system::RunSystemOnce, prelude::*, time::TimeUpdateStrategy};
use bevy_voxelquest::{
    VqWorldSettings,
    physics::VqPhysicsPlugin,
    structure::{VqPrim, VqPrimTemplate, VqStructure},
    terrain::{TerrainField, VqTerrain, VqTerrainFocus},
};

fn app(field: Arc<TerrainField>) -> App {
    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins,
        TransformPlugin,
        AssetPlugin::default(),
        bevy::mesh::MeshPlugin,
        bevy::scene::ScenePlugin,
        bevy::world_serialization::WorldSerializationPlugin,
        PhysicsPlugins::default(),
        VqPhysicsPlugin,
    ))
    .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f32(1.0 / 60.0)))
    .insert_resource(VqTerrain {
        field,
        heightmap: default(),
        voro: default(),
    });
    app.finish();
    app.cleanup();
    app
}

fn field() -> Arc<TerrainField> {
    Arc::new(TerrainField::generate(&VqWorldSettings {
        heightmap_resolution: 256,
        world_size: 2048.0,
        ..default()
    }))
}

/// Steps until terrain colliders have been generated, then `seconds` more.
fn run(app: &mut App, seconds: f32) {
    for _ in 0..600 {
        app.update();
        let ready = app
            .world_mut()
            .query::<&bevy_voxelquest::physics::VqTerrainCollider>()
            .iter(app.world())
            .count();
        if ready >= 9 {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    for _ in 0..(seconds * 60.0) as usize {
        app.update();
    }
}

#[test]
fn ball_rests_on_terrain() {
    let field = field();
    let (x, z) = (300.0, 200.0);
    let ground = field.height_at(x, z);
    let mut app = app(field);
    app.world_mut()
        .spawn((VqTerrainFocus, Transform::from_xyz(x, ground, z)));
    let ball = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::sphere(1.0),
            Transform::from_xyz(x, ground + 10.0, z),
        ))
        .id();
    run(&mut app, 4.0);
    // The ball may roll downhill, so compare against the ground where it ended
    // up. The heightfield samples every 2 units, so small rock cracks between
    // samples are bridged; allow for that.
    let p = app.world().get::<Transform>(ball).unwrap().translation;
    let below = app.world().resource::<VqTerrain>().field.height_at(p.x, p.z);
    assert!(
        (p.y - 1.0 - below).abs() < 1.5,
        "ball should rest on the ground: centre {p}, ground there {below} (started over {ground})"
    );
}

#[test]
fn crate_rests_on_tower_wall() {
    let field = field();
    let (x, z) = (-150.0, 400.0);
    let ground = field.height_at(x, z);
    let mut app = app(field);
    app.world_mut()
        .spawn((VqTerrainFocus, Transform::from_xyz(x, ground, z)));
    // Tower walls span radius 6..8 and are visible up to +8 (local).
    let top = ground + 40.0;
    app.world_mut().spawn((
        VqStructure::new(vec![VqPrim::new(VqPrimTemplate::tower(), Vec3::ZERO)]),
        Transform::from_xyz(x, top - 8.0, z),
    ));
    let body = app
        .world_mut()
        .spawn((
            RigidBody::Dynamic,
            Collider::cuboid(1.0, 1.0, 1.0),
            Transform::from_xyz(x + 7.0, top + 6.0, z),
        ))
        .id();
    run(&mut app, 4.0);
    let y = app.world().get::<Transform>(body).unwrap().translation.y;
    assert!(
        (top - 1.0..top + 2.0).contains(&y),
        "crate should rest on the wall top at {top}, got {y}"
    );
}

#[test]
fn heightfield_matches_terrain() {
    let field = field();
    let mut app = app(field.clone());
    app.world_mut()
        .spawn((VqTerrainFocus, Transform::from_xyz(300.0, 0.0, 200.0)));
    run(&mut app, 0.1);
    let mut worst: f32 = 0.0;
    for (x, z) in [(300.0, 200.0), (310.0, 150.0), (270.0, 230.0), (400.0, 60.0), (500.0, 20.0)] {
        let hit = app
            .world_mut()
            .run_system_once(move |q: SpatialQuery| {
                q.cast_ray(
                    Vec3::new(x, 1000.0, z),
                    Dir3::NEG_Y,
                    2000.0,
                    true,
                    &SpatialQueryFilter::default(),
                )
                .map(|h| 1000.0 - h.distance)
            })
            .unwrap();
        let expected = field.height_at(x, z);
        let got = hit.expect("ray should hit the terrain collider");
        worst = worst.max((got - expected).abs());
        println!("({x}, {z}): collider {got}, field {expected}");
    }
    assert!(worst < 2.0, "heightfield deviates by {worst}");
}
