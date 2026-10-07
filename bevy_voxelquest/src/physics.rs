//! avian3d colliders for the Voxel Quest world.
//!
//! Voxel Quest kept its world out of Bullet: it read the GPU terrain back
//! into a voxel grid and pushed characters out of it with velocity hacks.
//! Here the world gets real colliders built from the same CPU distance fields
//! the shaders mirror:
//!
//! * terrain tiles near the [`VqTerrainFocus`] get static heightfield
//!   colliders, generated on background threads;
//! * each [`VqStructure`] gets a static voxel collider sampled from its walls
//!   (hollow, so things can go inside towers).
//!
//! Add avian's `PhysicsPlugins` yourself; this plugin only adds colliders.

use avian3d::prelude::*;
use bevy::{
    platform::collections::HashMap,
    prelude::*,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
};

use crate::{
    structure::VqStructure,
    terrain::{VqTerrain, VqTerrainFocus},
};

/// Adds static colliders for terrain and structures.
pub struct VqPhysicsPlugin;

impl Plugin for VqPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VqPhysicsSettings>()
            .init_resource::<TerrainColliders>()
            .add_systems(
                Update,
                (
                    stream_terrain_colliders,
                    finish_terrain_colliders,
                    structure_colliders,
                ),
            );
    }
}

/// Physics options.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource)]
pub struct VqPhysicsSettings {
    /// Terrain colliders exist within this many tiles of the focus.
    pub terrain_radius_tiles: i32,
    /// Heightfield samples per tile side.
    pub terrain_resolution: usize,
    /// Edge length of the voxels used for structure colliders.
    pub structure_voxel_size: f32,
}

impl Default for VqPhysicsSettings {
    fn default() -> Self {
        Self {
            terrain_radius_tiles: 1,
            terrain_resolution: 129,
            structure_voxel_size: 0.5,
        }
    }
}

/// A terrain collider entity covering one tile.
#[derive(Component, Clone, Copy, Debug)]
pub struct VqTerrainCollider {
    pub coord: IVec2,
}

/// Heightfield samples, `[z][x]`.
type Heights = Vec<Vec<f32>>;

#[derive(Resource, Default)]
struct TerrainColliders {
    loaded: HashMap<IVec2, Entity>,
    pending: HashMap<IVec2, Task<Heights>>,
}

fn stream_terrain_colliders(
    mut commands: Commands,
    terrain: Option<Res<VqTerrain>>,
    settings: Res<VqPhysicsSettings>,
    focus: Query<&GlobalTransform, With<VqTerrainFocus>>,
    mut colliders: ResMut<TerrainColliders>,
) {
    let Some(terrain) = terrain else { return };
    let tile = terrain.field.settings.tile_size;
    let center = focus
        .iter()
        .next()
        .map(|t| t.translation())
        .unwrap_or_default();
    let center_tile = IVec2::new(
        (center.x / tile).floor() as i32,
        (center.z / tile).floor() as i32,
    );
    let r = settings.terrain_radius_tiles;
    let wanted = |c: &IVec2| (*c - center_tile).abs().max_element() <= r;

    let colliders = &mut *colliders;
    colliders.loaded.retain(|coord, entity| {
        let keep = wanted(coord);
        if !keep {
            commands.entity(*entity).despawn();
        }
        keep
    });
    colliders.pending.retain(|coord, _| wanted(coord));

    let pool = AsyncComputeTaskPool::get();
    for dz in -r..=r {
        for dx in -r..=r {
            let coord = center_tile + IVec2::new(dx, dz);
            if colliders.loaded.contains_key(&coord) || colliders.pending.contains_key(&coord) {
                continue;
            }
            let field = terrain.field.clone();
            let n = settings.terrain_resolution.max(2);
            let task = pool.spawn(async move {
                let x0 = coord.x as f32 * tile;
                let z0 = coord.y as f32 * tile;
                let step = tile / (n - 1) as f32;
                // heights[i][j]: i along X, j along Z.
                (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| field.height_at(x0 + i as f32 * step, z0 + j as f32 * step))
                            .collect()
                    })
                    .collect::<Heights>()
            });
            colliders.pending.insert(coord, task);
        }
    }
}

fn finish_terrain_colliders(
    mut commands: Commands,
    terrain: Option<Res<VqTerrain>>,
    mut colliders: ResMut<TerrainColliders>,
) {
    let Some(terrain) = terrain else { return };
    let tile = terrain.field.settings.tile_size;
    let colliders = &mut *colliders;
    let mut done = Vec::new();
    for (coord, task) in &mut colliders.pending {
        if let Some(heights) = block_on(future::poll_once(task)) {
            done.push((*coord, heights));
        }
    }
    for (coord, heights) in done {
        colliders.pending.remove(&coord);
        let center = Vec3::new(
            (coord.x as f32 + 0.5) * tile,
            0.0,
            (coord.y as f32 + 0.5) * tile,
        );
        let entity = commands
            .spawn((
                Name::new(format!("VQ terrain collider {coord}")),
                VqTerrainCollider { coord },
                RigidBody::Static,
                Collider::heightfield(heights, Vec3::new(tile, 1.0, tile)),
                Transform::from_translation(center),
            ))
            .id();
        colliders.loaded.insert(coord, entity);
    }
}

/// Builds a voxel collider for a structure (local space). Returns `None` if
/// the structure has no solid voxels.
pub fn structure_collider(structure: &VqStructure, voxel_size: f32) -> Option<Collider> {
    let (min, max) = structure.local_aabb()?;
    let lo = (min / voxel_size).floor().as_ivec3();
    let hi = (max / voxel_size).ceil().as_ivec3();
    let mut cells = Vec::new();
    for z in lo.z..hi.z {
        for y in lo.y..hi.y {
            for x in lo.x..hi.x {
                let c = IVec3::new(x, y, z);
                let center = (c.as_vec3() + 0.5) * voxel_size;
                if structure.distance(center) < voxel_size * 0.25 {
                    cells.push(c);
                }
            }
        }
    }
    if cells.is_empty() {
        return None;
    }
    Some(Collider::voxels(Vec3::splat(voxel_size), &cells))
}

fn structure_colliders(
    mut commands: Commands,
    settings: Res<VqPhysicsSettings>,
    structures: Query<(Entity, &VqStructure, Has<RigidBody>), Changed<VqStructure>>,
) {
    for (entity, structure, has_body) in &structures {
        let mut e = commands.entity(entity);
        match structure_collider(structure, settings.structure_voxel_size) {
            Some(collider) => {
                e.insert(collider);
                if !has_body {
                    e.insert(RigidBody::Static);
                }
            }
            None => {
                e.remove::<Collider>();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structure::{VqPrim, VqPrimTemplate};

    #[test]
    fn tower_collider_is_hollow() {
        let tower = VqStructure::new(vec![VqPrim::new(VqPrimTemplate::tower(), Vec3::ZERO)]);
        let collider = structure_collider(&tower, 1.0).expect("tower has voxels");
        // A point in the wall is inside the collider, the courtyard is not.
        let inside = |p: Vec3| collider.contains_point(Vec3::ZERO, Quat::IDENTITY, p);
        assert!(inside(Vec3::new(7.0, 0.0, 0.0)));
        assert!(!inside(Vec3::new(0.0, 0.0, 0.0)));
        assert!(!inside(Vec3::new(12.0, 0.0, 0.0)));
    }
}
