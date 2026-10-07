//! Voxel Quest's material palette.
//!
//! Every material in `materials.js` is a small grid of HSV colour stops: rows
//! are "variations" (e.g. light/dark sand) and columns go from unlit to fully
//! lit. `Singleton::updateMatVol` stretched those stops into a 64×64×256 RGBA
//! volume (light × variation × material) that the lighting shader samples per
//! colour channel. This module is a direct port of that function.

use bevy::{
    asset::RenderAssetUsages,
    image::{ImageAddressMode, ImageFilterMode, ImageSampler, ImageSamplerDescriptor},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use serde_json::Value;

/// Light-axis resolution of the palette volume (VQ: `matVolDim.x`).
pub const PALETTE_LIGHT_STEPS: u32 = 64;
/// Variation-axis resolution of the palette volume (VQ: `matVolDim.y`).
pub const PALETTE_VARIATION_STEPS: u32 = 64;

const DEFAULT_MATERIALS: &str = include_str!("data/materials.json");

/// Material ids, in the order of Voxel Quest's `materials.js` (`TEX_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
#[repr(u32)]
pub enum VqMat {
    Null = 0,
    Debug,
    Unused,
    Sand,
    Stone,
    Snow,
    Grass,
    Mortar,
    Wood,
    Brick,
    Shingle,
    Plaster,
    Earth,
    Bark,
    TreeWood,
    Leaf,
    Gold,
    Water,
    Metal,
    Glass,
    MapLand,
    MapWater,
    Sky,
    Skin,
    Leather,
    Explosion,
    Pants,
    Armor,
    Meat,
    Bone,
}

impl VqMat {
    /// Index of this material in the palette volume.
    pub fn id(self) -> u32 {
        self as u32
    }
}

/// The palette volume and its CPU copy.
#[derive(Resource, Clone)]
pub struct VqPalette {
    /// 3D texture: x = light (0..1), y = variation (0..1), z = material id.
    pub image: Handle<Image>,
    /// Material names, indexed by id (e.g. `"SAND"`).
    pub names: Vec<String>,
    /// Raw RGBA8 data, `light + variation * 64 + mat * 64 * 64`.
    pub data: Vec<[u8; 4]>,
}

impl VqPalette {
    /// Number of materials in the palette.
    pub fn len(&self) -> u32 {
        self.names.len() as u32
    }

    /// Whether the palette has no materials.
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Looks up a palette colour (nearest sample, display/sRGB space).
    ///
    /// Handy for colouring ordinary Bevy meshes so they match the voxel world.
    pub fn color(&self, mat: VqMat, variation: f32, light: f32) -> Color {
        let x = (light.clamp(0.0, 1.0) * (PALETTE_LIGHT_STEPS - 1) as f32).round() as usize;
        let y = (variation.clamp(0.0, 1.0) * (PALETTE_VARIATION_STEPS - 1) as f32).round() as usize;
        let z = (mat.id() as usize).min(self.names.len().saturating_sub(1));
        let [r, g, b, _] = self.data[x
            + y * PALETTE_LIGHT_STEPS as usize
            + z * (PALETTE_LIGHT_STEPS * PALETTE_VARIATION_STEPS) as usize];
        Color::srgb_u8(r, g, b)
    }
}

/// Builds [`VqPalette`] from Voxel Quest's default `materials.js`, or from
/// [`VqPaletteSource`] if the app inserts one before startup.
pub struct VqPalettePlugin;

/// Override the palette with your own `materials.js`-formatted JSON.
#[derive(Resource, Clone)]
pub struct VqPaletteSource(pub String);

impl Plugin for VqPalettePlugin {
    fn build(&self, _app: &mut App) {}

    fn finish(&self, app: &mut App) {
        let json = app
            .world()
            .get_resource::<VqPaletteSource>()
            .map(|s| s.0.clone())
            .unwrap_or_else(|| DEFAULT_MATERIALS.to_string());
        let (names, data) = build_palette(&json).unwrap_or_else(|err| {
            error!("bevy_voxelquest: invalid materials JSON ({err}); using built-in palette");
            build_palette(DEFAULT_MATERIALS).expect("built-in materials.js is valid")
        });

        let mut image = Image::new(
            Extent3d {
                width: PALETTE_LIGHT_STEPS,
                height: PALETTE_VARIATION_STEPS,
                depth_or_array_layers: names.len() as u32,
            },
            TextureDimension::D3,
            data.iter().flatten().copied().collect(),
            TextureFormat::Rgba8Unorm,
            RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            address_mode_u: ImageAddressMode::ClampToEdge,
            address_mode_v: ImageAddressMode::ClampToEdge,
            address_mode_w: ImageAddressMode::ClampToEdge,
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            ..default()
        });
        let image = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        app.insert_resource(VqPalette { image, names, data });
    }
}

#[derive(Clone, Copy, Default)]
struct Stop {
    rgb: Vec3,
    power: f32,
    ratio: f32,
}

/// VQ's `hsv2rgb` (the "l" component is really HSV value).
fn hsv2rgb(h: f32, s: f32, v: f32) -> Vec3 {
    let k = Vec4::new(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = (Vec3::splat(h) + k.truncate()).fract() * 6.0 - Vec3::splat(k.w);
    let p = p.abs();
    v * Vec3::ONE.lerp((p - Vec3::ONE).clamp(Vec3::ZERO, Vec3::ONE), s)
}

/// Sorted `(key, value)` pairs of a JSON object (VQ relies on key order).
fn sorted_entries(v: &Value) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = v
        .as_object()
        .map(|o| o.iter().collect())
        .unwrap_or_default();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
}

/// Port of `Singleton::updateMatVol`. Returns material names and the
/// 64×64×N RGBA8 volume.
pub fn build_palette(json: &str) -> Result<(Vec<String>, Vec<[u8; 4]>), String> {
    let root: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let materials = root.get("materials").ok_or("missing \"materials\"")?;

    let mvx = PALETTE_LIGHT_STEPS as usize;
    let mvy = PALETTE_VARIATION_STEPS as usize;
    let mut names = Vec::new();
    let mut volume = Vec::new();

    for (key, material) in sorted_entries(materials) {
        // Keys look like "i003_SAND".
        names.push(
            key.split_once('_')
                .map_or(key.as_str(), |(_, n)| n)
                .to_string(),
        );

        // Pass 1: stretch each row's stops along the light axis.
        let mut rows: Vec<Vec<Vec3>> = Vec::new();
        for (_, row) in sorted_entries(material) {
            let mut stops: Vec<Stop> = sorted_entries(row)
                .into_iter()
                .map(|(_, stop)| {
                    let c = &stop["i0_color"];
                    let f = |i: usize| c[i].as_f64().unwrap_or(0.0) as f32;
                    let mut ratio = stop["i2_ratio"].as_f64().unwrap_or(1.0) as f32;
                    if ratio <= 0.0 {
                        ratio = 1.0 / (mvx as f32 - 1.0);
                    }
                    Stop {
                        rgb: hsv2rgb(f(0), f(1), f(2)),
                        power: stop["i1_power"].as_f64().unwrap_or(0.125) as f32,
                        ratio,
                    }
                })
                .collect();
            if stops.is_empty() {
                continue;
            }
            let tot: f32 = stops.iter().map(|s| s.ratio).sum();
            for s in &mut stops {
                s.ratio = s.ratio * (mvx as f32 - 1.0) / tot;
            }

            let count = stops.len();
            let mut line = Vec::with_capacity(mvx);
            for m in 0..count {
                let (prev, next) = if count == 1 {
                    (m, m)
                } else {
                    (m.saturating_sub(1), (m + 1).min(count - 1))
                };
                let (cur, p, nx) = (stops[m], stops[prev], stops[next]);
                let mut n = 0;
                while (n as f32) < cur.ratio + 0.1 && line.len() < mvx {
                    let lerp = n as f32 / cur.ratio;
                    let (a, b, l) = if m == 0 {
                        (cur, nx, lerp * 0.5)
                    } else if m == count - 1 {
                        (p, cur, lerp * 0.5 + 0.5)
                    } else if lerp < 0.5 {
                        (p, cur, lerp + 0.5)
                    } else {
                        (cur, nx, lerp - 0.5)
                    };
                    let power = a.power + (b.power - a.power) * l;
                    let w = l.max(0.0).powf(power * 8.0);
                    line.push(a.rgb.lerp(b.rgb, w));
                    n += 1;
                }
            }
            let last = *line.last().unwrap_or(&Vec3::ZERO);
            line.resize(mvx, last);
            rows.push(line);
        }

        // Pass 2: stretch the rows along the variation axis.
        let mut slice = vec![[0u8; 4]; mvx * mvy];
        let count = rows.len();
        if count > 0 {
            let ratio = mvy as f32 / count as f32;
            for x in 0..mvx {
                let mut tot_n = 0;
                for m in 0..count {
                    let (prev, next) = if count == 1 {
                        (m, m)
                    } else {
                        (m.saturating_sub(1), (m + 1).min(count - 1))
                    };
                    let mut n = 0;
                    while (n as f32) < ratio + 0.1 && tot_n < mvy {
                        let lerp = n as f32 / ratio;
                        let (a, b, w) = if m == 0 {
                            (m, next, lerp * 0.5)
                        } else if m == count - 1 {
                            (prev, m, lerp * 0.5 + 0.5)
                        } else if lerp < 0.5 {
                            (prev, m, lerp + 0.5)
                        } else {
                            (m, next, lerp - 0.5)
                        };
                        let c = rows[a][x].lerp(rows[b][x], w) * 255.0;
                        slice[x + tot_n * mvx] = [c.x as u8, c.y as u8, c.z as u8, 255];
                        tot_n += 1;
                        n += 1;
                    }
                }
                // Fill any remainder with the last written value.
                for y in tot_n.max(1)..mvy {
                    slice[x + y * mvx] = slice[x + (y - 1) * mvx];
                }
            }
        }
        volume.extend(slice);
    }

    if names.is_empty() {
        return Err("no materials".into());
    }
    Ok((names, volume))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_palette_matches_vq_material_order() {
        let (names, data) = build_palette(DEFAULT_MATERIALS).unwrap();
        assert_eq!(names.len(), 30);
        assert_eq!(names[VqMat::Sand.id() as usize], "SAND");
        assert_eq!(names[VqMat::Bone.id() as usize], "BONE");
        assert_eq!(data.len(), 64 * 64 * 30);
        // Lit grass should be brighter than unlit grass.
        let at = |light: usize| {
            let c = data[light + 32 * 64 + VqMat::Grass.id() as usize * 64 * 64];
            c[0] as u32 + c[1] as u32 + c[2] as u32
        };
        assert!(at(63) > at(0));
    }
}
