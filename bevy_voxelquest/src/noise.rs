//! CPU-side noise and procedural texture generation.
//!
//! The simplex noise is a line-for-line port of the Ashima Arts GLSL noise
//! Voxel Quest uses (`Simplex2D.c`), so CPU and GPU agree.

use bevy::math::{Vec2, Vec3, Vec4};

#[inline]
fn mod289(x: f32) -> f32 {
    x - (x * (1.0 / 289.0)).floor() * 289.0
}

#[inline]
fn permute(x: f32) -> f32 {
    mod289(((x * 34.0) + 1.0) * x)
}

#[inline]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}

/// 3D simplex noise in roughly [-1, 1].
pub fn snoise3(v: Vec3) -> f32 {
    const C: Vec2 = Vec2::new(1.0 / 6.0, 1.0 / 3.0);
    let i = (v + Vec3::splat(v.dot(Vec3::splat(C.y)))).floor();
    let x0 = v - i + Vec3::splat(i.dot(Vec3::splat(C.x)));

    let g = Vec3::new(step(x0.y, x0.x), step(x0.z, x0.y), step(x0.x, x0.z));
    let l = Vec3::ONE - g;
    let lzxy = Vec3::new(l.z, l.x, l.y);
    let i1 = g.min(lzxy);
    let i2 = g.max(lzxy);

    let x1 = x0 - i1 + Vec3::splat(C.x);
    let x2 = x0 - i2 + Vec3::splat(C.y);
    let x3 = x0 - Vec3::splat(0.5);

    let i = Vec3::new(mod289(i.x), mod289(i.y), mod289(i.z));
    let p4 = |a: Vec4| Vec4::new(permute(a.x), permute(a.y), permute(a.z), permute(a.w));
    let p = p4(p4(p4(Vec4::splat(i.z) + Vec4::new(0.0, i1.z, i2.z, 1.0))
        + Vec4::splat(i.y)
        + Vec4::new(0.0, i1.y, i2.y, 1.0))
        + Vec4::splat(i.x)
        + Vec4::new(0.0, i1.x, i2.x, 1.0));

    let n_ = 1.0_f32 / 7.0;
    let ns = Vec3::new(n_ * 2.0 - 0.0, n_ * 0.5 - 1.0, n_ * 1.0 - 0.0);
    let j = p - 49.0 * (p * ns.z * ns.z).floor();
    let x_ = (j * ns.z).floor();
    let y_ = (j - 7.0 * x_).floor();
    let x = x_ * ns.x + Vec4::splat(ns.y);
    let y = y_ * ns.x + Vec4::splat(ns.y);
    let h = Vec4::ONE - x.abs() - y.abs();

    let b0 = Vec4::new(x.x, x.y, y.x, y.y);
    let b1 = Vec4::new(x.z, x.w, y.z, y.w);
    let s0 = b0.floor() * 2.0 + Vec4::ONE;
    let s1 = b1.floor() * 2.0 + Vec4::ONE;
    let sh = -Vec4::new(
        step(h.x, 0.0),
        step(h.y, 0.0),
        step(h.z, 0.0),
        step(h.w, 0.0),
    );

    let a0 = Vec4::new(b0.x, b0.z, b0.y, b0.w)
        + Vec4::new(s0.x, s0.z, s0.y, s0.w) * Vec4::new(sh.x, sh.x, sh.y, sh.y);
    let a1 = Vec4::new(b1.x, b1.z, b1.y, b1.w)
        + Vec4::new(s1.x, s1.z, s1.y, s1.w) * Vec4::new(sh.z, sh.z, sh.w, sh.w);

    let mut p0 = Vec3::new(a0.x, a0.y, h.x);
    let mut p1 = Vec3::new(a0.z, a0.w, h.y);
    let mut p2 = Vec3::new(a1.x, a1.y, h.z);
    let mut p3 = Vec3::new(a1.z, a1.w, h.w);
    let tis = |r: f32| 1.792_842_9 - 0.853_734_7 * r;
    p0 *= tis(p0.dot(p0));
    p1 *= tis(p1.dot(p1));
    p2 *= tis(p2.dot(p2));
    p3 *= tis(p3.dot(p3));

    let m = (Vec4::splat(0.6) - Vec4::new(x0.dot(x0), x1.dot(x1), x2.dot(x2), x3.dot(x3)))
        .max(Vec4::ZERO);
    let m = m * m;
    42.0 * (m * m).dot(Vec4::new(p0.dot(x0), p1.dot(x1), p2.dot(x2), p3.dot(x3)))
}

/// VQ `calcNoise`: two octaves of simplex.
pub fn calc_noise(p: Vec3) -> f32 {
    snoise3(p) + 0.5 * snoise3(p * 2.0)
}

/// Simplex noise that tiles with the given integer `period` in x and y, using
/// VQ's `caclNoiseSL` four-copy blend.
pub fn tiled_noise(x: f32, y: f32, z: f32, period: f32, f: impl Fn(Vec3) -> f32) -> f32 {
    let (u, v) = (x / period, y / period);
    f(Vec3::new(x, y, z)) * (1.0 - u) * (1.0 - v)
        + f(Vec3::new(x - period, y, z)) * u * (1.0 - v)
        + f(Vec3::new(x - period, y - period, z)) * u * v
        + f(Vec3::new(x, y - period, z)) * (1.0 - u) * v
}

/// Cheap deterministic hash -> [0, 1).
#[inline]
pub fn hash3(p: [i32; 3], seed: u32) -> [f32; 3] {
    let mut h = (p[0] as u32).wrapping_mul(0x8da6_b343)
        ^ (p[1] as u32).wrapping_mul(0xd816_3841)
        ^ (p[2] as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x9e37_79b9);
    let mut out = [0.0; 3];
    for o in &mut out {
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
        h = h.wrapping_mul(0x297a_2d39);
        h ^= h >> 15;
        *o = (h >> 8) as f32 / (1u32 << 24) as f32;
    }
    out
}

/// Tiling 3D Voronoi "centreness" volume, VQ's `cell2D` / `E_VW_VORO` bake.
///
/// Each texel holds `1 - 2·d1 / (d1 + d2)`: 1 at a cell's feature point, 0 on
/// the border between two cells. `size` texels per side, `cells` cells per side.
pub fn voronoi_volume(size: u32, cells: u32, seed: u32) -> Vec<u8> {
    let n = size as usize;
    let cell = size as f32 / cells as f32;
    let c = cells as i32;
    let mut points = vec![Vec3::ZERO; (cells * cells * cells) as usize];
    for z in 0..c {
        for y in 0..c {
            for x in 0..c {
                let h = hash3([x, y, z], seed);
                points[(x + y * c + z * c * c) as usize] =
                    (Vec3::new(x as f32, y as f32, z as f32) + Vec3::from(h) * 0.8 + 0.1) * cell;
            }
        }
    }
    let mut out = vec![0u8; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                let base = (p / cell).floor().as_ivec3();
                let (mut d1, mut d2) = (f32::MAX, f32::MAX);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let ci = base + bevy::math::IVec3::new(dx, dy, dz);
                            let wrapped = ci.rem_euclid(bevy::math::IVec3::splat(c));
                            let shift = (ci - wrapped).as_vec3() * cell;
                            let fp = points
                                [(wrapped.x + wrapped.y * c + wrapped.z * c * c) as usize]
                                + shift;
                            let d = p.distance(fp);
                            if d < d1 {
                                d2 = d1;
                                d1 = d;
                            } else if d < d2 {
                                d2 = d;
                            }
                        }
                    }
                }
                let grad = 1.0 - 2.0 * d1 / (d1 + d2).max(1e-6);
                out[x + y * n + z * n * n] = (grad.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Tiling 2D Voronoi centreness map (stand-in for VQ's missing `voro.bmp`,
/// which caps mountains into mesas).
pub fn voronoi_map(size: u32, cells: u32, seed: u32) -> Vec<f32> {
    let n = size as usize;
    let cell = size as f32 / cells as f32;
    let c = cells as i32;
    let mut out = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let p = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let base = (p / cell).floor().as_ivec2();
            let (mut d1, mut d2) = (f32::MAX, f32::MAX);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let ci = base + bevy::math::IVec2::new(dx, dy);
                    let w = ci.rem_euclid(bevy::math::IVec2::splat(c));
                    let h = hash3([w.x, w.y, 7], seed);
                    let fp = (ci.as_vec2() + Vec2::new(h[0], h[1]) * 0.8 + 0.1) * cell;
                    let d = p.distance(fp);
                    if d < d1 {
                        d2 = d1;
                        d1 = d;
                    } else if d < d2 {
                        d2 = d;
                    }
                }
            }
            out[x + y * n] = (1.0 - 2.0 * d1 / (d1 + d2).max(1e-6)).clamp(0.0, 1.0);
        }
    }
    out
}

/// Tiling ridged fBm in [0, 1] — a stand-in for one channel of VQ's real-world
/// heightmaps. `uv` in [0, 1).
pub fn ridged_fbm(uv: Vec2, base_freq: f32, octaves: u32, z: f32) -> f32 {
    let mut sum = 0.0;
    let mut norm = 0.0;
    let mut amp = 1.0;
    let mut freq = base_freq;
    let mut prev = 1.0;
    for o in 0..octaves {
        let n = tiled_noise(uv.x * freq, uv.y * freq, z + o as f32 * 17.0, freq, snoise3);
        let r = 1.0 - n.abs().min(1.0);
        let r = r * r * prev;
        prev = r.clamp(0.0, 1.0);
        sum += r * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}
