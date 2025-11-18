from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Float, Array

from ._geometry import HalfSpace, Sphere, Capsule, Heightmap, Box
from . import _utils


# --- HalfSpace Collision Implementations ---


def _halfspace_sphere_dist(
    halfspace_normal: Float[Array, "*batch 3"],
    halfspace_point: Float[Array, "*batch 3"],
    sphere_pos: Float[Array, "*batch 3"],
    sphere_radius: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between a halfspace boundary plane and sphere center, minus radius."""
    dist = (
        jnp.einsum("...i,...i->...", sphere_pos - halfspace_point, halfspace_normal)
        - sphere_radius
    )
    return dist


def halfspace_sphere(halfspace: HalfSpace, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculates distance between a halfspace and a sphere."""
    dist = _halfspace_sphere_dist(
        halfspace.normal,
        halfspace.pose.translation(),
        sphere.pose.translation(),
        sphere.radius,
    )
    return dist


def halfspace_capsule(halfspace: HalfSpace, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculates distance between halfspace and capsule (closest end)."""
    halfspace_normal = halfspace.normal
    halfspace_point = halfspace.pose.translation()
    cap_center = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    dist1 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center + segment_offset, cap_radius
    )
    dist2 = _halfspace_sphere_dist(
        halfspace_normal, halfspace_point, cap_center - segment_offset, cap_radius
    )
    final_dist = jnp.minimum(dist1, dist2)
    return final_dist


# --- Sphere/Capsule Collision Implementations ---


def _sphere_sphere_dist(
    pos1: Float[Array, "*batch 3"],
    radius1: Float[Array, "*batch"],
    pos2: Float[Array, "*batch 3"],
    radius2: Float[Array, "*batch"],
) -> Float[Array, "*batch"]:
    """Helper: Calculates distance between two spheres."""
    _, dist_center = _utils.normalize_with_norm(pos2 - pos1)
    dist = dist_center - (radius1 + radius2)
    return dist


def sphere_sphere(sphere1: Sphere, sphere2: Sphere) -> Float[Array, "*batch"]:
    """Calculate distance between two spheres."""
    dist = _sphere_sphere_dist(
        sphere1.pose.translation(),
        sphere1.radius,
        sphere2.pose.translation(),
        sphere2.radius,
    )
    return dist


def sphere_capsule(sphere: Sphere, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between sphere and capsule."""
    cap_pos = capsule.pose.translation()
    sphere_pos = sphere.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    cap_a = cap_pos - segment_offset
    cap_b = cap_pos + segment_offset
    pt_on_axis = _utils.closest_segment_point(cap_a, cap_b, sphere_pos)
    dist = _sphere_sphere_dist(sphere_pos, sphere.radius, pt_on_axis, capsule.radius)
    return dist


def capsule_capsule(capsule1: Capsule, capsule2: Capsule) -> Float[Array, "*batch"]:
    """Calculate distance between two capsules."""
    pos1 = capsule1.pose.translation()
    axis1 = capsule1.axis
    length1 = capsule1.height
    radius1 = capsule1.radius
    segment1_offset = axis1 * length1[..., None] / 2
    a1 = pos1 - segment1_offset
    b1 = pos1 + segment1_offset

    pos2 = capsule2.pose.translation()
    axis2 = capsule2.axis
    length2 = capsule2.height
    radius2 = capsule2.radius
    segment2_offset = axis2 * length2[..., None] / 2
    a2 = pos2 - segment2_offset
    b2 = pos2 + segment2_offset

    pt1_on_axis, pt2_on_axis = _utils.closest_segment_to_segment_points(a1, b1, a2, b2)
    dist = _sphere_sphere_dist(pt1_on_axis, radius1, pt2_on_axis, radius2)
    return dist


# --- Heightmap Collision Implementations ---


def heightmap_sphere(heightmap: Heightmap, sphere: Sphere) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and sphere.

    Approximation: Considers the heightmap point directly below the sphere center
    using bilinear interpolation and calculates vertical distance minus radius.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), sphere.get_batch_axes()
    )

    sphere_pos_w = sphere.pose.translation()
    sphere_radius = sphere.radius
    interpolated_local_z = heightmap._interpolate_height_at_coords(sphere_pos_w)
    sphere_pos_h = heightmap.pose.inverse().apply(sphere_pos_w)
    sphere_local_z = sphere_pos_h[..., 2]
    dist = sphere_local_z - interpolated_local_z - sphere_radius

    assert dist.shape == batch_axes
    return dist


def heightmap_capsule(heightmap: Heightmap, capsule: Capsule) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and capsule, by
    checking heightmap points below capsule endpoints.

    Note that this may miss collisions when capsule body intersects but endpoints are above heightmap!
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), capsule.get_batch_axes()
    )

    cap_pos_w = capsule.pose.translation()
    cap_radius = capsule.radius
    cap_axis_w = capsule.axis  # World frame axis
    segment_offset_w = cap_axis_w * capsule.height[..., None] / 2

    # Calculate world positions of the two end-sphere centers.
    p1_w = cap_pos_w + segment_offset_w
    p2_w = cap_pos_w - segment_offset_w

    # Interpolate heightmap surface height (local Z) below each end-sphere center.
    h_surf1_local = heightmap._interpolate_height_at_coords(p1_w)
    h_surf2_local = heightmap._interpolate_height_at_coords(p2_w)

    # Get end-sphere centers Z coordinates in heightmap's local frame.
    p1_h = heightmap.pose.inverse().apply(p1_w)
    p2_h = heightmap.pose.inverse().apply(p2_w)
    z1_local = p1_h[..., 2]
    z2_local = p2_h[..., 2]

    # Calculate vertical distance for each end sphere.
    dist1 = z1_local - h_surf1_local - cap_radius
    dist2 = z2_local - h_surf2_local - cap_radius

    # Return the minimum distance.
    min_dist = jnp.minimum(dist1, dist2)
    assert min_dist.shape == batch_axes
    return min_dist


def heightmap_halfspace(
    heightmap: Heightmap, halfspace: HalfSpace
) -> Float[Array, "*batch"]:
    """Calculate approximate distance between heightmap and halfspace.

    Approximation: Finds the minimum signed distance between any heightmap vertex
    and the halfspace plane.
    """
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )

    # Heightmap vertices in world frame.
    verts_local = heightmap._get_vertices_local()  # (*batch, N, 3), N=H*W
    verts_world = heightmap.pose.apply(verts_local)  # (*batch, N, 3)

    # Halfspace plane properties (world frame).
    hs_normal_w = halfspace.normal  # (*batch, 3)
    hs_point_w = halfspace.pose.translation()  # (*batch, 3)

    # Ensure batch dimensions are compatible for broadcasting.
    batch_axes = jnp.broadcast_shapes(
        heightmap.get_batch_axes(), halfspace.get_batch_axes()
    )
    # Expand dims for broadcasting against vertices.
    hs_normal_w = jnp.broadcast_to(hs_normal_w, batch_axes + (3,))[..., None, :]
    hs_point_w = jnp.broadcast_to(hs_point_w, batch_axes + (3,))[..., None, :]
    verts_world = jnp.broadcast_to(verts_world, batch_axes + verts_world.shape[-2:])

    # Calculate signed distance for each vertex to the plane:
    vertex_distances = jnp.einsum(
        "...vi,...i->...v", verts_world - hs_point_w, hs_normal_w.squeeze(-2)
    )

    # Find the minimum distance among all vertices.
    min_dist = jnp.min(vertex_distances, axis=-1)
    assert min_dist.shape == batch_axes
    return min_dist


# --- Box Collision Implementations ---


def _accumulate_axis_distances(face_dists: Float[Array, "*batch 6"]) -> Float[Array, "*batch"]:
    """Reduce six face distances to a single scalar distance.

    face_dists ordering: (px, nx, py, ny, pz, nz)
    For each axis we sum the positive parts of the two opposite-face distances
    (i.e. how far outside the box along that axis the other object is). The
    final distance is the Euclidean norm of the three axis contributions.
    """
    # positive contributions per face (separations)
    pos = jnp.maximum(face_dists, 0.0)
    dx = pos[..., 0] + pos[..., 1]
    dy = pos[..., 2] + pos[..., 3]
    dz = pos[..., 4] + pos[..., 5]
    sep = jnp.sqrt(dx * dx + dy * dy + dz * dz)

    # penetration case: if sep == 0 (no positive separation) we are inside the box
    # Compute per-axis penetration (most negative face value per axis)
    pen_x = jnp.minimum(face_dists[..., 0], face_dists[..., 1])
    pen_y = jnp.minimum(face_dists[..., 2], face_dists[..., 3])
    pen_z = jnp.minimum(face_dists[..., 4], face_dists[..., 5])
    pen_mag = -jnp.sqrt(pen_x * pen_x + pen_y * pen_y + pen_z * pen_z)

    # Choose separation when positive, otherwise penetration (negative)
    dist = jnp.where(sep > 0.0, sep, pen_mag)
    return dist


def box_sphere(box: Box, sphere: Sphere) -> Float[Array, "*batch"]:
    """Compute signed distance between an oriented box and a sphere.

    Uses the standard box SDF in the box's local frame. This yields a signed
    distance that grows (in absolute value) as penetration increases.
    """
    # Sphere center in box local frame
    sph_pos_w = sphere.pose.translation()
    sph_pos_b = box.pose.inverse().apply(sph_pos_w)

    hl = box.half_lengths
    q = jnp.abs(sph_pos_b) - hl
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    sdist_box = outside + inside

    dist = sdist_box - sphere.radius
    return dist


def box_capsule(box: Box, capsule: Capsule) -> Float[Array, "*batch"]:
    """Compute distance between box and capsule by approximating the capsule
    as a short series of spheres along its axis and taking the minimum.
    """
    # Approximate capsule by sampling spheres along its axis and taking the min.
    n_segs = 8

    cap_pos = capsule.pose.translation()
    cap_axis = capsule.axis
    segment_offset = cap_axis * capsule.height[..., None] / 2
    a = cap_pos - segment_offset
    b = cap_pos + segment_offset

    t = jnp.linspace(0.0, 1.0, n_segs)
    # centers shape: (n_segs, *batch_capsule, 3)
    centers = a[None, ...] * (1.0 - t)[:, None, None] + b[None, ...] * t[:, None, None]

    # radii shape: (n_segs, *batch_capsule)
    radii = jnp.broadcast_to(capsule.radius, centers.shape[:-1])

    # Build batched spheres
    spheres = Sphere.from_center_and_radius(center=centers, radius=radii)

    # Broadcast box to spheres batch axes and compute per-sphere distances
    box_bc = box.broadcast_to(spheres.get_batch_axes())
    dists = box_sphere(box_bc, spheres)
    dist = jnp.min(dists, axis=0)
    # Remove any accidental singleton dimensions introduced by sampling
    return jnp.squeeze(dist)


def box_halfspace(box: Box, halfspace: HalfSpace) -> Float[Array, "*batch"]:
    """Compute signed distance between box and a halfspace plane.

    We evaluate the halfspace plane signed distance at all eight box vertices
    and return the minimum value. This gives a penetration depth that grows
    (in absolute value) as the box penetrates the halfspace.
    """
    # Box vertices in local frame: combinations of +/- half_lengths
    hl = box.half_lengths
    # Create array of shape (8,3) with vertex signs
    signs = jnp.array(
        [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)]
    )
    verts_local = signs[None, ...] * hl[..., None, :]
    # verts_local shape: (*batch_box, 8, 3)
    verts_world = box.pose.apply(verts_local)

    hs_n = halfspace.normal
    hs_pt = halfspace.pose.translation()

    # Broadcast for einsum: ensure hs_n and hs_pt have a vertices axis
    hs_n_bc = jnp.broadcast_to(hs_n, verts_world.shape[:-1] + (3,))[..., None, :]
    hs_pt_bc = jnp.broadcast_to(hs_pt, verts_world.shape[:-1] + (3,))[..., None, :]

    vertex_distances = jnp.einsum("...vi,...i->...v", verts_world - hs_pt_bc, hs_n_bc.squeeze(-2))
    min_dist = jnp.min(vertex_distances, axis=-1)
    return min_dist


def box_heightmap(box: Box, heightmap: Heightmap) -> Float[Array, "*batch"]:
    """Compute approximate signed distance between box vertices and heightmap.

    We check the heightmap surface under each of the box's eight vertices
    (after transforming them into world then heightmap local frame) and
    return the minimum vertical signed distance (vertex_z - surface_z).
    """
    hl = box.half_lengths
    signs = jnp.array(
        [[sx, sy, sz] for sx in (1.0, -1.0) for sy in (1.0, -1.0) for sz in (1.0, -1.0)]
    )
    verts_local = signs[None, ...] * hl[..., None, :]
    verts_world = box.pose.apply(verts_local)

    # verts_world shape: (*batch_box, 8, 3)
    batch_axes = jnp.broadcast_shapes(box.get_batch_axes(), heightmap.get_batch_axes())
    verts_world = jnp.broadcast_to(verts_world, batch_axes + verts_world.shape[-2:])

    # Interpolate heightmap at each vertex world position: flatten verts for call
    flat_verts = verts_world.reshape(batch_axes + (-1, 3))
    # heightmap._interpolate_height_at_coords expects (*batch, 3) -> returns (*batch)
    interp = heightmap._interpolate_height_at_coords(flat_verts)

    # Reshape back to per-vertex and compute vertex z in heightmap local frame
    interp = interp.reshape(batch_axes + (verts_world.shape[-2],))
    verts_h = heightmap.pose.inverse().apply(verts_world)
    vert_local_z = verts_h[..., 2]

    # Signed vertical distance for each vertex: vertex_z - surface_z
    vert_dists = vert_local_z - interp
    min_dist = jnp.min(vert_dists, axis=-1)
    return min_dist
