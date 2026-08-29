"""Core calculations for the Janus-particle and single-particle models.

This module contains the functions required for lattice generation, Voronoi
tessellation, phase partitioning, contact detection, and percolation analysis.
Production calculations and 2D/3D visualization are run from the run script,
whereas numerical and visual checks are run from the validation script.
"""

import numpy as np

from scipy.spatial import Voronoi

from scipy.spatial import ConvexHull

from scipy.spatial import cKDTree

from scipy.spatial import QhullError 

import networkx as nx



def Generate_lattice(a=1.0, nx=3, ny=3, nz=3, s_sets='nx'):

    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]])

    coords = []

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                origin = np.array([i, j, k]) * a

                for f in frac:

                    coords.append(origin + f * a)

    return np.vstack(coords)



def Generate_FCC(a=1.0, nx=3, ny=3, nz=3, s_sets='nx'):

    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])

    coords = []

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                origin = np.array([i, j, k]) * a

                for f in frac:

                    coords.append(origin + f * a)

    return np.vstack(coords)



def Generate_BCC(a=1.0, nx=3, ny=3, nz=3):

    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    coords = []

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                origin = np.array([i, j, k]) * a

                for f in frac:

                    coords.append(origin + f * a)

    return np.vstack(coords)



def Generate_HCP(a=1.0, nx=3, ny=3, nz=3, c_over_a=np.sqrt(8.0 / 3.0)):

    if a <= 0:

        raise ValueError('a must be greater than zero.')

    if nx < 1 or ny < 1 or nz < 1:

        raise ValueError('nx, ny, and nz must all be at least 1.')

    if c_over_a <= 0:

        raise ValueError('c_over_a must be greater than zero.')

    c = c_over_a * a

    cell_lengths = np.array([a, np.sqrt(3.0) * a, c], dtype=float)

    frac = np.array([

        [0.0, 0.0, 0.0],

        [0.5, 0.5, 0.0],

        [0.5, 1.0 / 6.0, 0.5],

        [0.0, 2.0 / 3.0, 0.5],

    ])

    coords = []

    for i in range(nx):

        for j in range(ny):

            for k in range(nz):

                origin = np.array([i, j, k], dtype=float) * cell_lengths

                for f in frac:

                    coords.append(origin + f * cell_lengths)

    return np.vstack(coords)



def generate_points(lattice_type='lattice', a=1.0, nx=3, ny=3, nz=3):

    lattice_type = lattice_type.lower()

    if lattice_type == 'lattice':

        return Generate_lattice(a=a, nx=nx, ny=ny, nz=nz)

    elif lattice_type == 'fcc':

        return Generate_FCC(a=a, nx=nx, ny=ny, nz=nz)

    elif lattice_type == 'bcc':

        return Generate_BCC(a=a, nx=nx, ny=ny, nz=nz)

    elif lattice_type == 'hcp':

        return Generate_HCP(a=a, nx=nx, ny=ny, nz=nz)

    else:

        raise ValueError("lattice_type must be 'lattice', 'fcc', 'bcc', or 'hcp'.")



def get_lattice_box_lengths(lattice_type='lattice', a=1.0, nx=3, ny=3, nz=3, hcp_c_over_a=np.sqrt(8.0 / 3.0)):

    lattice_type = lattice_type.lower()

    if a <= 0:

        raise ValueError('a must be greater than zero.')

    if nx < 1 or ny < 1 or nz < 1:

        raise ValueError('nx, ny, and nz must all be at least 1.')

    if lattice_type in ['lattice', 'fcc', 'bcc']:

        return np.array([nx * a, ny * a, nz * a], dtype=float)

    if lattice_type == 'hcp':

        if hcp_c_over_a <= 0:

            raise ValueError('hcp_c_over_a must be greater than zero.')

        return np.array([

            nx * a,

            ny * np.sqrt(3.0) * a,

            nz * hcp_c_over_a * a,

        ], dtype=float)

    raise ValueError("lattice_type must be 'lattice', 'fcc', 'bcc', or 'hcp'.")



def make_periodic_xy_images(points, box_lengths):

    points = np.asarray(points, dtype=float)

    box_lengths = np.asarray(box_lengths, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:

        raise ValueError('points must be an array with shape (N, 3).')

    if box_lengths.shape != (3,):

        raise ValueError('box_lengths must be an array of length 3.')

    if np.any(box_lengths <= 0):

        raise ValueError('Every component of box_lengths must be greater than zero.')

    shifts_xy = [(0, 0)]

    shifts_xy.extend(

        (sx, sy)

        for sx in (-1, 0, 1)

        for sy in (-1, 0, 1)

        if (sx, sy) != (0, 0)

    )

    point_blocks = []

    source_blocks = []

    shift_blocks = []

    source_indices = np.arange(len(points), dtype=int)

    for sx, sy in shifts_xy:

        shift = np.array([

            sx * box_lengths[0],

            sy * box_lengths[1],

            0.0,

        ])

        point_blocks.append(points + shift)

        source_blocks.append(source_indices)

        shift_blocks.append(np.repeat(shift.reshape(1, 3), len(points), axis=0))

    periodic_points = np.vstack(point_blocks)

    periodic_sources = np.concatenate(source_blocks)

    periodic_shifts = np.vstack(shift_blocks)

    central_indices = np.arange(len(points), dtype=int)

    return periodic_points, periodic_sources, periodic_shifts, central_indices



def get_cell_vertices(vor, i_cell):

    region_index = vor.point_region[i_cell]

    region = vor.regions[region_index]

    if -1 in region or len(region) == 0:

        return None

    verts = vor.vertices[region]

    return verts



def compute_cell_edges_from_hull(hull):

    edges = set()

    for tri in hull.simplices:

        i, j, k = tri

        for a, b in [(i, j), (j, k), (k, i)]:

            if a > b:

                a, b = (b, a)

            edges.add((a, b))

    result = list(edges)

    return result



def volume_A_for_threshold(verts, center, normal, threshold, hull_total=None, edges=None):

    verts = np.asarray(verts, dtype=float)

    center = np.asarray(center, dtype=float)

    n = np.asarray(normal, dtype=float)

    n = n / np.linalg.norm(n)

    if hull_total is None:

        hull_total = ConvexHull(verts)

    if edges is None:

        edges = compute_cell_edges_from_hull(hull_total)

    s = (verts - center) @ n

    d = s - threshold

    inside_mask = d <= 0.0

    inside_points = verts[inside_mask]

    new_points = []

    if len(inside_points) > 0:

        new_points.extend(list(inside_points))

    for i, j in edges:

        di, dj = (d[i], d[j])

        if di * dj < 0.0:

            t = di / (di - dj)

            p = verts[i] + t * (verts[j] - verts[i])

            new_points.append(p)

    if len(new_points) < 4:

        return 0.0

    new_points = np.asarray(new_points)

    new_points = np.unique(new_points, axis=0)

    if len(new_points) < 4:

        return 0.0

    hull_A = ConvexHull(new_points)

    vol_A = hull_A.volume

    return vol_A



def find_threshold_for_fraction(verts, center, normal, target_frac_A, tol=0.001, max_iter=40):

    if not np.isfinite(target_frac_A) or not 0.0 <= target_frac_A <= 1.0:

        raise ValueError('target_frac_A must be finite and between 0 and 1.')

    if not np.isfinite(tol) or tol <= 0.0:

        raise ValueError('tol must be finite and greater than zero.')

    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:

        raise ValueError('max_iter must be an integer of at least 1.')

    verts = np.asarray(verts, dtype=float)

    center = np.asarray(center, dtype=float)

    n = np.asarray(normal, dtype=float)

    n = n / np.linalg.norm(n)

    hull_total = ConvexHull(verts)

    vol_total = hull_total.volume

    if vol_total == 0.0:

        raise ValueError('The cell volume is zero; its vertices may be degenerate.')

    edges = compute_cell_edges_from_hull(hull_total)

    s = (verts - center) @ n

    s_min, s_max = (s.min(), s.max())

    if target_frac_A <= 0.0:

        return (s_min - 1e-06, 0.0, vol_total)

    if target_frac_A >= 1.0:

        return (s_max + 1e-06, vol_total, 0.0)

    low, high = (s_min, s_max)

    for _ in range(max_iter):

        mid = 0.5 * (low + high)

        vol_mid = volume_A_for_threshold(verts, center, n, mid, hull_total=hull_total, edges=edges)

        frac_mid = vol_mid / vol_total

        if abs(frac_mid - target_frac_A) < tol:

            vol_A = vol_mid

            vol_B = vol_total - vol_A

            return (mid, vol_A, vol_B)

        if frac_mid > target_frac_A:

            high = mid

        else:

            low = mid

    vol_A = vol_mid

    vol_B = vol_total - vol_A

    

    achieved_frac = vol_A / vol_total

    fraction_error = abs(achieved_frac - target_frac_A)

    

    if fraction_error > tol:

        raise RuntimeError(

            f'Volume-fraction search did not converge: '

            f'target={target_frac_A}, '

            f'achieved={achieved_frac}, '

            f'error={fraction_error}, '

            f'tol={tol}, '

            f'max_iter={max_iter}'

        )



    return mid, vol_A, vol_B



def random_unit_vector(rng):

    while True:

        v = rng.normal(size=3)

        norm = np.linalg.norm(v)

        if norm > 1e-08:

            return v / norm



def order_coplanar_points(points):

    points = np.asarray(points, dtype=float)

    if len(points) <= 2:

        return points

    centroid = points.mean(axis=0)

    X = points - centroid

    _, _, vh = np.linalg.svd(X, full_matrices=False)

    u = vh[0]

    v = vh[1]

    coords_2d = np.column_stack([X @ u, X @ v])

    angles = np.arctan2(coords_2d[:, 1], coords_2d[:, 0])

    order = np.argsort(angles)

    return points[order]

def all_cells_AB_volumes(pts, target_frac_A=0.6, tol=0.001, max_iter=40, seed=None):

    pts = np.asarray(pts, dtype=float)

    vor = Voronoi(pts)

    rng = np.random.default_rng(seed)

    N = len(pts)

    cell_indices = []

    normals = []

    thresholds = []

    vols_A = []

    vols_B = []

    for i in range(N):

        verts = get_cell_vertices(vor, i)

        if verts is None or len(verts) < 4:

            continue

        center = pts[i]

        n = random_unit_vector(rng)

        try:

            threshold, vol_A, vol_B = find_threshold_for_fraction(

                verts,

                center,

                n,

                target_frac_A,

                tol=tol,

                max_iter=max_iter,

            )

        except Exception as error:

            raise RuntimeError(

                f'Threshold search failed for cell {i}: '

                f'cause={type(error).__name__}: {error}'

            ) from error

        cell_indices.append(i)

        normals.append(n)

        thresholds.append(threshold)

        vols_A.append(vol_A)

        vols_B.append(vol_B)

    return (cell_indices, normals, thresholds, vols_A, vols_B, vor)



def polygon_area_3d(points):

    points = np.asarray(points, dtype=float)

    if len(points) < 3:

        return 0.0

    points = order_coplanar_points(points)

    cross_sum = np.zeros(3, dtype=float)

    for i in range(len(points)):

        p1 = points[i]

        p2 = points[(i + 1) % len(points)]

        cross_sum += np.cross(p1, p2)

    return 0.5 * np.linalg.norm(cross_sum)



def remove_adjacent_duplicate_points(points, tol=1e-10):

    points = np.asarray(points, dtype=float)

    if len(points) == 0:

        return np.empty((0, 3), dtype=float)

    cleaned = [points[0]]

    for p in points[1:]:

        if np.linalg.norm(p - cleaned[-1]) > tol:

            cleaned.append(p)

    if len(cleaned) >= 2 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tol:

        cleaned.pop()

    return np.asarray(cleaned, dtype=float)



def clip_convex_polygon(polygon, signed_distance, eps=1e-12):

    polygon = np.asarray(polygon, dtype=float)

    if len(polygon) < 3:

        return np.empty((0, 3), dtype=float)

    polygon = order_coplanar_points(polygon)

    output = []

    for i in range(len(polygon)):

        p1 = polygon[i]

        p2 = polygon[(i + 1) % len(polygon)]

        d1 = float(signed_distance(p1))

        d2 = float(signed_distance(p2))

        inside1 = d1 <= eps

        inside2 = d2 <= eps

        if inside1 and inside2:

            output.append(p2)

        elif inside1 and (not inside2):

            denominator = d1 - d2

            if abs(denominator) > eps:

                t = d1 / denominator

                intersection = p1 + t * (p2 - p1)

                output.append(intersection)

        elif not inside1 and inside2:

            denominator = d1 - d2

            if abs(denominator) > eps:

                t = d1 / denominator

                intersection = p1 + t * (p2 - p1)

                output.append(intersection)

            output.append(p2)

    if len(output) < 3:

        return np.empty((0, 3), dtype=float)

    output = remove_adjacent_duplicate_points(output)

    if len(output) < 3:

        return np.empty((0, 3), dtype=float)

    return order_coplanar_points(output)



def clip_polygon_to_z_slab(polygon, z_bottom, z_top):

    polygon = np.asarray(polygon, dtype=float)

    polygon = clip_convex_polygon(polygon, signed_distance=lambda p: z_bottom - p[2])

    if len(polygon) < 3:

        return np.empty((0, 3), dtype=float)

    polygon = clip_convex_polygon(polygon, signed_distance=lambda p: p[2] - z_top)

    return polygon



def shared_face_contact_fractions(face_vertices, center_i, normal_i, threshold_i, center_j, normal_j, threshold_j, z_bottom=None, z_top=None, area_tol=1e-12):

    face = np.asarray(face_vertices, dtype=float)

    if len(face) < 3:

        return (0.0, 0.0, 0.0, 0.0, 0.0)

    face = order_coplanar_points(face)

    if z_bottom is not None and z_top is not None:

        face = clip_polygon_to_z_slab(face, z_bottom=z_bottom, z_top=z_top)

    if len(face) < 3:

        return (0.0, 0.0, 0.0, 0.0, 0.0)

    face_area = polygon_area_3d(face)

    if face_area <= area_tol:

        return (0.0, 0.0, 0.0, 0.0, face_area)

    center_i = np.asarray(center_i, dtype=float)

    center_j = np.asarray(center_j, dtype=float)

    normal_i = np.asarray(normal_i, dtype=float)

    normal_j = np.asarray(normal_j, dtype=float)

    normal_i = normal_i / np.linalg.norm(normal_i)

    normal_j = normal_j / np.linalg.norm(normal_j)

    polygon_AA = clip_convex_polygon(face, signed_distance=lambda p: np.dot(p - center_i, normal_i) - threshold_i)

    if len(polygon_AA) >= 3:

        polygon_AA = clip_convex_polygon(polygon_AA, signed_distance=lambda p: np.dot(p - center_j, normal_j) - threshold_j)

    area_AA = polygon_area_3d(polygon_AA)

    polygon_BB = clip_convex_polygon(face, signed_distance=lambda p: threshold_i - np.dot(p - center_i, normal_i))

    if len(polygon_BB) >= 3:

        polygon_BB = clip_convex_polygon(polygon_BB, signed_distance=lambda p: threshold_j - np.dot(p - center_j, normal_j))

    area_BB = polygon_area_3d(polygon_BB)

    if area_AA <= area_tol:

        area_AA = 0.0

    if area_BB <= area_tol:

        area_BB = 0.0

    frac_AA = area_AA / face_area

    frac_BB = area_BB / face_area

    return (frac_AA, frac_BB, area_AA, area_BB, face_area)



def intersect_cell_with_z_plane(verts, z_plane, tol=1e-10):

    verts = np.asarray(verts, dtype=float)

    if len(verts) < 4:

        return np.empty((0, 3), dtype=float)

    hull = ConvexHull(verts)

    edges = compute_cell_edges_from_hull(hull)

    section_points = []

    for p in verts:

        if abs(p[2] - z_plane) <= tol:

            section_points.append(p)

    for i, j in edges:

        p1 = verts[i]

        p2 = verts[j]

        d1 = p1[2] - z_plane

        d2 = p2[2] - z_plane

        if d1 * d2 < -tol ** 2:

            t = d1 / (d1 - d2)

            intersection = p1 + t * (p2 - p1)

            section_points.append(intersection)

    if len(section_points) < 3:

        return np.empty((0, 3), dtype=float)

    section_points = np.unique(np.round(np.asarray(section_points, dtype=float), decimals=12), axis=0)

    if len(section_points) < 3:

        return np.empty((0, 3), dtype=float)

    centroid = section_points.mean(axis=0)

    centered = section_points - centroid

    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    axis_u = vh[0]

    axis_v = vh[1]

    points_2d = np.column_stack([centered @ axis_u, centered @ axis_v])

    # Stop the calculation if Qhull cannot construct the cell cross-section.
    try:

        hull_2d = ConvexHull(points_2d)

    except QhullError as error:

        raise RuntimeError(

            f'Failed to compute the convex hull of the cell cross-section at z={z_plane}.'

        ) from error

    section_polygon = section_points[hull_2d.vertices]

    if len(section_polygon) < 3:

        return np.empty((0, 3), dtype=float)

    section_polygon = order_coplanar_points(section_polygon)

    if polygon_area_3d(section_polygon) <= tol:

        return np.empty((0, 3), dtype=float)

    return section_polygon



def cell_phase_fractions_on_z_plane(verts, center, normal, threshold, z_plane, area_tol=1e-12):

    section = intersect_cell_with_z_plane(verts, z_plane=z_plane)

    if len(section) < 3:

        return (0.0, 0.0, 0.0)

    section_area = polygon_area_3d(section)

    if section_area <= area_tol:

        return (0.0, 0.0, section_area)

    center = np.asarray(center, dtype=float)

    normal = np.asarray(normal, dtype=float)

    normal = normal / np.linalg.norm(normal)

    section_A = clip_convex_polygon(section, signed_distance=lambda p: np.dot(p - center, normal) - threshold)

    section_B = clip_convex_polygon(section, signed_distance=lambda p: threshold - np.dot(p - center, normal))

    area_A = polygon_area_3d(section_A)

    area_B = polygon_area_3d(section_B)

    if area_A <= area_tol:

        area_A = 0.0

    if area_B <= area_tol:

        area_B = 0.0

    frac_A = area_A / section_area

    frac_B = area_B / section_area

    return (frac_A, frac_B, section_area)



def _unique_points(points, tol=1e-10):

    unique = []

    for point in np.asarray(points, dtype=float):

        if not any(np.linalg.norm(point - existing) <= tol for existing in unique):

            unique.append(point)

    if len(unique) == 0:

        return np.empty((0, 3), dtype=float)

    return np.asarray(unique, dtype=float)


def clip_convex_polyhedron_by_halfspace(vertices, signed_distance, eps=1e-10):
    """Clip a convex polyhedron to the region signed_distance(point) <= 0."""

    vertices = _unique_points(vertices, tol=eps)

    if len(vertices) < 4:

        return np.empty((0, 3), dtype=float)

    try:

        hull = ConvexHull(vertices)

    except QhullError:

        return np.empty((0, 3), dtype=float)

    distances = np.asarray([signed_distance(point) for point in vertices], dtype=float)

    if np.all(distances <= eps):

        return vertices

    if np.all(distances > eps):

        return np.empty((0, 3), dtype=float)

    clipped_points = [vertices[i] for i in range(len(vertices)) if distances[i] <= eps]

    for i, j in compute_cell_edges_from_hull(hull):

        distance_i = distances[i]

        distance_j = distances[j]

        if (distance_i < -eps and distance_j > eps) or (distance_i > eps and distance_j < -eps):

            t = distance_i / (distance_i - distance_j)

            clipped_points.append(vertices[i] + t * (vertices[j] - vertices[i]))

    clipped_points = _unique_points(clipped_points, tol=eps)

    if len(clipped_points) < 4:

        return np.empty((0, 3), dtype=float)

    return clipped_points


def clip_convex_polyhedron_to_z_slab(vertices, z_bottom, z_top, eps=1e-10):
    """Return vertices of a convex polyhedron restricted to z_bottom <= z <= z_top."""

    clipped = clip_convex_polyhedron_by_halfspace(

        vertices,

        signed_distance=lambda point: z_bottom - point[2],

        eps=eps,

    )

    if len(clipped) < 4:

        return np.empty((0, 3), dtype=float)

    return clip_convex_polyhedron_by_halfspace(

        clipped,

        signed_distance=lambda point: point[2] - z_top,

        eps=eps,

    )


def convex_polyhedron_volume(vertices):

    vertices = _unique_points(vertices)

    if len(vertices) < 4:

        return 0.0

    try:

        return float(ConvexHull(vertices).volume)

    except QhullError:

        return 0.0


def get_evaluation_slab_geometry(

    prepared_geometry,

    cell_vertices,

    valid_indices,

    z_bottom,

    z_top,

):
    """Cache the z-clipped cell geometry because it is unchanged between trials."""

    cache = prepared_geometry.get('_evaluation_slab_geometry')

    if (

        cache is not None

        and np.isclose(cache['z_bottom'], z_bottom)

        and np.isclose(cache['z_top'], z_top)

        and len(cache['vertices']) == len(cell_vertices)

    ):

        return cache['vertices'], cache['volumes']

    slab_vertices = [np.empty((0, 3), dtype=float) for _ in cell_vertices]

    slab_volumes = np.zeros(len(cell_vertices), dtype=float)

    for i in valid_indices:

        slab_vertices[i] = clip_convex_polyhedron_to_z_slab(

            cell_vertices[i],

            z_bottom=z_bottom,

            z_top=z_top,

        )

        slab_volumes[i] = convex_polyhedron_volume(slab_vertices[i])

    prepared_geometry['_evaluation_slab_geometry'] = {

        'z_bottom': float(z_bottom),

        'z_top': float(z_top),

        'vertices': slab_vertices,

        'volumes': slab_volumes,

    }

    return slab_vertices, slab_volumes


def spanning_component_volume_fraction(G, bottom_nodes, top_nodes, node_volumes):
    """Fraction of phase volume belonging to components spanning bottom to top."""

    bottom_nodes = set(bottom_nodes)

    top_nodes = set(top_nodes)

    total_volume = float(sum(max(0.0, float(node_volumes[node])) for node in G.nodes))

    if total_volume <= 0.0 or len(bottom_nodes) == 0 or len(top_nodes) == 0:

        return 0.0

    spanning_volume = 0.0

    for component in nx.connected_components(G):

        component = set(component)

        if (component & bottom_nodes) and (component & top_nodes):

            spanning_volume += sum(max(0.0, float(node_volumes[node])) for node in component)

    return float(np.clip(spanning_volume / total_volume, 0.0, 1.0))



def prepare_voronoi_geometry(points, periodic_xy=False, box_lengths=None):

    points = np.asarray(points, dtype=float)

    if periodic_xy:

        if box_lengths is None:

            raise ValueError('box_lengths is required when periodic_xy=True.')

        vor_points, source_indices, image_shifts, central_indices = make_periodic_xy_images(

            points,

            box_lengths=box_lengths,

        )

    else:

        vor_points = points

        source_indices = np.arange(len(points), dtype=int)

        image_shifts = np.zeros_like(points)

        central_indices = np.arange(len(points), dtype=int)

    vor = Voronoi(vor_points)

    return {

        'points': points.copy(),

        'periodic_xy': bool(periodic_xy),

        'box_lengths': None if box_lengths is None else np.asarray(box_lengths, dtype=float).copy(),

        'vor_points': vor_points,

        'source_indices': source_indices,

        'image_shifts': image_shifts,

        'central_indices': central_indices,

        'vor': vor,

    }



def janus_voronoi_percolation_random(points, target_frac_A=0.5, tol=0.001, max_iter=40, seed=None, contact_method='shared_face', min_contact_area=1e-6, min_boundary_area=0.0, periodic_xy=False, box_lengths=None, prepared_geometry=None):

    if not np.isfinite(target_frac_A) or not 0.0 <= target_frac_A <= 1.0:

        raise ValueError('target_frac_A must be finite and between 0 and 1.')

    if not np.isfinite(tol) or tol <= 0.0:

        raise ValueError('tol must be finite and greater than zero.')

    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:

        raise ValueError('max_iter must be an integer of at least 1.')

    if not np.isfinite(min_contact_area) or min_contact_area < 0.0:

        raise ValueError('min_contact_area must be finite and non-negative.')

    if not np.isfinite(min_boundary_area) or min_boundary_area < 0.0:

        raise ValueError('min_boundary_area must be finite and non-negative.')

    points = np.asarray(points, dtype=float)

    if prepared_geometry is None:

        prepared_geometry = prepare_voronoi_geometry(

            points,

            periodic_xy=periodic_xy,

            box_lengths=box_lengths,

        )

    else:

        prepared_points = np.asarray(prepared_geometry['points'], dtype=float)

        if prepared_points.shape != points.shape or not np.allclose(prepared_points, points):

            raise ValueError('prepared_geometry must be created from the same points.')

        if bool(prepared_geometry['periodic_xy']) != bool(periodic_xy):

            raise ValueError('prepared_geometry and periodic_xy do not match.')

        if periodic_xy:

            if box_lengths is None:

                raise ValueError('box_lengths is required when periodic_xy=True.')

            if not np.allclose(prepared_geometry['box_lengths'], np.asarray(box_lengths, dtype=float)):

                raise ValueError('prepared_geometry and box_lengths do not match.')

    vor_points = prepared_geometry['vor_points']

    source_indices = prepared_geometry['source_indices']

    image_shifts = prepared_geometry['image_shifts']

    central_indices = prepared_geometry['central_indices']

    vor = prepared_geometry['vor']

    rng = np.random.default_rng(seed)

    N = len(points)

    centers = [None] * N

    cell_vertices = [None] * N

    normals = [None] * N

    thresholds = [None] * N

    volumes_A = np.zeros(N, dtype=float)

    volumes_B = np.zeros(N, dtype=float)

    valid = np.zeros(N, dtype=bool)

    for i in range(N):

        extended_i = central_indices[i]

        verts = get_cell_vertices(vor, extended_i)

        if verts is None or len(verts) < 4:

            continue

        center = points[i]

        normal = random_unit_vector(rng)

        try:

            threshold, vol_A, vol_B = find_threshold_for_fraction(verts, center, normal, target_frac_A, tol=tol, max_iter=max_iter)

        except Exception as e:

            raise RuntimeError(

                f'Janus-interface calculation failed for finite Voronoi cell {i}.'

            ) from e

        centers[i] = center

        cell_vertices[i] = verts

        normals[i] = normal

        thresholds[i] = threshold

        volumes_A[i] = vol_A

        volumes_B[i] = vol_B

        valid[i] = True

    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:

        raise RuntimeError(

            'No finite Voronoi cells were available for Janus percolation analysis.'

        )

    valid_z = points[valid_indices, 2]

    z_bottom = float(valid_z.min())

    z_top = float(valid_z.max())

    if np.isclose(z_bottom, z_top):

        raise ValueError('The domain is too thin in z; increase nz.')

    evaluated_volumes_A = np.zeros(N, dtype=float)

    evaluated_volumes_B = np.zeros(N, dtype=float)

    slab_vertices_by_cell, slab_volumes = get_evaluation_slab_geometry(

        prepared_geometry,

        cell_vertices,

        valid_indices,

        z_bottom,

        z_top,

    )

    for i in valid_indices:

        slab_vertices = slab_vertices_by_cell[i]

        if len(slab_vertices) < 4:

            continue

        phase_A_vertices = clip_convex_polyhedron_by_halfspace(

            slab_vertices,

            signed_distance=lambda point, index=i: (

                np.dot(point - centers[index], normals[index]) - thresholds[index]

            ),

        )

        evaluated_volumes_A[i] = convex_polyhedron_volume(phase_A_vertices)

        evaluated_volumes_B[i] = max(0.0, slab_volumes[i] - evaluated_volumes_A[i])

    G_A = nx.Graph()

    G_B = nx.Graph()

    volume_tol = 1e-12

    numerical_area_tol = 1e-10

    contact_area_tol = max(numerical_area_tol, min_contact_area)

    boundary_area_tol = max(numerical_area_tol, min_boundary_area)

    A_nodes = [i for i in valid_indices if evaluated_volumes_A[i] > volume_tol]

    B_nodes = [i for i in valid_indices if evaluated_volumes_B[i] > volume_tol]

    G_A.add_nodes_from(A_nodes)

    G_B.add_nodes_from(B_nodes)

    contact_method = contact_method.lower()

    if contact_method == 'center':

        for extended_i, extended_j in vor.ridge_points:

            if periodic_xy and extended_i >= N and extended_j >= N:

                continue

            i = int(source_indices[extended_i])

            j = int(source_indices[extended_j])

            if i == j or not (valid[i] and valid[j]):

                continue

            ci = vor_points[extended_i]

            cj = vor_points[extended_j]

            s_ji = np.dot(cj - ci, normals[i])

            s_ij = np.dot(ci - cj, normals[j])

            comp_j_from_i = 'A' if s_ji <= thresholds[i] else 'B'

            comp_i_from_j = 'A' if s_ij <= thresholds[j] else 'B'

            if comp_j_from_i == 'A' and comp_i_from_j == 'A' and (i in G_A) and (j in G_A):

                G_A.add_edge(i, j)

            if comp_j_from_i == 'B' and comp_i_from_j == 'B' and (i in G_B) and (j in G_B):

                G_B.add_edge(i, j)

    elif contact_method in ['shared_face', 'exact_shared_face']:

        for (extended_i, extended_j), ridge in zip(vor.ridge_points, vor.ridge_vertices):

            if periodic_xy and extended_i >= N and extended_j >= N:

                continue

            i = int(source_indices[extended_i])

            j = int(source_indices[extended_j])

            if i == j or not (valid[i] and valid[j]):

                continue

            if -1 in ridge or len(ridge) < 3:

                continue

            face_vertices = vor.vertices[ridge]

            center_i = vor_points[extended_i]

            center_j = vor_points[extended_j]

            frac_AA, frac_BB, area_AA, area_BB, face_area = shared_face_contact_fractions(face_vertices, center_i, normals[i], thresholds[i], center_j, normals[j], thresholds[j], z_bottom=z_bottom, z_top=z_top)

            if area_AA > contact_area_tol and i in G_A and (j in G_A):

                G_A.add_edge(i, j, contact_area=area_AA)

            if area_BB > contact_area_tol and i in G_B and (j in G_B):

                G_B.add_edge(i, j, contact_area=area_BB)

    else:

        raise ValueError("contact_method must be 'center', 'shared_face', or 'exact_shared_face'.")

    bottom_A_nodes = set()

    top_A_nodes = set()

    bottom_B_nodes = set()

    top_B_nodes = set()

    for i in valid_indices:

        verts = cell_vertices[i]

        center = centers[i]

        normal = normals[i]

        threshold = thresholds[i]

        frac_A_bottom, frac_B_bottom, bottom_area = cell_phase_fractions_on_z_plane(verts, center, normal, threshold, z_plane=z_bottom)

        area_A_bottom = frac_A_bottom * bottom_area

        area_B_bottom = frac_B_bottom * bottom_area

        if area_A_bottom > boundary_area_tol and i in G_A:

            bottom_A_nodes.add(i)

        if area_B_bottom > boundary_area_tol and i in G_B:

            bottom_B_nodes.add(i)

        frac_A_top, frac_B_top, top_area = cell_phase_fractions_on_z_plane(verts, center, normal, threshold, z_plane=z_top)

        area_A_top = frac_A_top * top_area

        area_B_top = frac_B_top * top_area

        if area_A_top > boundary_area_tol and i in G_A:

            top_A_nodes.add(i)

        if area_B_top > boundary_area_tol and i in G_B:

            top_B_nodes.add(i)

    spanning_volume_fraction_A = spanning_component_volume_fraction(

        G_A,

        bottom_nodes=bottom_A_nodes,

        top_nodes=top_A_nodes,

        node_volumes=evaluated_volumes_A,

    )

    spanning_volume_fraction_B = spanning_component_volume_fraction(

        G_B,

        bottom_nodes=bottom_B_nodes,

        top_nodes=top_B_nodes,

        node_volumes=evaluated_volumes_B,

    )

    return (

        spanning_volume_fraction_A,

        spanning_volume_fraction_B,

        normals,

        thresholds,

    )



def single_particle_percolation_random(

    points,

    target_frac_A=0.5,

    seed=None,

    min_contact_area=1e-6,

    min_boundary_area=0.0,

    periodic_xy=False,

    box_lengths=None,

    prepared_geometry=None,

):



    points = np.asarray(points, dtype=float)

    if not 0.0 <= target_frac_A <= 1.0:

        raise ValueError('target_frac_A must be between 0 and 1.')

    if min_contact_area < 0.0 or min_boundary_area < 0.0:

        raise ValueError('min_contact_area and min_boundary_area must be non-negative.')

    if prepared_geometry is None:

        prepared_geometry = prepare_voronoi_geometry(

            points,

            periodic_xy=periodic_xy,

            box_lengths=box_lengths,

        )

    else:

        prepared_points = np.asarray(prepared_geometry['points'], dtype=float)

        if prepared_points.shape != points.shape or not np.allclose(prepared_points, points):

            raise ValueError('prepared_geometry must be created from the same points.')

        if bool(prepared_geometry['periodic_xy']) != bool(periodic_xy):

            raise ValueError('prepared_geometry and periodic_xy do not match.')

        if periodic_xy:

            if box_lengths is None:

                raise ValueError('box_lengths is required when periodic_xy=True.')

            if not np.allclose(prepared_geometry['box_lengths'], np.asarray(box_lengths, dtype=float)):

                raise ValueError('prepared_geometry and box_lengths do not match.')



    vor_points = prepared_geometry['vor_points']

    source_indices = prepared_geometry['source_indices']

    central_indices = prepared_geometry['central_indices']

    vor = prepared_geometry['vor']

    N = len(points)



    cell_vertices = [None] * N

    valid = np.zeros(N, dtype=bool)

    for i in range(N):

        verts = get_cell_vertices(vor, central_indices[i])

        if verts is None or len(verts) < 4:

            continue

        cell_vertices[i] = verts

        valid[i] = True

    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:

        raise RuntimeError(

            'No finite Voronoi cells were available for single-particle percolation analysis.'

        )



    valid_z = points[valid_indices, 2]

    z_bottom = float(valid_z.min())

    z_top = float(valid_z.max())

    if np.isclose(z_bottom, z_top):

        raise ValueError('The domain is too thin in z; increase nz.')

    _, evaluated_cell_volumes = get_evaluation_slab_geometry(

        prepared_geometry,

        cell_vertices,

        valid_indices,

        z_bottom,

        z_top,

    )



    # Fix the number of A particles in each trial to remove composition noise.

    rng = np.random.default_rng(seed)

    n_A = int(np.floor(target_frac_A * len(valid_indices) + 0.5))

    shuffled = rng.permutation(valid_indices)

    A_indices = shuffled[:n_A]

    labels = np.full(N, fill_value=-1, dtype=np.int8)

    labels[valid_indices] = 0

    labels[A_indices] = 1



    G_A = nx.Graph()

    G_B = nx.Graph()

    volume_tol = 1e-12

    evaluated_indices = valid_indices[evaluated_cell_volumes[valid_indices] > volume_tol]

    G_A.add_nodes_from(evaluated_indices[labels[evaluated_indices] == 1])

    G_B.add_nodes_from(evaluated_indices[labels[evaluated_indices] == 0])

    numerical_area_tol = 1e-10

    contact_area_tol = max(numerical_area_tol, min_contact_area)

    boundary_area_tol = max(numerical_area_tol, min_boundary_area)



    for (extended_i, extended_j), ridge in zip(vor.ridge_points, vor.ridge_vertices):

        if periodic_xy and extended_i >= N and extended_j >= N:

            continue

        i = int(source_indices[extended_i])

        j = int(source_indices[extended_j])

        if i == j or not (valid[i] and valid[j]) or labels[i] != labels[j]:

            continue

        if -1 in ridge or len(ridge) < 3:

            continue

        face_vertices = vor.vertices[ridge]

        face_in_slab = clip_polygon_to_z_slab(face_vertices, z_bottom, z_top)

        face_area = polygon_area_3d(face_in_slab)

        if face_area <= contact_area_tol:

            continue

        graph = G_A if labels[i] == 1 else G_B

        graph.add_edge(i, j, contact_area=face_area)



    bottom_A_nodes = set()

    top_A_nodes = set()

    bottom_B_nodes = set()

    top_B_nodes = set()

    for i in valid_indices:

        bottom_section = intersect_cell_with_z_plane(

            cell_vertices[i],

            z_plane=z_bottom,

        )

        top_section = intersect_cell_with_z_plane(

            cell_vertices[i],

            z_plane=z_top,

        )

        bottom_area = polygon_area_3d(bottom_section)

        top_area = polygon_area_3d(top_section)

        if labels[i] == 1:

            if bottom_area > boundary_area_tol:

                bottom_A_nodes.add(i)

            if top_area > boundary_area_tol:

                top_A_nodes.add(i)

        else:

            if bottom_area > boundary_area_tol:

                bottom_B_nodes.add(i)

            if top_area > boundary_area_tol:

                top_B_nodes.add(i)



    spanning_volume_fraction_A = spanning_component_volume_fraction(

        G_A,

        bottom_nodes=bottom_A_nodes,

        top_nodes=top_A_nodes,

        node_volumes=evaluated_cell_volumes,

    )

    spanning_volume_fraction_B = spanning_component_volume_fraction(

        G_B,

        bottom_nodes=bottom_B_nodes,

        top_nodes=top_B_nodes,

        node_volumes=evaluated_cell_volumes,

    )

    return (spanning_volume_fraction_A, spanning_volume_fraction_B, labels)



def make_single_particle_volume(

    points,

    labels,

    nx=100,

    ny=100,

    nz=50,

    margin=0.3,

    dtype=np.int8,

    periodic_xy=False,

    box_lengths=None,

):

    """Convert the single-particle model into a nearest-cell 3D array."""

    points = np.asarray(points, dtype=float)

    labels = np.asarray(labels, dtype=np.int8)

    if labels.shape != (len(points),):

        raise ValueError('labels must be a 1D array with the same length as points.')

    if nx < 2 or ny < 2 or nz < 2:

        raise ValueError('nx, ny, and nz must all be at least 2.')

    valid_mask = np.isin(labels, [0, 1])

    if not np.any(valid_mask):

        raise ValueError('make_single_particle_volume: no valid cells were found.')

    points_valid = points[valid_mask]

    if periodic_xy:

        if box_lengths is None:

            raise ValueError('box_lengths is required when periodic_xy=True.')

        box_lengths = np.asarray(box_lengths, dtype=float)

        x_min, x_max = 0.0, box_lengths[0]

        y_min, y_max = 0.0, box_lengths[1]

    else:

        x_min = points_valid[:, 0].min() - margin

        x_max = points_valid[:, 0].max() + margin

        y_min = points_valid[:, 1].min() - margin

        y_max = points_valid[:, 1].max() + margin

    z_min = points_valid[:, 2].min()

    z_max = points_valid[:, 2].max()

    if np.isclose(z_min, z_max):

        raise ValueError('The domain is too thin in z.')



    xs = np.linspace(x_min, x_max, nx)

    ys = np.linspace(y_min, y_max, ny)

    zs = np.linspace(z_min, z_max, nz)

    if periodic_xy:

        query_points, query_sources, _, _ = make_periodic_xy_images(

            points,

            box_lengths=box_lengths,

        )

        query_valid_mask = valid_mask[query_sources]

    else:

        query_points = points

        query_sources = np.arange(len(points), dtype=int)

        query_valid_mask = valid_mask

    tree_all = cKDTree(query_points)

    volume = np.full((nz, ny, nx), fill_value=-1, dtype=dtype)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')

    xy_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    for iz, z in enumerate(zs):

        grid_points = np.column_stack(

            [xy_points, np.full(len(xy_points), z, dtype=float)]

        )

        _, nearest_query_indices = tree_all.query(grid_points, k=1)

        nearest_is_valid = query_valid_mask[nearest_query_indices]

        slice_values = np.full(len(grid_points), fill_value=-1, dtype=dtype)

        if np.any(nearest_is_valid):

            selected = nearest_query_indices[nearest_is_valid]

            cell_indices = query_sources[selected]

            slice_values[nearest_is_valid] = labels[cell_indices]

        volume[iz] = slice_values.reshape(ny, nx)

    return (volume, xs, ys, zs)



def make_AB_volume(points, normals, thresholds, nx=100, ny=100, nz=50, margin=0.3, dtype=np.int8, periodic_xy=False, box_lengths=None):

    points = np.asarray(points, dtype=float)

    normals_list = list(normals)

    thresholds_list = list(thresholds)

    if len(points) != len(normals_list):

        raise ValueError('points and normals must have the same length.')

    if len(points) != len(thresholds_list):

        raise ValueError('points and thresholds must have the same length.')

    if nx < 2 or ny < 2 or nz < 2:

        raise ValueError('nx, ny, and nz must all be at least 2.')

    if margin < 0:

        raise ValueError('margin must be non-negative.')

    valid_mask = np.array([normal is not None and threshold is not None for normal, threshold in zip(normals_list, thresholds_list)], dtype=bool)

    if not np.any(valid_mask):

        raise ValueError('make_AB_volume: no valid cells were found.')

    valid_indices = np.where(valid_mask)[0]

    points_valid = points[valid_mask]

    normals_array = np.zeros((len(points), 3), dtype=float)

    thresholds_array = np.zeros(len(points), dtype=float)

    for i in valid_indices:

        normal = np.asarray(normals_list[i], dtype=float)

        normal_norm = np.linalg.norm(normal)

        if normal_norm <= 1e-15:

            raise ValueError(f'The normal vector of cell {i} is zero.')

        normals_array[i] = normal / normal_norm

        thresholds_array[i] = float(thresholds_list[i])

    if periodic_xy:

        if box_lengths is None:

            raise ValueError('box_lengths is required when periodic_xy=True.')

        box_lengths = np.asarray(box_lengths, dtype=float)

        if box_lengths.shape != (3,) or np.any(box_lengths <= 0):

            raise ValueError('box_lengths must be a positive array of length 3.')

        x_min = 0.0

        x_max = box_lengths[0]

        y_min = 0.0

        y_max = box_lengths[1]

    else:

        x_min = points_valid[:, 0].min() - margin

        x_max = points_valid[:, 0].max() + margin

        y_min = points_valid[:, 1].min() - margin

        y_max = points_valid[:, 1].max() + margin

    z_min = points_valid[:, 2].min()

    z_max = points_valid[:, 2].max()

    if np.isclose(z_min, z_max):

        raise ValueError('The domain is too thin in z.')

    xs = np.linspace(x_min, x_max, nx)

    ys = np.linspace(y_min, y_max, ny)

    zs = np.linspace(z_min, z_max, nz)

    if periodic_xy:

        query_points, query_sources, _, _ = make_periodic_xy_images(

            points,

            box_lengths=box_lengths,

        )

        query_valid_mask = valid_mask[query_sources]

    else:

        query_points = points

        query_sources = np.arange(len(points), dtype=int)

        query_valid_mask = valid_mask

    tree_all = cKDTree(query_points)

    volume = np.full((nz, ny, nx), fill_value=-1, dtype=dtype)

    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')

    xy_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    for iz, z in enumerate(zs):

        grid_points = np.column_stack([xy_points, np.full(len(xy_points), z, dtype=float)])

        _, nearest_query_indices = tree_all.query(grid_points, k=1)

        nearest_is_valid = query_valid_mask[nearest_query_indices]

        slice_values = np.full(len(grid_points), fill_value=-1, dtype=dtype)

        if np.any(nearest_is_valid):

            valid_grid_points = grid_points[nearest_is_valid]

            selected_query_indices = nearest_query_indices[nearest_is_valid]

            cell_indices = query_sources[selected_query_indices]

            centers = query_points[selected_query_indices]

            normals_here = normals_array[cell_indices]

            thresholds_here = thresholds_array[cell_indices]

            signed_positions = np.einsum('ij,ij->i', valid_grid_points - centers, normals_here)

            is_A = signed_positions <= thresholds_here

            slice_values[nearest_is_valid] = 0

            valid_positions = np.where(nearest_is_valid)[0]

            slice_values[valid_positions[is_A]] = 1

        volume[iz] = slice_values.reshape(ny, nx)

    return (volume, xs, ys, zs)
