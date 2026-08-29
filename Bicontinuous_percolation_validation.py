"""Validation and diagnostic visualization for the percolation model."""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import Bicontinuous_percolation_main as core
from Bicontinuous_percolation_main import (
    Generate_lattice,
    Generate_FCC,
    Generate_BCC,
    Generate_HCP,
    generate_points,
    get_lattice_box_lengths,
    make_periodic_xy_images,
    polygon_area_3d,
    prepare_voronoi_geometry,
    get_cell_vertices,
    random_unit_vector,
    compute_cell_edges_from_hull,
    volume_A_for_threshold,
    find_threshold_for_fraction,
    shared_face_contact_fractions,
    all_cells_AB_volumes,
    janus_voronoi_percolation_random,
    single_particle_percolation_random,
)

def test_generate_simple_lattice():
    points = Generate_lattice(a=1.0, nx=3, ny=3, nz=3)
    for i in range(0, len(points)):
        print(points[i][0], points[i][1], points[i][2])

def test_generate_fcc():
    points = Generate_FCC(a=1.0, nx=3, ny=3, nz=3)
    for i in range(0, len(points)):
        print(points[i][0], points[i][1], points[i][2])

def test_generate_bcc():
    points = Generate_BCC(a=1.0, nx=3, ny=3, nz=3)
    for i in range(0, len(points)):
        print(points[i][0], points[i][1], points[i][2])

def test_generate_hcp():
    points = Generate_HCP(a=1.0, nx=3, ny=3, nz=3)
    for i in range(0, len(points)):
        print(points[i][0], points[i][1], points[i][2])

def test_generate_points_fcc_bcc_hcp():
    nx_cell, ny_cell, nz_cell = 3, 4, 5
    fcc_points = generate_points(
        lattice_type='fcc',
        a=1.0,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    bcc_points = generate_points(
        lattice_type='bcc',
        a=1.0,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    hcp_points = generate_points(
        lattice_type='hcp',
        a=1.0,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    expected_fcc = 4 * nx_cell * ny_cell * nz_cell
    expected_bcc = 2 * nx_cell * ny_cell * nz_cell
    expected_hcp = 4 * nx_cell * ny_cell * nz_cell
    assert len(fcc_points) == expected_fcc
    assert len(bcc_points) == expected_bcc
    assert len(hcp_points) == expected_hcp
    assert len(np.unique(fcc_points, axis=0)) == expected_fcc
    assert len(np.unique(bcc_points, axis=0)) == expected_bcc
    assert len(np.unique(hcp_points, axis=0)) == expected_hcp
    print(f'FCC point count: {len(fcc_points)}')
    print(f'BCC point count: {len(bcc_points)}')
    print(f'HCP point count: {len(hcp_points)}')
    print('OK: test_generate_points_fcc_bcc_hcp')

def test_periodic_xy_images():
    lattice_type = 'hcp'
    a = 1.0
    nx_cell, ny_cell, nz_cell = 3, 4, 5
    points = generate_points(
        lattice_type=lattice_type,
        a=a,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    box_lengths = get_lattice_box_lengths(
        lattice_type=lattice_type,
        a=a,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    periodic_points, source_indices, image_shifts, central_indices = make_periodic_xy_images(
        points,
        box_lengths=box_lengths,
    )
    assert len(periodic_points) == 9 * len(points)
    assert len(source_indices) == len(periodic_points)
    assert len(image_shifts) == len(periodic_points)
    assert np.array_equal(central_indices, np.arange(len(points)))
    assert np.allclose(periodic_points[:len(points)], points)
    assert np.allclose(image_shifts[:len(points)], 0.0)
    assert np.all(source_indices[:len(points)] == np.arange(len(points)))
    assert np.isclose(box_lengths[0], nx_cell * a)
    assert np.isclose(box_lengths[1], ny_cell * np.sqrt(3.0) * a)
    assert np.isclose(box_lengths[2], nz_cell * np.sqrt(8.0 / 3.0) * a)
    print('HCP box lengths:', box_lengths)
    print('Point count including periodic images:', len(periodic_points))
    print('OK: test_periodic_xy_images')

def test_periodic_xy_voronoi_boundary_cell():
    lattice_type = 'fcc'
    a = 1.0
    nx_cell, ny_cell, nz_cell = 3, 3, 5
    points = generate_points(
        lattice_type=lattice_type,
        a=a,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    box_lengths = get_lattice_box_lengths(
        lattice_type=lattice_type,
        a=a,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    periodic_points, _, _, central_indices = make_periodic_xy_images(
        points,
        box_lengths=box_lengths,
    )
    vor_periodic = Voronoi(periodic_points)
    target_candidates = np.where(
        np.isclose(points[:, 0], 0.0)
        & np.isclose(points[:, 1], 0.0)
        & (points[:, 2] > a)
        & (points[:, 2] < (nz_cell - 1) * a)
    )[0]
    if len(target_candidates) == 0:
        raise AssertionError('No boundary cell was found for the periodic test.')
    target = int(target_candidates[0])
    verts = get_cell_vertices(vor_periodic, int(central_indices[target]))
    assert verts is not None
    assert len(verts) >= 4
    print('Tested source-cell index:', target)
    print('Voronoi vertex count under periodic boundaries:', len(verts))
    print('OK: test_periodic_xy_voronoi_boundary_cell')

def make_simple_cubic_points(nx=4, ny=4, nz=4, a=1.0):
    """Create a simple-cubic point set with an analytical reference solution."""
    return np.array(
        [
            [ix * a, iy * a, iz * a]
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz)
        ],
        dtype=float,
    )

def source_index(ix, iy, iz, ny, nz):
    """Return the source index for coordinates in make_simple_cubic_points()."""
    return ix * ny * nz + iy * nz + iz

def minimum_image_displacement(point_i, point_j, box_lengths):
    """
    Apply the minimum-image convention in x and y, but not in z.

    In periodic directions, wrap displacement into [-L/2, L/2).
    """
    displacement = np.asarray(point_j, dtype=float) - np.asarray(point_i, dtype=float)
    box_lengths = np.asarray(box_lengths, dtype=float)
    displacement[:2] -= box_lengths[:2] * np.floor(
        displacement[:2] / box_lengths[:2] + 0.5
    )
    return displacement

def find_ridge(
    vor,
    source_indices,
    image_shifts,
    central_source,
    neighbor_source,
    neighbor_shift,
):
    """
    Find the ridge between a central particle and a specified periodic image.
    """
    neighbor_shift = np.asarray(neighbor_shift, dtype=float)
    for (extended_i, extended_j), ridge in zip(
        vor.ridge_points,
        vor.ridge_vertices,
    ):
        if extended_i == central_source:
            other = extended_j
        elif extended_j == central_source:
            other = extended_i
        else:
            continue

        if (
            source_indices[other] == neighbor_source
            and np.allclose(image_shifts[other], neighbor_shift)
        ):
            return ridge
    return None

def finite_ridge_area(vor, ridge):
    """Return the area of a finite Voronoi ridge."""
    if ridge is None:
        raise AssertionError('The expected Voronoi ridge was not found.')
    if -1 in ridge or len(ridge) < 3:
        raise AssertionError('The ridge is not a finite polygon.')
    return polygon_area_3d(vor.vertices[ridge])

def test_periodic_boundary_conditions_xy():
    """
    Verify x/y-periodic and z-nonperiodic Voronoi adjacency.

    In a simple-cubic lattice, the analytical shared-face area is a^2.
    Compare an interior face with faces crossing the x and y boundaries.
    """
    nx, ny, nz = 4, 4, 4
    a = 1.0
    points = make_simple_cubic_points(nx=nx, ny=ny, nz=nz, a=a)
    box_lengths = np.array([nx * a, ny * a, nz * a], dtype=float)

    periodic_points, source_indices, image_shifts, central_indices = (
        make_periodic_xy_images(points, box_lengths=box_lengths)
    )
    vor = Voronoi(periodic_points)

    number_of_points = len(points)
    assert len(periodic_points) == 9 * number_of_points
    assert np.array_equal(central_indices, np.arange(number_of_points))
    assert np.allclose(image_shifts[:, 2], 0.0)

    # Use an interior z layer to avoid unbounded cells at nonperiodic z edges.
    z_mid = 2

    # 1. Ordinary interior neighboring face.
    interior = source_index(1, 1, z_mid, ny, nz)
    interior_neighbor = source_index(2, 1, z_mid, ny, nz)
    interior_ridge = find_ridge(
        vor,
        source_indices,
        image_shifts,
        central_source=interior,
        neighbor_source=interior_neighbor,
        neighbor_shift=[0.0, 0.0, 0.0],
    )

    # 2. Face crossing x=0/Lx to the periodic image at x=nx-1.
    x_boundary = source_index(0, 1, z_mid, ny, nz)
    x_wrapped_neighbor = source_index(nx - 1, 1, z_mid, ny, nz)
    x_ridge = find_ridge(
        vor,
        source_indices,
        image_shifts,
        central_source=x_boundary,
        neighbor_source=x_wrapped_neighbor,
        neighbor_shift=[-box_lengths[0], 0.0, 0.0],
    )

    # 3. Face crossing y=0/Ly to the periodic image at y=ny-1.
    y_boundary = source_index(1, 0, z_mid, ny, nz)
    y_wrapped_neighbor = source_index(1, ny - 1, z_mid, ny, nz)
    y_ridge = find_ridge(
        vor,
        source_indices,
        image_shifts,
        central_source=y_boundary,
        neighbor_source=y_wrapped_neighbor,
        neighbor_shift=[0.0, -box_lengths[1], 0.0],
    )

    interior_area = finite_ridge_area(vor, interior_ridge)
    x_boundary_area = finite_ridge_area(vor, x_ridge)
    y_boundary_area = finite_ridge_area(vor, y_ridge)
    expected_area = a ** 2

    assert np.isclose(interior_area, expected_area, rtol=1e-10, atol=1e-10)
    assert np.isclose(x_boundary_area, expected_area, rtol=1e-10, atol=1e-10)
    assert np.isclose(y_boundary_area, expected_area, rtol=1e-10, atol=1e-10)
    assert np.isclose(x_boundary_area, interior_area, rtol=1e-10, atol=1e-10)
    assert np.isclose(y_boundary_area, interior_area, rtol=1e-10, atol=1e-10)

    # 4. Minimum-image distances equal the interior nearest-neighbor distance a.
    dx_min = minimum_image_displacement(
        points[x_boundary],
        points[x_wrapped_neighbor],
        box_lengths,
    )
    dy_min = minimum_image_displacement(
        points[y_boundary],
        points[y_wrapped_neighbor],
        box_lengths,
    )
    assert np.isclose(np.linalg.norm(dx_min), a)
    assert np.isclose(np.linalg.norm(dy_min), a)

    # 5. No z images are created, so the bottom and top are not periodic neighbors.
    z_bottom = source_index(1, 1, 0, ny, nz)
    z_top = source_index(1, 1, nz - 1, ny, nz)
    z_wrap_ridge = find_ridge(
        vor,
        source_indices,
        image_shifts,
        central_source=z_bottom,
        neighbor_source=z_top,
        neighbor_shift=[0.0, 0.0, -box_lengths[2]],
    )
    assert z_wrap_ridge is None

    print('=== Independent periodic-boundary test ===')
    print(f'Interior shared-face area : {interior_area:.12f}')
    print(f'x-boundary face area      : {x_boundary_area:.12f}')
    print(f'y-boundary face area      : {y_boundary_area:.12f}')
    print(f'Analytical value a^2      : {expected_area:.12f}')
    print(f'x-boundary minimum distance: {np.linalg.norm(dx_min):.12f}')
    print(f'y-boundary minimum distance: {np.linalg.norm(dy_min):.12f}')
    print('Periodic images in z: none')
    print('OK: x/y periodic boundaries are correctly reflected in Voronoi ridges.')

def print_cell_edge_report(verts, valid_i):
    print('Cell index:', valid_i)
    print('Vertex count:', len(verts))
    hull_total = ConvexHull(verts)
    edges = compute_cell_edges_from_hull(hull_total)
    print('verts.shape:', verts.shape)
    print('hull_total.simplices.shape:', hull_total.simplices.shape)
    print('\n=== edges (index pairs) ===')
    for a, b in sorted(edges):
        print(int(a), int(b))
    print('\n=== Triangles (simplices) ===')
    for tri in hull_total.simplices:
        print(tri)

def test_compute_cell_edges_from_hull():
    points = Generate_lattice(a=1.0, nx=3, ny=3, nz=3)
    vor = Voronoi(points)
    valid_i = None
    for i in range(len(points)):
        verts_i = get_cell_vertices(vor, i)
        if verts_i is not None and len(verts_i) >= 4:
            valid_i = i
            verts = verts_i
            break
    print_cell_edge_report(verts, valid_i)
    points = Generate_lattice(a=1.0, nx=3, ny=3, nz=3)
    vor = Voronoi(points)
    valid_i = None
    i = len(points) // 2
    verts_i = get_cell_vertices(vor, i)
    if verts_i is not None and len(verts_i) >= 4:
        valid_i = i
        verts = verts_i
    print_cell_edge_report(verts, valid_i)

def make_unit_cube_verts():
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)

def test_volume_A_for_threshold():
    verts = make_unit_cube_verts()
    center = np.zeros(3, dtype=float)
    normal = np.array([1.0, 0.0, 0.0], dtype=float)
    thresholds = [-0.1, 0.25, 0.5, 0.9, 1.1]
    hull_total = ConvexHull(verts)
    edges = compute_cell_edges_from_hull(hull_total)
    vol_total = hull_total.volume
    print('normal =', normal)
    print('center =', center)
    print('total volume =', vol_total)
    print('-' * 50)
    for th in thresholds:
        vol_A = volume_A_for_threshold(verts, center, normal, th, hull_total=hull_total, edges=edges)
        frac_A = vol_A / vol_total if vol_total > 0 else np.nan
        print(f'threshold = {th:7.3f} -> vol_A = {vol_A:8.5f}, frac_A = {frac_A:7.4f}')

def test_find_threshold_for_fraction_unit_cube_x():
    verts = make_unit_cube_verts()
    center = np.zeros(3)
    normal = np.array([1.0, 0.0, 0.0])
    targets = [0.1, 0.25, 0.5, 0.75, 0.9]
    hull_total = ConvexHull(verts)
    vol_total = hull_total.volume
    print(f'total volume (should be 1.0): {vol_total:.6f}')
    for phi in targets:
        th, vol_A, vol_B = find_threshold_for_fraction(verts, center, normal, target_frac_A=phi, tol=0.0001, max_iter=50)
        frac_A = vol_A / vol_total
        print(f'target_frac_A = {phi:.3f}')
        print(f'  threshold   = {th:.6f} (expected ≈ {phi:.6f})')
        print(f'  vol_A\t   = {vol_A:.6f}, vol_B = {vol_B:.6f}')
        print(f'  frac_A\t  = {frac_A:.6f}')
        print()
        assert np.isclose(th, phi, atol=0.001), f'threshold mismatch: phi={phi}, th={th}'
        assert np.isclose(frac_A, phi, atol=0.001), f'frac_A mismatch: phi={phi}, frac_A={frac_A}'

def test_random_unit_vector(n_samples_print=5, n_samples_stat=1000, seed=None):
    rng = np.random.default_rng(seed)
    print('=== test_random_unit_vector ===')
    print(f'\n-- First {n_samples_print} vectors --')
    for i in range(n_samples_print):
        v = random_unit_vector(rng)
        norm = np.linalg.norm(v)
        print(f'case{i}: v = {v}, norm = {norm:.6f}')
    vecs = []
    for _ in range(n_samples_stat):
        v = random_unit_vector(rng)
        vecs.append(v)
    vecs = np.asarray(vecs)
    norms = np.linalg.norm(vecs, axis=1)
    mean_norm = norms.mean()
    max_dev = np.abs(norms - 1.0).max()
    mean_vec = vecs.mean(axis=0)
    mean_vec_norm = np.linalg.norm(mean_vec)
    print('\n-- Statistics --')
    print(f'Mean norm\t = {mean_norm:.6f}')
    print(f'Maximum |norm-1|\t = {max_dev:.6e}')
    print(f'Mean vector\t = {mean_vec}')
    print(f'Norm of mean vector = {mean_vec_norm:.6f}')

def test_all_cells_AB_volumes():
    pts = Generate_lattice(a=1.0, nx=5, ny=5, nz=5)
    target_frac_A = 0.6
    cell_indices, normals, thresholds, vols_A, vols_B, vor = all_cells_AB_volumes(pts, target_frac_A=target_frac_A, tol=0.001, max_iter=40)
    vols_A = np.array(vols_A)
    vols_B = np.array(vols_B)
    frac_A = vols_A / (vols_A + vols_B)
    errors = frac_A - target_frac_A
    rmse = np.sqrt(np.mean(errors ** 2))
    max_abs_error = np.max(np.abs(errors))
    print('Number of computed cells:', len(cell_indices))
    print('Target A-volume fraction:', target_frac_A)
    print('Mean A-volume fraction:', frac_A.mean())
    print('RMSE of A-volume fraction:', rmse)
    print('Minimum/maximum A-volume fraction:', frac_A.min(), frac_A.max())
    print('Maximum relative error in A-volume fraction:', max_abs_error / 0.6 * 100, '%')

def test_single_particle_model_basic():
    """Check composition, spanning fractions, and endpoints for single particles."""
    points = generate_points(
        lattice_type='fcc',
        a=1.0,
        nx=4,
        ny=4,
        nz=4,
    )
    box_lengths = get_lattice_box_lengths(
        lattice_type='fcc',
        a=1.0,
        nx=4,
        ny=4,
        nz=4,
    )
    prepared_geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=True,
        box_lengths=box_lengths,
    )
    perA, perB, labels = single_particle_percolation_random(
        points,
        target_frac_A=0.5,
        seed=0,
        periodic_xy=True,
        box_lengths=box_lengths,
        prepared_geometry=prepared_geometry,
    )
    valid_labels = labels[labels >= 0]
    expected_A_count = int(np.floor(0.5 * len(valid_labels) + 0.5))
    assert np.count_nonzero(valid_labels == 1) == expected_A_count
    assert np.isscalar(perA) and 0.0 <= perA <= 1.0
    assert np.isscalar(perB) and 0.0 <= perB <= 1.0

    all_A, no_B, labels_A = single_particle_percolation_random(
        points,
        target_frac_A=1.0,
        seed=0,
        periodic_xy=True,
        box_lengths=box_lengths,
        prepared_geometry=prepared_geometry,
    )
    no_A, all_B, labels_B = single_particle_percolation_random(
        points,
        target_frac_A=0.0,
        seed=0,
        periodic_xy=True,
        box_lengths=box_lengths,
        prepared_geometry=prepared_geometry,
    )
    assert np.isclose(all_A, 1.0) and np.isclose(no_B, 0.0)
    assert np.isclose(no_A, 0.0) and np.isclose(all_B, 1.0)
    assert np.all(labels_A[labels_A >= 0] == 1)
    assert np.all(labels_B[labels_B >= 0] == 0)
    print('OK: test_single_particle_model_basic')

def test_janus_model_basic():
    """Check normal execution, spanning fractions, and endpoints for Janus cells."""
    points = generate_points(
        lattice_type='fcc',
        a=1.0,
        nx=3,
        ny=3,
        nz=3,
    )
    box_lengths = get_lattice_box_lengths(
        lattice_type='fcc',
        a=1.0,
        nx=3,
        ny=3,
        nz=3,
    )
    prepared_geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=True,
        box_lengths=box_lengths,
    )
    common_kwargs = {
        'seed': 0,
        'periodic_xy': True,
        'box_lengths': box_lengths,
        'prepared_geometry': prepared_geometry,
    }

    perA, perB, normals, thresholds = janus_voronoi_percolation_random(
        points,
        target_frac_A=0.5,
        **common_kwargs,
    )
    assert np.isscalar(perA) and 0.0 <= perA <= 1.0
    assert np.isscalar(perB) and 0.0 <= perB <= 1.0
    assert len(normals) == len(points)
    assert len(thresholds) == len(points)

    all_A, no_B, _, _ = janus_voronoi_percolation_random(
        points,
        target_frac_A=1.0,
        **common_kwargs,
    )
    no_A, all_B, _, _ = janus_voronoi_percolation_random(
        points,
        target_frac_A=0.0,
        **common_kwargs,
    )
    assert np.isclose(all_A, 1.0) and np.isclose(no_B, 0.0)
    assert np.isclose(no_A, 0.0) and np.isclose(all_B, 1.0)
    print('OK: test_janus_model_basic')

def test_janus_finite_cell_failure_is_not_ignored():
    """Verify that a finite-cell split failure is not treated as nonpercolation."""
    points = generate_points(
        lattice_type='fcc',
        a=1.0,
        nx=3,
        ny=3,
        nz=3,
    )
    box_lengths = get_lattice_box_lengths(
        lattice_type='fcc',
        a=1.0,
        nx=3,
        ny=3,
        nz=3,
    )
    original_function = core.find_threshold_for_fraction

    def forced_failure(*args, **kwargs):
        raise RuntimeError('Forced error for validation')

    core.find_threshold_for_fraction = forced_failure
    try:
        try:
            janus_voronoi_percolation_random(
                points,
                target_frac_A=0.5,
                periodic_xy=True,
                box_lengths=box_lengths,
                seed=0,
            )
        except RuntimeError as error:
            assert 'finite Voronoi cell' in str(error)
        else:
            raise AssertionError('A finite-cell calculation failure was ignored.')
    finally:
        core.find_threshold_for_fraction = original_function
    print('OK: test_janus_finite_cell_failure_is_not_ignored')

def set_axes_equal_3d(ax, points, margin=0.05):
    points = np.asarray(points, dtype=float)
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half_range = 0.5 * max_range * (1.0 + margin)
    ax.set_xlim(x_mid - half_range, x_mid + half_range)
    ax.set_ylim(y_mid - half_range, y_mid + half_range)
    ax.set_zlim(z_mid - half_range, z_mid + half_range)
    ax.set_box_aspect([1, 1, 1])

def visualize_cell_with_plane_and_Apoly(verts, center, normal, threshold, hull_total=None, edges=None, title=None):
    verts = np.asarray(verts, dtype=float)
    center = np.asarray(center, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if hull_total is None:
        hull_total = ConvexHull(verts)
    if edges is None:
        edges = compute_cell_edges_from_hull(hull_total)
    faces = hull_total.simplices
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    for tri in faces:
        tri_verts = verts[tri]
        tri_verts = np.vstack([tri_verts, tri_verts[0]])
        ax.plot(tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2], linewidth=0.5, color='k')
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
    new_points = np.asarray(new_points)
    if len(new_points) >= 4:
        new_points = np.unique(new_points, axis=0)
    if len(new_points) >= 4:
        hull_A = ConvexHull(new_points)
        faces_A = hull_A.simplices
        poly = Poly3DCollection(new_points[faces_A], alpha=0.4)
        poly.set_edgecolor('none')
        ax.add_collection3d(poly)
        ax.scatter(*new_points.T, s=40, color='C0', label='A-side vertices (including intersections)')
    else:
        print('Fewer than four A-side points; no polyhedron can be constructed.')
    p0 = center + n * threshold
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    bbox_size = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    L = bbox_size * 0.8
    uu = np.linspace(-L, L, 2)
    vv = np.linspace(-L, L, 2)
    U, V = np.meshgrid(uu, vv)
    X = p0[0] + U * u[0] + V * v[0]
    Y = p0[1] + U * u[1] + V * v[1]
    Z = p0[2] + U * u[2] + V * v[2]
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title is not None:
        ax.set_title(title)
    plane_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    all_plot_points = np.vstack([verts, np.asarray([center]), plane_points])
    set_axes_equal_3d(ax, all_plot_points, margin=0.15)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def visualize_unit_cube_with_plane_and_Apoly(target_frac_A=0.5, n_check_cubes=3, tol=0.01, max_iter=40, seed=None):
    rng = np.random.default_rng(seed)
    print('=== visualize_unit_cube_with_plane_and_Apoly ===')
    print(f'target_frac_A = {target_frac_A}\n')
    verts = np.asarray(make_unit_cube_verts(), dtype=float)
    center = verts.mean(axis=0)
    for k in range(n_check_cubes):
        normal = random_unit_vector(rng)
        threshold, vol_A, vol_B = find_threshold_for_fraction(verts, center, normal, target_frac_A, tol=tol, max_iter=max_iter)
        hull_total = ConvexHull(verts)
        edges = compute_cell_edges_from_hull(hull_total)
        vol_A_check = volume_A_for_threshold(verts, center, normal, threshold, hull_total=hull_total, edges=edges)
        vol_total = hull_total.volume
        frac_A = vol_A_check / vol_total if vol_total > 0 else np.nan
        print(f'cube {k:3d}')
        print('  center :', center)
        print('  normal :', normal)
        print(f'  threshold = {threshold:.4f}')
        print(f'  frac_A\t= {frac_A:.4f}')
        if not np.isclose(frac_A, target_frac_A, atol=tol):
            print('  [WARN] The computed fraction may deviate from target_frac_A.')
        visualize_cell_with_plane_and_Apoly(verts, center, normal, threshold, hull_total=hull_total, edges=edges, title=f'unit cube {k}, frac_A={frac_A:.3f}')
    print(f'\nNumber of checked cubes: {n_check_cubes}')

def visualize_cell_with_plane_and_Apoly_check(nx=3, ny=3, nz=3, target_frac_A=0.5, n_check_cells=3, tol=0.01, seed=None, lattice_type='lattice'):
    rng = np.random.default_rng(seed)
    pts = generate_points(lattice_type=lattice_type, a=1.0, nx=nx, ny=ny, nz=nz)
    vor = Voronoi(pts)
    checked = 0
    print(f'target_frac_A = {target_frac_A}\n')
    for i_cell in range(len(pts)):
        verts = get_cell_vertices(vor, i_cell)
        if verts is None:
            continue
        center = pts[i_cell]
        normal = random_unit_vector(rng)
        threshold, vol_A, vol_B = find_threshold_for_fraction(verts, center, normal, target_frac_A)
        hull_total = ConvexHull(verts)
        edges = compute_cell_edges_from_hull(hull_total)
        vol_A_check = volume_A_for_threshold(verts, center, normal, threshold, hull_total=hull_total, edges=edges)
        vol_total = hull_total.volume
        frac_A = vol_A_check / vol_total if vol_total > 0 else np.nan
        print(f'cell {i_cell:3d}')
        print('  Cell center:', center)
        print('  Normal vector:', normal)
        print(f'  threshold = {threshold:.3f}')
        print(f'  frac_A\t= {frac_A:.3f}')
        if not np.isclose(frac_A, target_frac_A, atol=tol):
            print('  [WARN] The computed fraction may deviate from target_frac_A.')
        visualize_cell_with_plane_and_Apoly(verts, center, normal, threshold, hull_total=hull_total, edges=edges, title=f'cell {i_cell}, frac_A={frac_A:.3f}')
        checked += 1
        if checked >= n_check_cells:
            break
    print(f'\nNumber of checked cells: {checked}')

def make_plane_patch(center, normal, threshold, scale=0.18):
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    p0 = center + threshold * normal
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, tmp)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    corners = np.array([p0 + scale * (+u + v), p0 + scale * (+u - v), p0 + scale * (-u - v), p0 + scale * (-u + v)])
    return corners

def visualize_janus_voronoi_two_cells(
    seed=None,
    nx=5,
    ny=5,
    nz=5,
    lattice_type='fcc',
    target_frac_A=0.5,
    periodic_xy=True,
):
    """Visualize two adjacent Janus cells using exact shared-face areas."""
    points = generate_points(
        lattice_type=lattice_type,
        a=1.0,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    box_lengths = get_lattice_box_lengths(
        lattice_type=lattice_type,
        a=1.0,
        nx=nx,
        ny=ny,
        nz=nz,
    )
    geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )
    _, _, normals, thresholds = janus_voronoi_percolation_random(
        points,
        target_frac_A=target_frac_A,
        tol=0.001,
        max_iter=40,
        seed=seed,
        contact_method='exact_shared_face',
        min_contact_area=1e-6,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
        prepared_geometry=geometry,
    )

    vor = geometry['vor']
    vor_points = geometry['vor_points']
    source_indices = geometry['source_indices']
    number_of_points = len(points)
    selected = None
    for (extended_i, extended_j), ridge in zip(
        vor.ridge_points,
        vor.ridge_vertices,
    ):
        if periodic_xy and extended_i >= number_of_points and extended_j >= number_of_points:
            continue
        i = int(source_indices[extended_i])
        j = int(source_indices[extended_j])
        if i == j or normals[i] is None or normals[j] is None:
            continue
        if -1 in ridge or len(ridge) < 3:
            continue
        verts_i = get_cell_vertices(vor, extended_i)
        verts_j = get_cell_vertices(vor, extended_j)
        if verts_i is None or verts_j is None:
            continue
        selected = (extended_i, extended_j, i, j, ridge, verts_i, verts_j)
        break

    if selected is None:
        raise RuntimeError('No finite adjacent Voronoi-cell pair was found.')

    extended_i, extended_j, i, j, ridge, verts_i, verts_j = selected
    center_i = vor_points[extended_i]
    center_j = vor_points[extended_j]
    face_vertices = vor.vertices[ridge]
    frac_AA, frac_BB, area_AA, area_BB, face_area = (
        shared_face_contact_fractions(
            face_vertices,
            center_i,
            normals[i],
            thresholds[i],
            center_j,
            normals[j],
            thresholds[j],
        )
    )

    print('=== Exact shared-face contact ===')
    print(f'Cells: {i} and {j}')
    print(f'Total shared-face area: {face_area:.8f}')
    print(f'A-A contact: fraction={frac_AA:.6f}, area={area_AA:.8f}')
    print(f'B-B contact: fraction={frac_BB:.6f}, area={area_BB:.8f}')
    print(f'A-A connected at 1e-6: {area_AA > 1e-6}')
    print(f'B-B connected at 1e-6: {area_BB > 1e-6}')

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    for vertices, color in ((verts_i, 'navy'), (verts_j, 'darkorange')):
        hull = ConvexHull(vertices)
        for triangle in hull.simplices:
            triangle_vertices = np.vstack([vertices[triangle], vertices[triangle[0]]])
            ax.plot(
                triangle_vertices[:, 0],
                triangle_vertices[:, 1],
                triangle_vertices[:, 2],
                color=color,
                linewidth=0.6,
                alpha=0.35,
            )

    plane_i = make_plane_patch(center_i, normals[i], thresholds[i], scale=0.25)
    plane_j = make_plane_patch(center_j, normals[j], thresholds[j], scale=0.25)
    ax.add_collection3d(
        Poly3DCollection([plane_i], alpha=0.25, facecolor='blue', edgecolor='blue')
    )
    ax.add_collection3d(
        Poly3DCollection([plane_j], alpha=0.25, facecolor='orange', edgecolor='orange')
    )
    ordered_face = core.order_coplanar_points(face_vertices)
    ax.add_collection3d(
        Poly3DCollection(
            [ordered_face],
            alpha=0.45,
            facecolor='red',
            edgecolor='darkred',
            linewidth=1.5,
        )
    )
    ax.scatter(*center_i, s=80, c='navy', label=f'center {i}')
    ax.scatter(*center_j, s=80, c='darkorange', label=f'center {j}')
    ax.quiver(*center_i, *normals[i], length=0.25, color='navy')
    ax.quiver(*center_j, *normals[j], length=0.25, color='darkorange')
    all_plot_points = np.vstack(
        [verts_i, verts_j, face_vertices, plane_i, plane_j, center_i, center_j]
    )
    set_axes_equal_3d(ax, all_plot_points, margin=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Two Janus cells and their exact shared face')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Uncomment only one validation or visualization to run.

    # 1. Inspect simple-cubic coordinates.
    # test_generate_simple_lattice()

    # 2. Inspect FCC coordinates.
    # test_generate_fcc()

    # 3. Inspect BCC coordinates.
    # test_generate_bcc()

    # 4. Inspect HCP coordinates.
    # test_generate_hcp()

    # 5. Validate FCC, BCC, and HCP point counts.
    #test_generate_points_fcc_bcc_hcp()

    # 6. Validate periodic copies in x and y.
    # test_periodic_xy_images()

    # 7. Validate a Voronoi cell on a periodic boundary.
    # test_periodic_xy_voronoi_boundary_cell()

    # 8. Validate x/y-periodic and z-nonperiodic boundaries.
    # test_periodic_boundary_conditions_xy()

    # 9. Validate Voronoi-cell edge extraction.
    # test_compute_cell_edges_from_hull()

    # 10. Validate the A-phase volume in a unit cube.
    # test_volume_A_for_threshold()

    # 11. Validate the split threshold for a specified A-phase fraction.
    # test_find_threshold_for_fraction_unit_cube_x()

    # 12. Validate random unit vectors.
    # test_random_unit_vector(seed=0)

    # 13. Validate A/B volumes in all finite cells.
    # test_all_cells_AB_volumes()

    # 14. Validate the two-species single-particle model.
    # test_single_particle_model_basic()

    # 15. Validate the basic Janus model.
    #test_janus_model_basic()

    # 16. Verify that finite-cell failures are not ignored.
    # test_janus_finite_cell_failure_is_not_ignored()

    # 17. Visualize the phase split in a unit cube.
    # visualize_unit_cube_with_plane_and_Apoly(seed=0)

    # 18. Visualize the phase split in a Voronoi cell.
    # visualize_cell_with_plane_and_Apoly_check(
    #     lattice_type="fcc",
    #     seed=0,
    # )

    # 19. Visualize two cells and their exact shared face.
    visualize_janus_voronoi_two_cells(
        seed=0,
        nx=6,
        ny=6,
        nz=6,
        lattice_type="hcp",
        target_frac_A=0.5,
        periodic_xy=True,
    )
