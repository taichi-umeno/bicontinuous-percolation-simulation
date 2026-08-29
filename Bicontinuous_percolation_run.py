"""Production calculations and 2D/3D visualization for the paper."""

import csv
import time
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from Bicontinuous_percolation_main import (
    generate_points,
    get_lattice_box_lengths,
    prepare_voronoi_geometry,
    get_cell_vertices,
    compute_cell_edges_from_hull,
    janus_voronoi_percolation_random,
    single_particle_percolation_random,
    make_AB_volume,
    make_single_particle_volume,
)


def _clip_cell_to_phase(verts, center, normal, threshold, phase):
    """Return the convex polyhedron occupied by one phase of a Janus cell."""
    verts = np.asarray(verts, dtype=float)
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    if phase not in (0, 1):
        raise ValueError('phase must be 0 (B) or 1 (A).')

    cell_hull = ConvexHull(verts)
    cell_edges = compute_cell_edges_from_hull(cell_hull)
    signed_distance = (verts - center) @ normal - threshold
    if phase == 0:
        signed_distance = -signed_distance

    clipped_points = [
        point for point, distance in zip(verts, signed_distance)
        if distance <= 1e-10
    ]
    for i, j in cell_edges:
        distance_i = signed_distance[i]
        distance_j = signed_distance[j]
        if distance_i * distance_j < -1e-20:
            fraction = distance_i / (distance_i - distance_j)
            clipped_points.append(verts[i] + fraction * (verts[j] - verts[i]))

    if len(clipped_points) < 4:
        return None
    clipped_points = np.unique(
        np.round(np.asarray(clipped_points, dtype=float), decimals=12),
        axis=0,
    )
    if len(clipped_points) < 4:
        return None
    try:
        clipped_hull = ConvexHull(clipped_points)
    except QhullError as error:
        raise RuntimeError(
            f'Convex-hull construction failed for Janus phase {phase}.'
        ) from error
    return clipped_points, clipped_hull

def _shrink_polyhedron(points, shrink_factor=0.94):
    """Shrink a polyhedron toward its centroid to reveal cell boundaries."""
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)

    return centroid + shrink_factor * (points - centroid)


def _adjust_hex_brightness(hex_color, brightness):
    """Return an RGB color obtained by darkening or lightening a hex color."""
    hex_color = hex_color.lstrip('#')
    base_rgb = np.array(
        [int(hex_color[index:index + 2], 16) for index in (0, 2, 4)],
        dtype=float,
    )

    if brightness <= 1.0:
        adjusted = base_rgb * brightness
    else:
        adjusted = base_rgb + (255.0 - base_rgb) * (brightness - 1.0)

    adjusted = np.clip(np.rint(adjusted), 0, 255).astype(int)
    return f'rgb({adjusted[0]},{adjusted[1]},{adjusted[2]})'


def _make_directional_face_colors(
    points,
    hull,
    base_color,
    lightposition,
    minimum_brightness=0.48,
    maximum_brightness=1.10,
):
    """Assign a light-to-dark color to every triangular face."""
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(hull.simplices, dtype=int)
    face_centers = points[triangles].mean(axis=1)

    face_normals = np.asarray(hull.equations[:, :3], dtype=float)
    normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(normal_lengths, 1e-12)

    light = np.array(
        [
            lightposition['x'],
            lightposition['y'],
            lightposition['z'],
        ],
        dtype=float,
    )
    light_vectors = light - face_centers
    light_lengths = np.linalg.norm(light_vectors, axis=1, keepdims=True)
    light_vectors = light_vectors / np.maximum(light_lengths, 1e-12)

    illumination = np.clip(
        np.sum(face_normals * light_vectors, axis=1),
        0.0,
        1.0,
    )
    brightness = (
        minimum_brightness
        + (maximum_brightness - minimum_brightness) * illumination
    )

    return [
        _adjust_hex_brightness(base_color, value)
        for value in brightness
    ]


def plot_janus_voronoi_cells(
    points,
    normals,
    thresholds,
    periodic_xy=False,
    box_lengths=None,
    phase='both',
    max_cells=None,
    cutaway=False,
    opacity=1.0,
    edge_width=0.6,
    shrink_factor=0.90,
    strong_shading=False,
    title=None,
    savepath=None,
    show=True,
):
    """Render the A and B regions of Janus Voronoi cells as polyhedra."""

    points = np.asarray(points, dtype=float)

    if phase not in ('A', 'B', 'both'):
        raise ValueError("phase must be 'A', 'B', or 'both'.")

    if max_cells is not None and max_cells < 1:
        raise ValueError('max_cells must be at least 1 or None.')

    if not 0.0 < opacity <= 1.0:
        raise ValueError(
            'opacity must be greater than 0 and at most 1.'
        )

    if edge_width < 0.0:
        raise ValueError('edge_width must be non-negative.')

    if not 0.0 < shrink_factor <= 1.0:
        raise ValueError(
            'shrink_factor must be greater than 0 and at most 1.'
        )

    geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )

    vor = geometry['vor']
    central_indices = geometry['central_indices']

    valid_indices = []
    cell_vertices = {}

    for i, extended_i in enumerate(central_indices):
        if normals[i] is None or thresholds[i] is None:
            continue

        verts = get_cell_vertices(vor, extended_i)

        if verts is None or len(verts) < 4:
            continue

        valid_indices.append(i)
        cell_vertices[i] = verts

    if not valid_indices:
        raise RuntimeError(
            'No finite Voronoi cells are available for plotting.'
        )

    valid_indices = np.asarray(valid_indices, dtype=int)
    valid_points = points[valid_indices]

    midpoint = 0.5 * (
        valid_points.min(axis=0)
        + valid_points.max(axis=0)
    )

    spans = np.maximum(
        valid_points.max(axis=0)
        - valid_points.min(axis=0),
        1e-12,
    )

    normalized = (valid_points - midpoint) / spans

    # Remove cells in the front-right quadrant to expose the interior.
    keep = np.ones(len(valid_indices), dtype=bool)

    if cutaway:
        keep &= ~(
            (normalized[:, 0] > 0.0)
            & (normalized[:, 1] > 0.0)
        )

    valid_indices = valid_indices[keep]
    normalized = normalized[keep]

    # If requested, retain only cells closest to the domain center.
    if max_cells is not None and len(valid_indices) > max_cells:
        score = (
            np.abs(normalized[:, 0])
            + np.abs(normalized[:, 1])
            + 0.35 * np.abs(normalized[:, 2])
        )

        selected = np.argsort(score)[:int(max_cells)]
        valid_indices = valid_indices[selected]

    phase_specs = []

    if phase in ('A', 'both'):
        phase_specs.append(
            (1, 'A phase', "#FF8A5F")
        )

    if phase in ('B', 'both'):
        phase_specs.append(
            (0, 'B phase', "#49A1FF")
        )

    figure = go.Figure()

    if strong_shading:
        mesh_lighting = dict(
            ambient=0.75,
            diffuse=0.6,
            specular=0.05,
            roughness=0.95,
            fresnel=0.04,
        )
        mesh_lightposition = dict(
            x=-100,
            y=180,
            z=220,
        )
    else:
        mesh_lighting = dict(
            ambient=0.75,
            diffuse=0.6,
            specular=0.05,
            roughness=0.95,
            fresnel=0.04,
        )
        mesh_lightposition = dict(
            x=100,
            y=120,
            z=160,
        )

    all_edge_x = []
    all_edge_y = []
    all_edge_z = []

    rendered_piece_count = 0

    for phase_value, phase_name, color in phase_specs:
        mesh_points = []
        mesh_i = []
        mesh_j = []
        mesh_k = []
        mesh_face_colors = []

        offset = 0

        for cell_index in valid_indices:
            clipped = _clip_cell_to_phase(
                cell_vertices[cell_index],
                points[cell_index],
                normals[cell_index],
                thresholds[cell_index],
                phase_value,
            )

            if clipped is None:
                continue

            piece_points, piece_hull = clipped

            # Shrink each phase polyhedron to reveal neighboring boundaries.
            piece_points = _shrink_polyhedron(
                piece_points,
                shrink_factor=shrink_factor,
            )

            try:
                piece_hull = ConvexHull(piece_points)
            except QhullError as error:
                raise RuntimeError(
                    f'Convex-hull construction failed while plotting '
                    f'Janus cell {cell_index}, phase {phase_name}.'
                ) from error

            mesh_points.append(piece_points)

            mesh_i.extend(
                piece_hull.simplices[:, 0] + offset
            )
            mesh_j.extend(
                piece_hull.simplices[:, 1] + offset
            )
            mesh_k.extend(
                piece_hull.simplices[:, 2] + offset
            )

            if strong_shading:
                mesh_face_colors.extend(
                    _make_directional_face_colors(
                        piece_points,
                        piece_hull,
                        color,
                        mesh_lightposition,
                    )
                )

            # Create cell-edge traces only when their width is positive.
            if edge_width > 0.0:
                piece_edges = compute_cell_edges_from_hull(
                    piece_hull
                )

                for edge_start, edge_end in piece_edges:
                    start = piece_points[edge_start]
                    end = piece_points[edge_end]

                    all_edge_x.extend(
                        [start[0], end[0], None]
                    )
                    all_edge_y.extend(
                        [start[1], end[1], None]
                    )
                    all_edge_z.extend(
                        [start[2], end[2], None]
                    )

            offset += len(piece_points)
            rendered_piece_count += 1

        if mesh_points:
            mesh_points = np.vstack(mesh_points)

            figure.add_trace(
                go.Mesh3d(
                    x=mesh_points[:, 0],
                    y=mesh_points[:, 1],
                    z=mesh_points[:, 2],
                    i=mesh_i,
                    j=mesh_j,
                    k=mesh_k,
                    color=color,
                    facecolor=(
                        mesh_face_colors
                        if strong_shading
                        else None
                    ),
                    opacity=opacity,
                    flatshading=True,
                    name=phase_name,
                    showlegend=True,
                    hoverinfo='skip',
                    lighting=mesh_lighting,
                    lightposition=mesh_lightposition,
                )
            )

    # Add subtle cell edges only when requested.
    if edge_width > 0.0 and all_edge_x:
        figure.add_trace(
            go.Scatter3d(
                x=all_edge_x,
                y=all_edge_y,
                z=all_edge_z,
                mode='lines',
                line=dict(
                    color='rgba(25, 30, 35, 0.35)',
                    width=edge_width,
                ),
                name='Cell edges',
                showlegend=False,
                hoverinfo='skip',
            )
        )

    figure.update_layout(
        title=(
            None
            if title is None
            else dict(text=title, x=0.5)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            bgcolor='white',
            camera=dict(
                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=0.95,
                ),
                projection=dict(
                    type='orthographic'
                ),
            ),
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.75)',
        ),
        width=950,
        height=760,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=30 if title else 0,
        ),
    )

    if savepath is not None:
        figure.write_html(
            savepath,
            include_plotlyjs='cdn',
        )

    if show:
        figure.show()

    print(
        f'Voronoi rendering: {len(valid_indices)} cells, '
        f'{rendered_piece_count} phase polyhedra'
    )

    return figure

def plot_single_particle_voronoi_cells(
    points,
    labels,
    periodic_xy=False,
    box_lengths=None,
    phase='both',
    max_cells=None,
    cutaway=False,
    opacity=1.0,
    edge_width=0.6,
    shrink_factor=0.90,
    strong_shading=False,
    title=None,
    savepath=None,
    show=True,
):
    """
    Render the two-species A/B particle model as Voronoi polyhedra.

    label == 1 : A particle
    label == 0 : B particle
    """

    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)

    if phase not in ('A', 'B', 'both'):
        raise ValueError("phase must be 'A', 'B', or 'both'.")

    if len(points) != len(labels):
        raise ValueError(
            'The numbers of points and labels must be equal.'
        )

    if not 0.0 < opacity <= 1.0:
        raise ValueError(
            'opacity must be greater than 0 and at most 1.'
        )

    if not 0.0 < shrink_factor <= 1.0:
        raise ValueError(
            'shrink_factor must be greater than 0 and at most 1.'
        )

    # --------------------------------------------------
    # Voronoi geometry
    # --------------------------------------------------

    geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )

    vor = geometry['vor']
    central_indices = geometry['central_indices']

    valid_indices = []
    cell_vertices = {}

    for i, extended_i in enumerate(central_indices):

        # A negative label denotes a cell excluded from the analysis.
        if labels[i] < 0:
            continue

        verts = get_cell_vertices(
            vor,
            extended_i,
        )

        if verts is None or len(verts) < 4:
            continue

        valid_indices.append(i)
        cell_vertices[i] = verts

    if not valid_indices:
        raise RuntimeError(
            'No finite Voronoi cells are available for plotting.'
        )

    valid_indices = np.asarray(
        valid_indices,
        dtype=int,
    )

    valid_points = points[valid_indices]

    midpoint = 0.5 * (
        valid_points.min(axis=0)
        + valid_points.max(axis=0)
    )

    spans = np.maximum(
        valid_points.max(axis=0)
        - valid_points.min(axis=0),
        1e-12,
    )

    normalized = (
        valid_points - midpoint
    ) / spans

    # --------------------------------------------------
    # Cutaway
    # --------------------------------------------------

    keep = np.ones(
        len(valid_indices),
        dtype=bool,
    )

    if cutaway:

        keep &= ~(
            (normalized[:, 0] > 0.0)
            & (normalized[:, 1] > 0.0)
        )

    valid_indices = valid_indices[keep]
    normalized = normalized[keep]

    # --------------------------------------------------
    # Limit the number of displayed cells only when requested.
    # --------------------------------------------------

    if (
        max_cells is not None
        and len(valid_indices) > max_cells
    ):

        score = (
            np.abs(normalized[:, 0])
            + np.abs(normalized[:, 1])
            + 0.35 * np.abs(normalized[:, 2])
        )

        selected = np.argsort(
            score
        )[:int(max_cells)]

        valid_indices = valid_indices[selected]

    # --------------------------------------------------
    # Use the same colors as in the Janus-cell visualization.
    # --------------------------------------------------

    phase_specs = []

    if phase in ('A', 'both'):
        phase_specs.append(
            (
                1,
                'A particle',
                '#CFE2AE',   # pastel green
            )
        )

    if phase in ('B', 'both'):
        phase_specs.append(
            (
                0,
                'B particle',
                '#EFD3A8',   # pastel orange
            )
        )

    figure = go.Figure()

    if strong_shading:
        mesh_lighting = dict(
            ambient=0.45,
            diffuse=0.95,
            specular=0.08,
            roughness=0.90,
            fresnel=0.04,
        )
        mesh_lightposition = dict(
            x=-100,
            y=180,
            z=220,
        )
    else:
        mesh_lighting = dict(
            ambient=0.82,
            diffuse=0.86,
            specular=0.02,
            roughness=0.99,
            fresnel=0.01,
        )
        mesh_lightposition = dict(
            x=100,
            y=120,
            z=160,
        )

    all_edge_x = []
    all_edge_y = []
    all_edge_z = []

    rendered_cell_count = 0

    # --------------------------------------------------
    # Render A and B particles separately.
    # --------------------------------------------------

    for phase_value, phase_name, color in phase_specs:

        mesh_points = []

        mesh_i = []
        mesh_j = []
        mesh_k = []
        mesh_face_colors = []

        offset = 0

        for cell_index in valid_indices:

            if labels[cell_index] != phase_value:
                continue

            piece_points = np.asarray(
                cell_vertices[cell_index],
                dtype=float,
            )

            # Shrink each Voronoi cell slightly.
            piece_points = _shrink_polyhedron(
                piece_points,
                shrink_factor=shrink_factor,
            )

            try:
                piece_hull = ConvexHull(
                    piece_points
                )

            except QhullError as error:
                raise RuntimeError(
                    f'Convex-hull construction failed while plotting '
                    f'single-particle cell {cell_index}.'
                ) from error

            mesh_points.append(
                piece_points
            )

            mesh_i.extend(
                piece_hull.simplices[:, 0]
                + offset
            )

            mesh_j.extend(
                piece_hull.simplices[:, 1]
                + offset
            )

            mesh_k.extend(
                piece_hull.simplices[:, 2]
                + offset
            )

            if strong_shading:
                mesh_face_colors.extend(
                    _make_directional_face_colors(
                        piece_points,
                        piece_hull,
                        color,
                        mesh_lightposition,
                    )
                )

            # ------------------------------------------
            # Cell edges.
            # ------------------------------------------

            if edge_width > 0.0:

                piece_edges = (
                    compute_cell_edges_from_hull(
                        piece_hull
                    )
                )

                for (
                    edge_start,
                    edge_end,
                ) in piece_edges:

                    start = piece_points[
                        edge_start
                    ]

                    end = piece_points[
                        edge_end
                    ]

                    all_edge_x.extend(
                        [
                            start[0],
                            end[0],
                            None,
                        ]
                    )

                    all_edge_y.extend(
                        [
                            start[1],
                            end[1],
                            None,
                        ]
                    )

                    all_edge_z.extend(
                        [
                            start[2],
                            end[2],
                            None,
                        ]
                    )

            offset += len(
                piece_points
            )

            rendered_cell_count += 1

        # ----------------------------------------------
        # Mesh
        # ----------------------------------------------

        if mesh_points:

            mesh_points = np.vstack(
                mesh_points
            )

            figure.add_trace(
                go.Mesh3d(
                    x=mesh_points[:, 0],
                    y=mesh_points[:, 1],
                    z=mesh_points[:, 2],

                    i=mesh_i,
                    j=mesh_j,
                    k=mesh_k,

                    color=color,
                    facecolor=(
                        mesh_face_colors
                        if strong_shading
                        else None
                    ),
                    opacity=opacity,

                    flatshading=True,

                    name=phase_name,
                    showlegend=True,

                    hoverinfo='skip',

                    lighting=mesh_lighting,
                    lightposition=mesh_lightposition,
                )
            )

    # --------------------------------------------------
    # Cell edges.
    # --------------------------------------------------

    if edge_width > 0.0 and all_edge_x:

        figure.add_trace(
            go.Scatter3d(
                x=all_edge_x,
                y=all_edge_y,
                z=all_edge_z,

                mode='lines',

                line=dict(
                    color=(
                        'rgba(110, 105, 95, 0.18)'
                    ),
                    width=edge_width,
                ),

                name='Cell edges',

                showlegend=False,
                hoverinfo='skip',
            )
        )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    figure.update_layout(

        title=(
            None
            if title is None
            else dict(
                text=title,
                x=0.5,
            )
        ),

        scene=dict(

            xaxis=dict(
                visible=False,
            ),

            yaxis=dict(
                visible=False,
            ),

            zaxis=dict(
                visible=False,
            ),

            # Preserve the physical aspect ratio of the structure.
            aspectmode='data',

            bgcolor='white',

            camera=dict(

                eye=dict(
                    x=1.45,
                    y=1.45,
                    z=0.95,
                ),

                projection=dict(
                    type='orthographic'
                ),
            ),
        ),

        paper_bgcolor='white',
        plot_bgcolor='white',

        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor=(
                'rgba(255,255,255,0.75)'
            ),
        ),

        width=950,
        height=760,

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=30 if title else 0,
        ),
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    if savepath is not None:

        figure.write_html(
            savepath,
            include_plotlyjs='cdn',
        )

    if show:
        figure.show()

    print(
        f'Single-particle Voronoi rendering: '
        f'{rendered_cell_count} cells'
    )

    return figure

def show_slices_paper(volume, xs, ys, zs, n_slices=5, savepath=None, dpi=300):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    if volume.ndim != 3:
        raise ValueError('volume must be a three-dimensional array.')
    nz = volume.shape[0]
    n_slices = int(min(max(1, n_slices), nz))
    indices = np.linspace(0, nz - 1, n_slices, dtype=int)
    indices = np.unique(indices)
    n_display = len(indices)
    cmap = ListedColormap(['#D9D9D9', '#F28E2B', '#4E79A7'])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(1, n_display, figsize=(3.2 * n_display, 3.2), constrained_layout=True, squeeze=False)
    axes = axes.ravel()
    for ax, iz in zip(axes, indices):
        ax.imshow(volume[iz], cmap=cmap, norm=norm, origin='lower', interpolation='nearest', extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect='equal')
        ax.set_title(f'z = {zs[iz]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    legend_handles = [Patch(facecolor='#4E79A7', label='A phase'), Patch(facecolor='#F28E2B', label='B phase'), Patch(facecolor='#D9D9D9', label='Outside')]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_single_phase_clsm(volume, xs, ys, zs, phase=1, smooth_sigma=0.7, opacity=0.12, surface_count=10, cutaway=False, title=None, savepath=None):
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError('volume must be a three-dimensional array.')
    if volume.shape != (len(zs), len(ys), len(xs)):
        raise ValueError('volume.shape must equal (len(zs), len(ys), len(xs)).')
    if phase not in (0, 1):
        raise ValueError('phase must be 0 or 1.')
    if smooth_sigma < 0:
        raise ValueError('smooth_sigma must be non-negative.')
    if not 0.0 < opacity <= 1.0:
        raise ValueError('opacity must be greater than 0 and at most 1.')
    surface_count = max(1, int(surface_count))
    intensity = (volume == phase).astype(np.float32)
    if smooth_sigma > 0:
        intensity = gaussian_filter(intensity, sigma=smooth_sigma)
    if cutaway:
        nx = intensity.shape[2]
        cut_index = int(0.35 * nx)
        intensity[:, :, :cut_index] = 0.0
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
    if title is None:
        phase_name = 'A-phase' if phase == 1 else 'B-phase'
        title = f'Stained {phase_name} void'
    fig = go.Figure(data=go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=intensity.ravel(), isomin=0.15, isomax=0.95, opacity=opacity, surface_count=surface_count, colorscale=[[0.0, '#DDF8FF'], [0.4, '#59D4F5'], [1.0, '#007EA7']], caps=dict(x_show=False, y_show=False, z_show=False), showscale=False))
    fig.update_layout(title=dict(text=title, x=0.5), scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z', aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=0.9))), width=850, height=650, margin=dict(l=0, r=0, b=0, t=60))
    if savepath is not None:
        fig.write_html(savepath, include_plotlyjs='cdn')
    fig.show()
    return fig

def visualize_AB_structure_for_fraction(points, 
                                        target_frac_A, 
                                        nx_vol=100, ny_vol=100, nz_vol=50, 
                                        margin=0.3, seed=None, 
                                        contact_method='exact_shared_face', 
                                        min_contact_area=1e-6, min_boundary_area=0.0, 
                                        n_slices=5, 
                                        save_prefix=None, display_phase=1, 
                                        smooth_sigma=0.7, volume_opacity=0.12, surface_count=6, 
                                        cutaway=False, periodic_xy=False, box_lengths=None, 
                                        render_style='voronoi', cell_phase='both', 
                                        max_display_cells=140, show_slices=True,
                                        create_separate_phase_views=True):
    perA, perB, normals, thresholds = janus_voronoi_percolation_random(points, target_frac_A=target_frac_A, seed=seed, contact_method=contact_method, min_contact_area=min_contact_area, min_boundary_area=min_boundary_area, periodic_xy=periodic_xy, box_lengths=box_lengths)
    volume, xs, ys, zs = make_AB_volume(points, normals, thresholds, nx=nx_vol, ny=ny_vol, nz=nz_vol, margin=margin, periodic_xy=periodic_xy, box_lengths=box_lengths)
    valid_voxels = volume >= 0
    if np.any(valid_voxels):
        actual_voxel_frac_A = np.mean(volume[valid_voxels] == 1)
    else:
        actual_voxel_frac_A = np.nan
    print()
    print('===== Structure to visualize =====')
    print(f'Target A-phase fraction: {target_frac_A:.3f}')
    print(f'Voxel-approximated A-phase fraction: {actual_voxel_frac_A:.3f}')
    print(f'A-phase spanning volume fraction: {perA:.3f}')
    print(f'B-phase spanning volume fraction: {perB:.3f}')
    print(f'Bicontinuity: {perA and perB}')
    print(f'min_contact_area: {min_contact_area}')
    print(f'min_boundary_area: {min_boundary_area}')
    print(f'Periodic boundaries in x and y: {periodic_xy}')
    if save_prefix is None:
        surface_path = None
        surface_path_A = None
        surface_path_B = None
        slice_path = None
    else:
        surface_path = f'{save_prefix}_phiA{target_frac_A:.2f}_3D.html'
        surface_path_A = f'{save_prefix}_phiA{target_frac_A:.2f}_A-phase_3D.html'
        surface_path_B = f'{save_prefix}_phiA{target_frac_A:.2f}_B-phase_3D.html'
        slice_path = f'{save_prefix}_phiA{target_frac_A:.2f}_slices.png'
    if render_style == 'voronoi':
        plot_janus_voronoi_cells(
            points,
            normals,
            thresholds,
            periodic_xy=periodic_xy,
            box_lengths=box_lengths,
            phase=cell_phase,
            max_cells=max_display_cells,
            cutaway=cutaway,
            opacity=1,
            edge_width=0.6,
            shrink_factor=0.9,
            strong_shading=True,
            title=None,
            savepath=surface_path,
        )
        if create_separate_phase_views:
            plot_janus_voronoi_cells(
                points,
                normals,
                thresholds,
                periodic_xy=periodic_xy,
                box_lengths=box_lengths,
                phase='A',
                max_cells=max_display_cells,
                cutaway=cutaway,
                opacity=1,
                edge_width=0.6,
                shrink_factor=0.9,
                strong_shading=True,
                title=None,
                savepath=surface_path_A,
            )
            plot_janus_voronoi_cells(
                points,
                normals,
                thresholds,
                periodic_xy=periodic_xy,
                box_lengths=box_lengths,
                phase='B',
                max_cells=max_display_cells,
                cutaway=cutaway,
                opacity=1,
                edge_width=0.6,
                shrink_factor=0.9,
                strong_shading=True,
                title=None,
                savepath=surface_path_B,
            )
    elif render_style == 'volume':
        plot_single_phase_clsm(
            volume,
            xs,
            ys,
            zs,
            phase=display_phase,
            smooth_sigma=smooth_sigma,
            opacity=volume_opacity,
            surface_count=surface_count,
            cutaway=cutaway,
            title=(
                f"Stained {('A' if display_phase == 1 else 'B')}-phase void, "
                f'phi_A = {actual_voxel_frac_A:.2f}'
            ),
            savepath=surface_path,
        )
    else:
        raise ValueError("render_style must be 'voronoi' or 'volume'.")
    if show_slices:
        show_slices_paper(volume, xs, ys, zs, n_slices=n_slices, savepath=slice_path)
    return (volume, xs, ys, zs, perA, perB)

def mean_confidence_interval(values, confidence_z=1.96):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    mean_value = float(np.mean(values))
    if len(values) == 1:
        return (mean_value, mean_value)
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    half_width = confidence_z * standard_error
    return (
        float(np.clip(mean_value - half_width, 0.0, 1.0)),
        float(np.clip(mean_value + half_width, 0.0, 1.0)),
    )

def plot_spanning_volume_fractions(results, savepath=None, show_confidence_interval=True):
    if len(results) == 0:
        raise ValueError('results must not be empty.')
    frac_values = np.asarray([result['frac_A'] for result in results], dtype=float)
    series = [
        ('spanning_volume_fraction_A', 'A_low', 'A_high', 'A-phase spanning volume', '#4E79A7'),
        ('spanning_volume_fraction_B', 'B_low', 'B_high', 'B-phase spanning volume', '#F28E2B'),
        ('bicontinuous_volume_fraction', 'both_low', 'both_high', 'Bicontinuous volume', '#59A14F'),
    ]
    order = np.argsort(frac_values)
    frac_values = frac_values[order]
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    for probability_key, low_key, high_key, label, color in series:
        probabilities = np.asarray(
            [result[probability_key] for result in results], dtype=float
        )[order]
        lower_limits = np.asarray(
            [result[low_key] for result in results], dtype=float
        )[order]
        upper_limits = np.asarray(
            [result[high_key] for result in results], dtype=float
        )[order]
        ax.plot(
            frac_values,
            probabilities,
            color=color,
            marker='o',
            markersize=5.0,
            linewidth=2.0,
            label=label,
            zorder=3,
        )
        if show_confidence_interval:
            ax.fill_between(
                frac_values,
                lower_limits,
                upper_limits,
                color=color,
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
    ax.set_xlabel('A-phase volume fraction, $\\phi_A$')
    ax.set_ylabel('Spanning volume fraction')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, loc='best')
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    return (fig, ax)

def save_volume_fraction_results(results, savepath):
    if savepath is None:
        return
    fieldnames = [
        'frac_A',
        'spanning_volume_fraction_A',
        'A_low',
        'A_high',
        'spanning_volume_fraction_B',
        'B_low',
        'B_high',
        'bicontinuous_volume_fraction',
        'both_low',
        'both_high',
    ]
    with open(savepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def run_janus_fraction_sweep(trials=500, nx_cell=10, ny_cell=10, nz_cell=5, fractions=None, lattice_type='fcc', a=1.0, min_contact_area=1e-6, min_boundary_area=0.0, random_seed=None, graph_savepath=None, data_savepath=None, periodic_xy=True):
    if trials < 1:
        raise ValueError('trials must be at least 1.')
    if fractions is None:
        fractions = np.arange(0.1, 0.901, 0.05)
    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or len(fractions) == 0:
        raise ValueError('fractions must be a non-empty one-dimensional array.')
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError('Every fraction must be between 0 and 1.')
    lattice_type = lattice_type.lower()
    points = generate_points(lattice_type=lattice_type, a=a, nx=nx_cell, ny=ny_cell, nz=nz_cell)
    box_lengths = get_lattice_box_lengths(
        lattice_type=lattice_type,
        a=a,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    prepared_geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )
    rng = np.random.default_rng(random_seed)
    trial_seeds = rng.integers(low=0, high=np.iinfo(np.uint32).max, size=trials, dtype=np.uint32)
    total_calculations = len(fractions) * trials
    print(f'Calculation started: {len(fractions)} fractions x {trials} trials = {total_calculations} calculations')
    print(f'Lattice: {lattice_type.upper()}')
    print(f'Lattice constant: {a}')
    print(f'Number of particle centers: {len(points)}')
    print(f'Periodic boundaries in x and y: {periodic_xy}')
    print(f'Box lengths: {box_lengths}')
    results = []
    total_start = time.perf_counter()
    progress_interval = max(1, trials // 10)
    for fraction_index, target_frac_A in enumerate(fractions, start=1):
        fraction_start = time.perf_counter()
        A_spanning_values = []
        B_spanning_values = []
        bicontinuous_values = []
        for trial in range(trials):
            spanning_A, spanning_B, normals, thresholds = janus_voronoi_percolation_random(points, target_frac_A=target_frac_A, seed=int(trial_seeds[trial]), contact_method='exact_shared_face', min_contact_area=min_contact_area, min_boundary_area=min_boundary_area, periodic_xy=periodic_xy, box_lengths=box_lengths, prepared_geometry=prepared_geometry)
            if not np.isfinite(spanning_A) or not 0.0 <= spanning_A <= 1.0:
                raise ValueError('A-phase spanning volume fraction must be between 0 and 1.')
            if not np.isfinite(spanning_B) or not 0.0 <= spanning_B <= 1.0:
                raise ValueError('B-phase spanning volume fraction must be between 0 and 1.')
            if len(normals) != len(points) or len(thresholds) != len(points):
                raise ValueError('The numbers of cells, normals, and thresholds do not match.')
            bicontinuous_fraction = min(spanning_A, spanning_B)
            A_spanning_values.append(spanning_A)
            B_spanning_values.append(spanning_B)
            bicontinuous_values.append(bicontinuous_fraction)
            completed = trial + 1
            if completed % progress_interval == 0 or completed == trials:
                print(
                    f'[{fraction_index}/{len(fractions)}] '
                    f'phi_A={target_frac_A:.2f}: '
                    f'{completed}/{trials} trials completed '
                    f'(mean A {np.mean(A_spanning_values):.3f}, '
                    f'B {np.mean(B_spanning_values):.3f}, '
                    f'bicontinuous {np.mean(bicontinuous_values):.3f})',
                    flush=True,
                )
        mean_A = float(np.mean(A_spanning_values))
        mean_B = float(np.mean(B_spanning_values))
        mean_bicontinuous = float(np.mean(bicontinuous_values))
        A_low, A_high = mean_confidence_interval(A_spanning_values)
        B_low, B_high = mean_confidence_interval(B_spanning_values)
        both_low, both_high = mean_confidence_interval(bicontinuous_values)
        results.append({'frac_A': target_frac_A, 'spanning_volume_fraction_A': mean_A, 'A_low': A_low, 'A_high': A_high, 'spanning_volume_fraction_B': mean_B, 'B_low': B_low, 'B_high': B_high, 'bicontinuous_volume_fraction': mean_bicontinuous, 'both_low': both_low, 'both_high': both_high})
        save_volume_fraction_results(results, data_savepath)
        fraction_elapsed = time.perf_counter() - fraction_start
        print(f'Result: phi_A={target_frac_A:.2f} | spanning A={mean_A:.3f} | spanning B={mean_B:.3f} | bicontinuous={mean_bicontinuous:.3f} | bicontinuous 95% CI=[{both_low:.3f}, {both_high:.3f}] | {fraction_elapsed:.1f} s')
    total_elapsed = time.perf_counter() - total_start
    plot_spanning_volume_fractions(results, savepath=graph_savepath, show_confidence_interval=True)
    print(f'Completed in {total_elapsed:.1f} s')
    if data_savepath is not None:
        print(f'Numerical data: {data_savepath}')
    if graph_savepath is not None:
        print(f'Graph: {graph_savepath}')
    return results

def run_single_particle_fraction_sweep(
    trials=500,
    nx_cell=10,
    ny_cell=10,
    nz_cell=5,
    fractions=None,
    lattice_type='fcc',
    a=1.0,
    min_contact_area=1e-6,
    min_boundary_area=0.0,
    random_seed=None,
    graph_savepath=None,
    data_savepath=None,
    periodic_xy=True,
):
    """Calculate spanning volume fractions for randomly mixed A/B particles.

    Because all particles have equal volume, frac_A is both the number fraction
    and volume fraction of A particles. The A-particle count is fixed per trial.
    """
    if trials < 1:
        raise ValueError('trials must be at least 1.')
    if fractions is None:
        fractions = np.arange(0.1, 0.901, 0.05)
    fractions = np.asarray(fractions, dtype=float)
    if fractions.ndim != 1 or len(fractions) == 0:
        raise ValueError('fractions must be a non-empty one-dimensional array.')
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError('Every fraction must be between 0 and 1.')

    lattice_type = lattice_type.lower()
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
    prepared_geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )
    rng = np.random.default_rng(random_seed)
    trial_seeds = rng.integers(
        low=0,
        high=np.iinfo(np.uint32).max,
        size=trials,
        dtype=np.uint32,
    )

    total_calculations = len(fractions) * trials
    print(
        f'Single-particle calculation started: '
        f'{len(fractions)} fractions x {trials} trials = {total_calculations} calculations'
    )
    print(f'Lattice: {lattice_type.upper()}')
    print(f'Lattice constant: {a}')
    print(f'Number of particle centers: {len(points)}')
    print(f'Periodic boundaries in x and y: {periodic_xy}')
    print(f'Box lengths: {box_lengths}')
    print(f'Minimum contact area: {min_contact_area}')
    print(f'Minimum boundary contact area: {min_boundary_area}')

    results = []
    total_start = time.perf_counter()
    progress_interval = max(1, trials // 10)
    for fraction_index, target_frac_A in enumerate(fractions, start=1):
        fraction_start = time.perf_counter()
        A_spanning_values = []
        B_spanning_values = []
        bicontinuous_values = []
        actual_frac_A = np.nan
        for trial in range(trials):
            spanning_A, spanning_B, labels = single_particle_percolation_random(
                points,
                target_frac_A=target_frac_A,
                seed=int(trial_seeds[trial]),
                min_contact_area=min_contact_area,
                min_boundary_area=min_boundary_area,
                periodic_xy=periodic_xy,
                box_lengths=box_lengths,
                prepared_geometry=prepared_geometry,
            )
            valid_labels = labels[labels >= 0]
            actual_frac_A = np.mean(valid_labels == 1)
            if not np.isfinite(spanning_A) or not 0.0 <= spanning_A <= 1.0:
                raise ValueError('A-phase spanning volume fraction must be between 0 and 1.')
            if not np.isfinite(spanning_B) or not 0.0 <= spanning_B <= 1.0:
                raise ValueError('B-phase spanning volume fraction must be between 0 and 1.')
            bicontinuous_fraction = min(spanning_A, spanning_B)
            A_spanning_values.append(spanning_A)
            B_spanning_values.append(spanning_B)
            bicontinuous_values.append(bicontinuous_fraction)
            completed = trial + 1
            if completed % progress_interval == 0 or completed == trials:
                print(
                    f'[{fraction_index}/{len(fractions)}] '
                    f'phi_A={target_frac_A:.2f}: '
                    f'{completed}/{trials} trials completed '
                    f'(mean A {np.mean(A_spanning_values):.3f}, '
                    f'B {np.mean(B_spanning_values):.3f}, '
                    f'bicontinuous {np.mean(bicontinuous_values):.3f})',
                    flush=True,
                )

        mean_A = float(np.mean(A_spanning_values))
        mean_B = float(np.mean(B_spanning_values))
        mean_bicontinuous = float(np.mean(bicontinuous_values))
        A_low, A_high = mean_confidence_interval(A_spanning_values)
        B_low, B_high = mean_confidence_interval(B_spanning_values)
        both_low, both_high = mean_confidence_interval(bicontinuous_values)
        results.append({
            'frac_A': target_frac_A,
            'actual_frac_A': actual_frac_A,
            'spanning_volume_fraction_A': mean_A,
            'A_low': A_low,
            'A_high': A_high,
            'spanning_volume_fraction_B': mean_B,
            'B_low': B_low,
            'B_high': B_high,
            'bicontinuous_volume_fraction': mean_bicontinuous,
            'both_low': both_low,
            'both_high': both_high,
        })
        save_single_particle_volume_fraction_results(results, data_savepath)
        fraction_elapsed = time.perf_counter() - fraction_start
        print(
            f'Result: phi_A={target_frac_A:.2f} '
            f'(actual {actual_frac_A:.3f}) | '
            f'spanning A={mean_A:.3f} | spanning B={mean_B:.3f} | '
            f'bicontinuous={mean_bicontinuous:.3f} | '
            f'95% CI=[{both_low:.3f}, {both_high:.3f}] | '
            f'{fraction_elapsed:.1f} s'
        )

    total_elapsed = time.perf_counter() - total_start
    plot_spanning_volume_fractions(
        results,
        savepath=graph_savepath,
        show_confidence_interval=True,
    )
    print(f'Completed in {total_elapsed:.1f} s')
    if data_savepath is not None:
        print(f'Numerical data: {data_savepath}')
    if graph_savepath is not None:
        print(f'Graph: {graph_savepath}')
    return results

def save_single_particle_volume_fraction_results(results, savepath):
    if savepath is None:
        return
    fieldnames = [
        'frac_A',
        'actual_frac_A',
        'spanning_volume_fraction_A',
        'A_low',
        'A_high',
        'spanning_volume_fraction_B',
        'B_low',
        'B_high',
        'bicontinuous_volume_fraction',
        'both_low',
        'both_high',
    ]
    with open(savepath, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def visualize_single_particle_structure(
    target_frac_A=0.5,
    random_seed=None,
    lattice_type='fcc',
    a=1.0,
    nx_cell=10,
    ny_cell=10,
    nz_cell=5,
    periodic_xy=True,
    save_prefix=None,
):
    """Visualize a random A/B single-particle mixture and percolation result."""
    lattice_type = lattice_type.lower()
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
    prepared_geometry = prepare_voronoi_geometry(
        points,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )
    perA, perB, labels = single_particle_percolation_random(
        points,
        target_frac_A=target_frac_A,
        seed=random_seed,
        min_contact_area=1e-6,
        min_boundary_area=0.0,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
        prepared_geometry=prepared_geometry,
    )
    volume, xs, ys, zs = make_single_particle_volume(
        points,
        labels,
        nx=48,
        ny=48,
        nz=24,
        margin=0.3,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
    )
    valid_labels = labels[labels >= 0]
    actual_frac_A = np.mean(valid_labels == 1)
    print()
    print('===== A/B single-particle structure =====')
    print(f'Target A-particle fraction: {target_frac_A:.3f}')
    print(f'Actual A-particle fraction: {actual_frac_A:.3f}')
    print(f'A-phase spanning volume fraction: {perA:.3f}')
    print(f'B-phase spanning volume fraction: {perB:.3f}')
    print(f'Bicontinuity: {perA and perB}')

    if save_prefix is None:
        slice_path = None
        surface_path = None
        surface_path_A = None
        surface_path_B = None
    else:
        slice_path = (
            f'{save_prefix}_single_particles_phiA'
            f'{target_frac_A:.2f}_slices.png'
        )
        surface_path = (
            f'{save_prefix}_single_particles_phiA'
            f'{target_frac_A:.2f}_3D.html'
        )
        surface_path_A = (
            f'{save_prefix}_single_particles_phiA'
            f'{target_frac_A:.2f}_A-phase_3D.html'
        )
        surface_path_B = (
            f'{save_prefix}_single_particles_phiA'
            f'{target_frac_A:.2f}_B-phase_3D.html'
        )
    for phase, phase_path in (
        ('both', surface_path),
        ('A', surface_path_A),
        ('B', surface_path_B),
    ):
        plot_single_particle_voronoi_cells(
            points,
            labels,
            periodic_xy=periodic_xy,
            box_lengths=box_lengths,
            phase=phase,
            max_cells=None,
            cutaway=False,
            opacity=1.0,
            edge_width=0.6,
            shrink_factor=0.90,
            strong_shading=True,
            title=None,
            savepath=phase_path,
        )
    return (volume, xs, ys, zs, perA, perB, labels)

def visualize_AB_structure_check(random_seed=None, lattice_type='hcp', a=1.0, nx_cell=20, ny_cell=12, nz_cell=10, periodic_xy=True, save_prefix=None):
    lattice_type = lattice_type.lower()
    points = generate_points(lattice_type=lattice_type, a=a, nx=nx_cell, ny=ny_cell, nz=nz_cell)
    box_lengths = get_lattice_box_lengths(
        lattice_type=lattice_type,
        a=a,
        nx=nx_cell,
        ny=ny_cell,
        nz=nz_cell,
    )
    volume, xs, ys, zs, perA, perB = visualize_AB_structure_for_fraction(
        points,
        target_frac_A=0.5,
        nx_vol=48,
        ny_vol=48,
        nz_vol=24,
        margin=0.3,
        seed=random_seed,
        contact_method='exact_shared_face',
        min_contact_area=1e-6,
        min_boundary_area=0.0,
        n_slices=5,
        save_prefix=save_prefix,
        periodic_xy=periodic_xy,
        box_lengths=box_lengths,
        render_style='voronoi',
        cell_phase='both',
        max_display_cells=None,
        cutaway=False,
        show_slices=False,
    )
    print('volume.shape:', volume.shape)
    print('A-phase spanning volume fraction:', perA)
    print('B-phase spanning volume fraction:', perB)
    print('Bicontinuous:', perA and perB)
    assert volume.shape == (24, 48, 48)
    assert np.all(np.isin(volume, [-1, 0, 1]))
    print('OK: visualize_AB_structure_check')

if __name__ == "__main__":
    # Uncomment only one operation to run.

    # Use one timestamp for all outputs from this execution so that repeated
    # runs do not overwrite earlier CSV and PNG files.
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Visualize a Janus-particle structure for inspection.
    visualize_AB_structure_check(
        random_seed=0,
        lattice_type="hcp",
        nx_cell=10,
        ny_cell=6,
        nz_cell=6,
        periodic_xy=True,
        save_prefix="janus_voronoi_cells",
    )

    # 2. Visualize the two-species single-particle model.
    '''visualize_single_particle_structure(
        target_frac_A=0.5,
        random_seed=0,
        lattice_type='hcp',
        nx_cell=10,
        ny_cell=6,
        nz_cell=6,
        periodic_xy=True,
        save_prefix='single_particle_voronoi_cells',
    )'''

    # 3. Production sweep for the Janus-particle model.
    '''run_janus_fraction_sweep(
        trials=100,
        nx_cell=6,
        ny_cell=6,
        nz_cell=7,
        fractions=np.arange(0.0, 1.001, 0.1),
        lattice_type="hcp",
        a=1.0,
        min_contact_area=1e-6,
        min_boundary_area=0.0,
        random_seed=0,
        periodic_xy=True,
        graph_savepath=f"janus_spanning_volume_fractions_hcp_{run_id}.png",
        data_savepath=f"janus_spanning_volume_fractions_hcp_{run_id}.csv",
    )'''

    # 4. Production sweep for the two-species single-particle model.
    '''run_single_particle_fraction_sweep(
        trials=100,
        nx_cell=6,
        ny_cell=6,
        nz_cell=7,
        fractions=np.arange(0.0, 1.001, 0.1),
        lattice_type="hcp",
        a=1.0,
        min_contact_area=1e-6,
        min_boundary_area=0.0,
        random_seed=0,
        periodic_xy=True,
        graph_savepath=f"single_particle_spanning_volume_fractions_hcp_{run_id}.png",
        data_savepath=f"single_particle_spanning_volume_fractions_hcp_{run_id}.csv",
    )'''
