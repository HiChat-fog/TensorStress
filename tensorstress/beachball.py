"""
Focal Mechanism Beachball Visualization.

Pure mathematical projection of a moment tensor onto a sphere.
Zero data dependencies — only requires strike/dip/rake or a moment tensor.

Based on the standard lower-hemisphere equal-area (Schmidt) projection
used in global seismology for over 50 years.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def _radiation_pattern(mt, azimuth, takeoff):
    """P-wave radiation amplitude at given direction.

    mt: 3x3 moment tensor
    azimuth: angle in radians from north (0 to 2*pi)
    takeoff: angle in radians from downward vertical (0 to pi)

    Returns: scalar radiation amplitude (positive = compression)
    """
    # Ray direction (from source toward station)
    n = np.array([
        np.sin(takeoff) * np.sin(azimuth),  # x (north)
        np.sin(takeoff) * np.cos(azimuth),  # y (east)
        np.cos(takeoff),                    # z (down)
    ])
    return n @ mt @ n


def _compute_nodal_contour(mt, n_phi=360, n_theta=180):
    """Find the nodal lines where radiation = 0.

    Returns phi (azimuth) coordinates of the zero-crossing at each theta.
    """
    phi_vals = np.linspace(0, 2 * np.pi, n_phi)
    theta_vals = np.linspace(0, np.pi, n_theta)

    radiation = np.zeros((n_theta, n_phi))
    for i, th in enumerate(theta_vals):
        for j, ph in enumerate(phi_vals):
            radiation[i, j] = _radiation_pattern(mt, ph, th)

    # Find zero crossings at each theta
    contours = []
    for i in range(n_theta):
        signs = np.sign(radiation[i, :])
        crossings = np.where(np.diff(signs) != 0)[0]
        for c in crossings:
            # Linear interpolation for more accurate crossing position
            r1, r2 = radiation[i, c], radiation[i, c + 1]
            if abs(r1 - r2) > 1e-12:
                frac = -r1 / (r2 - r1)
                phi_cross = phi_vals[c] + frac * (phi_vals[c + 1] - phi_vals[c])
            else:
                phi_cross = phi_vals[c]
            contours.append((theta_vals[i], phi_cross))

    return np.array(contours) if contours else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# Equal-area (Schmidt) projection
# ---------------------------------------------------------------------------

def _schmidt_projection(theta, phi):
    """Map (theta, phi) on the lower hemisphere to (x, y) in 2D.

    Equal-area projection: R = sqrt(2) * sin(theta/2)
    """
    # Only lower hemisphere (takeoff angle > pi/2... wait,
    # in our convention takeoff from downward vertical:
    # lower hemisphere = theta in [0, pi/2] (upward rays)
    # Actually for beachball we use the full sphere and project
    R = np.sqrt(2) * np.sin(theta / 2)
    x = R * np.sin(phi)
    y = R * np.cos(phi)
    return x, y


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_beachball(strike, dip, rake, ax=None, facecolor='#333333',
                    edgecolor='black', linewidth=1.5, bg_color='white',
                    compression_color='white', dilatation_color='#333333',
                    resolution=200):
    """Plot a focal mechanism beachball.

    Parameters:
        strike, dip, rake: fault geometry angles in degrees
        ax: matplotlib axes (creates new figure if None)
        facecolor: background fill of the circle
        compression_color: color for compressional quadrants
        dilatation_color: color for dilatational quadrants
        resolution: grid resolution (higher = smoother)

    Returns: matplotlib axes
    """
    import sys
    sys.path.insert(0, '.')
    from tensorstress.moment_tensor import mt_from_sdr

    mt = mt_from_sdr(strike, dip, rake)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Generate grid on the focal sphere
    phi = np.linspace(0, 2 * np.pi, resolution)
    theta = np.linspace(0, np.pi, resolution)
    PHI, THETA = np.meshgrid(phi, theta)

    # Compute radiation pattern
    radiation = np.zeros_like(PHI)
    for i in range(resolution):
        for j in range(resolution):
            radiation[i, j] = _radiation_pattern(mt, PHI[i, j], THETA[i, j])

    # Equal-area projection
    X, Y = _schmidt_projection(THETA, PHI)

    # Plot compressional and dilatational quadrants
    comp_mask = radiation > 0
    dil_mask = radiation < 0

    # Plot as filled contours
    ax.contourf(X, Y, radiation, levels=[-1e10, 0, 1e10],
                colors=[dilatation_color, compression_color],
                antialiased=True)

    # Draw nodal lines (where radiation = 0)
    ax.contour(X, Y, radiation, levels=[0], colors=edgecolor,
               linewidths=linewidth, antialiased=True)

    # Draw the outer circle
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    r_circle = np.sqrt(2) * np.sin(np.pi / 4)  # R at theta = pi/2
    ax.plot(r_circle * np.sin(theta_circle),
            r_circle * np.cos(theta_circle),
            color=edgecolor, linewidth=linewidth)

    # Annotations: N, E, S, W
    r_annot = r_circle + 0.08
    ax.text(0, r_annot, 'N', ha='center', va='bottom', fontsize=9)
    ax.text(0, -r_annot, 'S', ha='center', va='top', fontsize=9)
    ax.text(r_annot, 0, 'E', ha='left', va='center', fontsize=9)
    ax.text(-r_annot, 0, 'W', ha='right', va='center', fontsize=9)

    # P and T axes
    _plot_axes(mt, ax, r_circle)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-r_circle - 0.15, r_circle + 0.15)
    ax.set_ylim(-r_circle - 0.15, r_circle + 0.15)

    return ax


def _plot_axes(mt, ax, r_max):
    """Plot P (pressure) and T (tension) axes."""
    eigenvalues, eigenvectors = np.linalg.eigh(mt)
    # Sort: most compressive first (P axis = most negative eigenvalue)
    idx = np.argsort(eigenvalues)
    # T axis = most positive (largest eigenvalue)
    t_vec = eigenvectors[:, idx[2]]
    # P axis = most negative (smallest eigenvalue)
    p_vec = eigenvectors[:, idx[0]]

    for vec, marker, color, label in [
        (t_vec, 'o', '#d62728', 'T'),
        (-t_vec, 'o', '#d62728', None),
        (p_vec, 's', '#1f77b4', 'P'),
        (-p_vec, 's', '#1f77b4', None),
    ]:
        v = vec / np.linalg.norm(vec)
        # Determine if lower hemisphere
        if v[2] > 0:  # upward — project to opposite point on lower hemisphere
            v = -v
        theta_proj = np.arccos(np.clip(v[2], -1, 1))
        phi_proj = np.arctan2(v[0], v[1])
        x, y = _schmidt_projection(theta_proj, phi_proj)
        if abs(x) < r_max and abs(y) < r_max:
            ax.plot(x, y, marker=marker, color=color, markersize=4,
                    markeredgecolor='black', markeredgewidth=0.5)
            if label:
                ax.annotate(label, (x, y), xytext=(x + 0.04, y + 0.04),
                            fontsize=7, color=color, fontweight='bold')


# ---------------------------------------------------------------------------
# Multi-beachball grid (for comparison)
# ---------------------------------------------------------------------------

def plot_beachball_grid(mechanisms, n_cols=4, figsize=None):
    """Plot multiple beachballs in a grid for comparison.

    mechanisms: list of (strike, dip, rake) tuples
    """
    n = len(mechanisms)
    n_cols = min(n_cols, n)
    n_rows = (n + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3 * n_cols, 3 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    type_names = ['Thrust', 'Normal', 'Strike-Slip',
                  'Oblique Reverse', 'Oblique Normal']

    for i, (s, d, r) in enumerate(mechanisms):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        plot_beachball(s, d, r, ax=ax)
        # Classify mechanism type
        if r > 135 or r < -135:
            mtype = 'Normal'
        elif abs(r) < 45:
            mtype = 'Strike-Slip'
        elif r > 45:
            mtype = 'Thrust'
        else:
            mtype = 'Oblique'
        ax.set_title(f'({s}°, {d}°, {r}°)\n{mtype}', fontsize=8)

    # Hide unused axes
    for i in range(n, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig
