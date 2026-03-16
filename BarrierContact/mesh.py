"""Mesh dataclass and MeshLoader for 2D FEM simulations.

Provides:
- ``Mesh`` dataclass holding vertices, triangles, and boundary edges.
- ``MeshLoader`` with static factory methods for creating meshes from
  structured grids, polygon boundaries (via ``triangle`` library), or files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray


@dataclass
class Mesh:
    """Triangular mesh for 2D FEM.

    Parameters
    ----------
    vertices : ndarray, shape (n_nodes, 2)
        Node coordinates.
    triangles : ndarray, shape (n_tri, 3), dtype int
        Triangle connectivity (CCW orientation).
    boundary_edges : ndarray, shape (n_be, 2), dtype int
        Boundary edge connectivity.
    """

    vertices: ndarray
    triangles: ndarray
    boundary_edges: ndarray

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        return self.vertices.shape[0]

    @property
    def n_triangles(self) -> int:
        """Number of triangles."""
        return self.triangles.shape[0]

    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom (2 per node)."""
        return 2 * self.n_nodes

    def find_boundary_nodes(self) -> ndarray:
        """Return sorted array of unique node indices on boundary edges."""
        return np.unique(self.boundary_edges)

    def find_bottom_nodes(self, tol: float = 1e-8) -> ndarray:
        """Return node indices at the minimum y-coordinate (within tolerance)."""
        y_min = self.vertices[:, 1].min()
        return np.where(self.vertices[:, 1] <= y_min + tol)[0]


class MeshLoader:
    """Factory methods for creating ``Mesh`` instances."""

    @staticmethod
    def rectangle(width: float, height: float, nx: int, ny: int) -> Mesh:
        """Generate a structured triangular mesh for a rectangle centered at the origin.

        Parameters
        ----------
        width, height : float
            Rectangle dimensions.
        nx, ny : int
            Number of cells (quads) in x and y directions.

        Returns
        -------
        Mesh
            Structured mesh with ``(nx+1)*(ny+1)`` nodes and ``2*nx*ny`` triangles.
        """
        nv_x = nx + 1
        nv_y = ny + 1
        n_nodes = nv_x * nv_y

        # Node positions on a regular grid
        X_ref = np.zeros((n_nodes, 2))
        for j in range(nv_y):
            for i in range(nv_x):
                idx = j * nv_x + i
                X_ref[idx, 0] = -width / 2.0 + i * width / nx
                X_ref[idx, 1] = -height / 2.0 + j * height / ny

        # Triangulate: each quad split into 2 triangles (CCW)
        triangles = []
        for j in range(ny):
            for i in range(nx):
                n00 = j * nv_x + i
                n10 = j * nv_x + i + 1
                n01 = (j + 1) * nv_x + i
                n11 = (j + 1) * nv_x + i + 1
                # Lower-left triangle
                triangles.append([n00, n10, n11])
                # Upper-right triangle
                triangles.append([n00, n11, n01])
        triangles = np.array(triangles, dtype=int)

        # Bottom boundary edges (j=0 row)
        boundary_edges = []
        for i in range(nx):
            boundary_edges.append([i, i + 1])
        boundary_edges = np.array(boundary_edges, dtype=int)

        return Mesh(vertices=X_ref, triangles=triangles, boundary_edges=boundary_edges)

    @staticmethod
    def from_polygon(
        boundary_pts: ndarray,
        max_area: float = 0.1,
        min_angle: float = 20.0,
    ) -> Mesh:
        """Create a mesh from a polygon boundary using constrained Delaunay triangulation.

        Requires the ``triangle`` library (``pip install triangle``).

        Parameters
        ----------
        boundary_pts : ndarray, shape (n, 2)
            Boundary polygon vertices in order (CCW recommended).
        max_area : float
            Maximum triangle area constraint.
        min_angle : float
            Minimum angle constraint (degrees).

        Returns
        -------
        Mesh
        """
        import triangle  # noqa: I001

        n = len(boundary_pts)
        segments = np.array([[i, (i + 1) % n] for i in range(n)], dtype=int)

        tri_input = {
            "vertices": np.asarray(boundary_pts, dtype=float),
            "segments": segments,
        }
        tri_output = triangle.triangulate(tri_input, f"pq{min_angle}a{max_area}")

        vertices = tri_output["vertices"]
        triangles = tri_output["triangles"]

        boundary_edges = tri_output.get("segments", segments)

        return Mesh(
            vertices=vertices,
            triangles=triangles,
            boundary_edges=boundary_edges,
        )

    @staticmethod
    def from_file(path: str | Path) -> Mesh:
        """Load a mesh from a file (.msh, .obj, etc.) using meshio.

        Requires the ``meshio`` library.

        Parameters
        ----------
        path : str or Path
            Path to the mesh file.

        Returns
        -------
        Mesh
        """
        import meshio

        mio = meshio.read(str(path))
        vertices = mio.points[:, :2]  # take x, y only

        # Collect triangle cells
        tri_blocks = []
        for cell_block in mio.cells:
            if cell_block.type == "triangle":
                tri_blocks.append(cell_block.data)

        if not tri_blocks:
            msg = f"No triangle cells found in {path}"
            raise ValueError(msg)

        triangles = np.concatenate(tri_blocks, axis=0)

        # Extract boundary edges: edges that appear in only one triangle
        edge_count: dict[tuple[int, int], int] = {}
        for tri in triangles:
            for k in range(3):
                e = (int(tri[k]), int(tri[(k + 1) % 3]))
                canonical = (min(e), max(e))
                edge_count[canonical] = edge_count.get(canonical, 0) + 1

        boundary_edges = np.array(
            [list(e) for e, count in edge_count.items() if count == 1], dtype=int
        )

        return Mesh(
            vertices=vertices,
            triangles=triangles,
            boundary_edges=boundary_edges,
        )
