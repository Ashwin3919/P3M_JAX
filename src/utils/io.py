import os
import numpy as np
from pathlib import Path

def _to_xyz(data):
    """Ensure particle data has shape (N, 3), padding z=0 for 2D input."""
    data = np.asarray(data)
    if data.shape[1] == 3:
        return data.astype(np.float32)
    n = data.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, :2] = data
    return out

def write_vtk_particles(pos, mom, a_value, results_dir, config_name):
    """Write particle snapshot as a Legacy Binary VTK PolyData file."""
    vtk_particles_dir = os.path.join(results_dir, 'vtk', 'particles')
    os.makedirs(vtk_particles_dir, exist_ok=True)

    a_str = f"{a_value:.3f}".replace('.', '')
    filename = f"{config_name}_particles_a{a_str}.vtk"
    filepath = Path(os.path.join(vtk_particles_dir, filename))

    pos_3d = _to_xyz(np.array(pos))
    mom_3d = _to_xyz(np.array(mom))
    n = len(pos_3d)

    with filepath.open("wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(f"P3M-JAX Particles a={a_value:.4f}\n".encode())
        f.write(b"BINARY\n")
        f.write(b"DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n} float\n".encode())
        f.write(pos_3d.astype(">f4").tobytes())
        f.write(b"\n")

        # Vertices — build in native int32, convert whole array to big-endian once
        f.write(f"VERTICES {n} {2 * n}\n".encode())
        verts = np.empty((n, 2), dtype=np.int32)
        verts[:, 0] = 1
        verts[:, 1] = np.arange(n, dtype=np.int32)
        f.write(verts.astype(">i4").tobytes())
        f.write(b"\n")

        # Point Data (Momenta)
        f.write(f"POINT_DATA {n}\n".encode())
        f.write(b"VECTORS momentum float\n")
        f.write(mom_3d.astype(">f4").tobytes())
        f.write(b"\n")
    return str(filepath)

def write_vtk_density(rho, box, a_value, results_dir, config_name):
    """Write density field to VTK (Structured Points) in vtk/density/.

    Works for both 2D (N×N) and 3D (N×N×N) density arrays.
    """
    vtk_density_dir = os.path.join(results_dir, 'vtk', 'density')
    os.makedirs(vtk_density_dir, exist_ok=True)

    a_str = f"{a_value:.3f}".replace('.', '')
    filename = f"{config_name}_density_a{a_str}.vtk"
    filepath = os.path.join(vtk_density_dir, filename)

    rho = np.asarray(rho)
    res = box.res

    if rho.ndim == 2:
        nx, ny = rho.shape
        nz = 1
        # Non-zero z-spacing prevents degenerate cells that render as grid boxes
        spacing = f"{res} {res} {res}"
    else:
        nx, ny, nz = rho.shape
        spacing = f"{res} {res} {res}"

    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Density field at a={a_value:.3f}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write(f"ORIGIN 0 0 0\n")
        f.write(f"SPACING {spacing}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write("SCALARS density float 1\n")
        f.write("LOOKUP_TABLE default\n")
        # VTK STRUCTURED_POINTS expects x-fastest (Fortran) ordering
        for val in rho.flatten(order='F'):
            f.write(f"{val}\n")

    return filepath
