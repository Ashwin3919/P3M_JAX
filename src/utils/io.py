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

    # Convert 2D JAX/NumPy arrays to 3D NumPy for VTK
    pos_3d = _to_xyz(np.array(pos))
    mom_3d = _to_xyz(np.array(mom))
    n = len(pos_3d)

    with filepath.open("wb") as f:
        # Header
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(f"P3M-JAX Particles a={a_value:.4f}\n".encode())
        f.write(b"BINARY\n")
        f.write(b"DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n} float\n".encode())
        f.write(pos_3d.astype(">f4").tobytes())  # Big-endian for legacy VTK
        f.write(b"\n")

        # Vertices
        f.write(f"VERTICES {n} {2 * n}\n".encode())
        verts = np.empty((n, 2), dtype=">i4")
        verts[:, 0] = 1
        verts[:, 1] = np.arange(n, dtype=">i4")
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
        spacing = f"{res} {res} 0"
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
        for val in rho.flatten():
            f.write(f"{val}\n")

    return filepath
