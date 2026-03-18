import os
import numpy as np
from pathlib import Path

def _to_xyz(data):
    """Convert 2D (N, 2) data to 3D (N, 3) for VTK by adding z=0"""
    n = data.shape[0]
    out = np.zeros((n, 3), dtype=data.dtype)
    out[:, :2] = data
    return out

def write_vtk_particles(pos, mom, a_value, results_dir, config_name):
    """Write particle snapshot as a Legacy Binary VTK PolyData file."""
    vtk_particles_dir = os.path.join(results_dir, 'vtk', 'particles')
    os.makedirs(vtk_particles_dir, exist_ok=True)
    
    filename = f"{config_name}_particles_a{a_value:.3f}.vtk"
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
    """Write density field to VTK (Structured Points) in vtk/density/"""
    vtk_density_dir = os.path.join(results_dir, 'vtk', 'density')
    os.makedirs(vtk_density_dir, exist_ok=True)
    
    filename = f"{config_name}_density_a{a_value:.3f}.vtk"
    filepath = os.path.join(vtk_density_dir, filename)
    
    nx, ny = rho.shape
    res = box.res
    
    # Keeping density as ASCII for easier debugging, but structuring it for Paraview
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"Density field at a={a_value:.3f}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write(f"ORIGIN 0 0 0\n")
        f.write(f"SPACING {res} {res} 0\n")
        f.write(f"POINT_DATA {nx * ny}\n")
        f.write("SCALARS density float 1\n")
        f.write("LOOKUP_TABLE default\n")
        
        flat_rho = np.array(rho).flatten()
        for val in flat_rho:
            f.write(f"{val}\n")
            
    return filepath
