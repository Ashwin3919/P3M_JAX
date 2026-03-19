# P3M-JAX on Google Colab

This guide shows how to clone the repository, install dependencies, run simulations, and inspect outputs from a Colab notebook.

## Runtime Recommendation

- CPU is sufficient for most runs and development workflows.
- You do not need a GPU unless you are going beyond 256 particles per side or running heavier 3D production cases.
- If you want GPU acceleration in Colab, select a T4 runtime.

## 1. Clone Repository

Run this in a notebook code cell:

```bash
!git clone https://github.com/Ashwin3919/P3M_JAX.git
%cd P3M_JAX
```

## 2. Install Dependencies

Run this in a notebook code cell:

```bash
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

## 3. Run a Simulation

Choose one config and run it.

```bash
!python main.py --config configs/default.json
```

Other examples:

```bash
!python main.py --config configs/high_res.json
!python main.py --config configs/3d_default.json
!python main.py --config configs/3d_visual.json
```

Outputs are written under the results directory for each config, including:
- VTK particle files with momentum vectors
- VTK density grids
- Power spectrum CSV and plots

## 4. Package VTK Output for Download

If you want to visualize results locally in ParaView or VisIt, zip the output and download it.

```bash
!zip -r results_bundle.zip results
```

Then download from the Colab file browser.

## 5. Visualize in ParaView or VisIt

After downloading and extracting the results:
- Open particle VTK files from results/<config_name>/vtk/particles
- Open density VTK files from results/<config_name>/vtk/density
- For particles, use vector visualization tools (for example glyphs) on the momentum field
- For density, use volume/slice rendering to inspect structure formation

## Notes

This is experimental simulation software under active development. Numerical settings, defaults, and interfaces may evolve as the codebase matures.
