## Postprocessing and Mesh Visualization Sandbox

This repository contains modules designed to work with SeisSol output, allowing for mesh handling, field enrichment, and 3D visualizations using PyVista. The tools are designed for earthquake rupture dynamics analysis, focusing on postprocessing SeisSol XDMF output files and generating visualizations of fault-related data.

## Features

### Modules Overview

- **`pv_tools.py`**: Contains core functionality for converting SeisSol XDMF output to PyVista meshes, enriching the mesh with fields, and handling edges for visualization.
- **`Fault_CMaps.py`**: Provides colourmap choices for SeisSol data fields.
- **`PaperFigGen.py`**: Provides functions to easily generate figures, including the visualization of fields like accumulated slip, stress drops, and rupture time.
- **`Test`**: Contains a test jupyter notebook that demonstrates the use of the modules on a sample SeisSol output.

## Dependencies

The following Python libraries are required:

- `numpy`
- `pyvista`
- `matplotlib`
- `scipy`
- `seissolxdmf`

Ensure that you have these packages installed before running the scripts.
