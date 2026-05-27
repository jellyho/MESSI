# IsaacSim Installation on Cluster Servers (No Docker)

Isaac Sim can be installed entirely via pip into a conda environment — no Docker or root access required. This is the recommended path for HPC/cluster environments.

## Prerequisites

- Ubuntu 22.04+ (or a cluster node running it)
- NVIDIA GPU with driver ≥ 535
- CUDA 12.8 compatible driver
- `cmake` available (either system-installed or via `module load`)
- Internet access from compute nodes (or pre-downloaded packages)

If cmake is not in your PATH, load it before running the setup:

```bash
module load cmake        # typical on SLURM/PBS clusters
# or install it into the conda env later (see Troubleshooting)
```

## Installation

```bash
# Optional: set a custom workspace dir (useful when $HOME has a small quota)
export HS_WORKSPACE_DIR=/path/to/scratch/holosoma_deps

# Optional: choose a conda environment name (default: hssim)
export CONDA_ENV_NAME=hssim

bash scripts/setup_isaacsim_cluster.sh
```

The script will:
1. Download and install Miniconda under `$HS_WORKSPACE_DIR/miniconda3` (skipped if conda already exists there)
2. Create a Python 3.11 conda environment
3. Install `cmake` and other system deps via conda (no `apt`/`sudo` needed)
4. Install PyTorch 2.7 (CUDA 12.8)
5. Install `isaacsim[all,extscache]==5.1.0` from NVIDIA's PyPI index
6. Clone and install IsaacLab v2.3.0
7. Install the `holosoma` package

Installation takes 20–40 minutes depending on download speed.

## Activating the Environment

```bash
source scripts/source_isaacsim_setup.sh
# or manually:
source $HS_WORKSPACE_DIR/miniconda3/bin/activate hssim
export OMNI_KIT_ACCEPT_EULA=1
```

## Running on a Headless Cluster

Isaac Sim requires a display or a virtual framebuffer for rendering. On headless nodes, use:

```bash
# Option 1: EGL headless rendering (recommended, no display needed)
export DISPLAY=""
# IsaacSim will fall back to EGL automatically when no display is present

# Option 2: Virtual framebuffer via Xvfb (if EGL is unavailable)
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
```

For SLURM, add to your job script:

```bash
#SBATCH --gres=gpu:1
export OMNI_KIT_ACCEPT_EULA=1
source $HS_WORKSPACE_DIR/miniconda3/bin/activate hssim
python src/holosoma/holosoma/train_agent.py ...
```

## Troubleshooting

**`cmake` not found during IsaacLab install**

```bash
conda install -c conda-forge cmake
```

**`sudo apt install` fails (no root access)**

The `setup_issacsim_after.sh` variant calls `sudo apt install cmake build-essential`. Use `setup_isaacsim.sh` instead (this line is commented out). If `build-essential` tools are missing, install via conda:

```bash
conda install -c conda-forge gcc gxx cmake
```

**Disk quota exceeded in `$HOME`**

Set `HS_WORKSPACE_DIR` to a scratch filesystem before running setup. IsaacSim and IsaacLab together require ~25–30 GB.

```bash
export HS_WORKSPACE_DIR=/scratch/$USER/holosoma_deps
```

**`egl_probe` cmake version error**

Already handled by the setup script via `CMAKE_POLICY_VERSION_MINIMUM=3.5`. If you run the steps manually, make sure to export that variable before `./isaaclab.sh --install`.
