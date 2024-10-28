#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=200M

# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------
# For 10 sims, 1000 boot, 10_000 obs, local takes 6 minutes
# Hence, 600 minutes, or 10 hours, for 1000 sims remote.

# --------------------------------------------------------------------------------------
# Start script
# --------------------------------------------------------------------------------------
source ~/.bashrc
conda deactivate
conda activate lp_relax

pytask --parallel-backend loky -n 20 -m hpc
