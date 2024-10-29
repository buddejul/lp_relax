#!/bin/bash
#SBATCH --account=ag_ifs_freyberger
#SBATCH --partition=intelsr_medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --mem-per-cpu=200M

# --------------------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------------------
# For 10 sims, 1000 boot, 10_000 obs, local takes 6 minutes
# Hence, 600 minutes, or 10 hours, for 1000 sims remote.

# Run with 21 Cores, num_sims = 1000, num_boot = 1000, num_points = 20,
# num_obs = [1_000, 10_000]
# job id: 17692621
# pytask: 40 tasks
# -------
# Nodes: 1
# Cores per node: 21
# CPU Utilized: 3-14:02:14
# CPU Efficiency: 83.88% of 4-06:34:03 core-walltime
# Job Wall-clock time: 04:53:03 (= pytask time)
# Memory Utilized: 3.66 GB
# Memory Efficiency: 89.16% of 4.10 GB

# --------------------------------------------------------------------------------------
# Start script
# --------------------------------------------------------------------------------------
source ~/.bashrc
conda deactivate
conda activate lp_relax

pytask --parallel-backend loky -n 21 -m hpc
