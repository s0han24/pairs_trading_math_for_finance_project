#!/bin/bash
#SBATCH --job-name=pairs_trading_math_for_finance_project
#SBATCH --output=results/pairs_trading_math_for_finance_project_first_run.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --nodelist=gnode091
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ronit.jalihal@students.iiit.ac.in

# Print job info
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Change to project directory
cd ~/pairs_trading_math_for_finance_project

# Activate virtual environment
source .venv/bin/activate

# Print Python and package versions
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Run the script
python3 -u tuner_xgb.py --n-trials 100

# Print completion time
echo "Job finished at $(date)"