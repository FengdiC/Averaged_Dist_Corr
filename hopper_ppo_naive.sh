#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j-naive.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-200

source $HOME/Documents/ENV/bin/activate
module load python/3.10
module load mujoco mpi4py

SECONDS=0
python spinningup/Hyperparam/naive_ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --env "Hopper-v4" --log_dir=$SCRATCH/avg_discount/logs/ &

echo "Baseline job $seed took $SECONDS"
sleep 12h