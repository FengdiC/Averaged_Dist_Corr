#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=4  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-3:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-200

module load mujoco/2.2.0
source $HOME/Documents/ENV/bin/activate
module load python/3.7

SECONDS=0
python spinningup/Hyperparam/naive_ppo_tune.py --seed $SLURM_ARRAY_TASK_ID --env "Hopper-v3" --log_dir=$SCRATCH/avg_discount/logs/ &

echo "Baseline job $seed took $SECONDS"
sleep 3h