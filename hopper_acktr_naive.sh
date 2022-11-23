#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-9

module load mujoco/2.2.0
source $HOME/Documents/ENV/bin/activate
module load python/3.7

export PYTHONPATH="$PYTHONPATH:/home/fengdic/Documents/Averaged_Dist_Corr/acktr/Hyperparam"

SECONDS=0
python naive_tune.py --seed $SLURM_ARRAY_TASK_ID --env "Hopper-v3" --lr-vf 0.001 --max-timesteps 1000000 --timesteps-per-batch 2500  --mom 0.8  --naive True --log_dir=$SCRATCH/avg_discount/logs/ &
python naive_tune.py --seed $SLURM_ARRAY_TASK_ID --env "Hopper-v3" --lr-vf 0.001 --max-timesteps 1000000 --timesteps-per-batch 2500  --mom 0.9  --naive True --log_dir=$SCRATCH/avg_discount/logs/ &
python naive_tune.py --seed $SLURM_ARRAY_TASK_ID --env "Hopper-v3" --lr-vf 0.001 --max-timesteps 1000000 --timesteps-per-batch 2500  --mom 0.95  --naive True --log_dir=$SCRATCH/avg_discount/logs/ &

echo "Baseline job $seed took $SECONDS"
