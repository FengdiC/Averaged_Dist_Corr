#!/bin/bash
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-3:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ashique
#SBATCH --array=1-30 

source /home/vasan/src/rtrl/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/vasan/src/Averaged_Dist_Corr"

for env in "Walker2d-v2"
do
python train.py --agent "acktr" --env $env --seed $SLURM_ARRAY_TASK_ID --scale_weight 1 --batch_size 2048 --lr 0.3 --hidden 64 --continuous --lam 0.97 --kfac_clip 0.002
done
