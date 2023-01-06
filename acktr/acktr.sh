#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-12:00
#SBATCH --output=%N-%j.out
#SBATCH --account=rrg-ashique
#SBATCH --array=1-288 

source /home/vasan/src/rtrl/bin/activate
export PYTHONPATH="$PYTHONPATH:/home/vasan/src/Averaged_Dist_Corr"

SECONDS=0
python hyp_acktr.py --hyp_seed $SLURM_ARRAY_TASK_ID --agent "weighted_acktr" --env "Acrobot-v1" --continuous &
python hyp_acktr.py --hyp_seed $SLURM_ARRAY_TASK_ID --agent "weighted_acktr" --env "Acrobot-v1" --continuous --naive  &
python hyp_acktr.py --hyp_seed $SLURM_ARRAY_TASK_ID --agent "acktr" --env "Acrobot-v1" --continuous &
sleep 12h
echo "Baseline job $seed took $SECONDS"