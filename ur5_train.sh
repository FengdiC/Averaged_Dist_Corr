#!/bin/bash

export PYTHONPATH="$PYTHONPATH:/home/gautham/src/Averaged_Dist_Corr"

for seed in {1..5}
do
python ur5_train.py --agent "ppo" --buffer 2048 --batch_size 64 --gamma 0.99 --seed $seed
python ur5_train.py --agent "weighted_shared_ppo" --buffer 2048 --batch_size 64 --gamma 0.99 --lr_weight 0.0003 --seed $seed
python ur5_train.py --agent "ppo" --buffer 2048 --batch_size 64 --gamma 0.99 --naive --seed $seed
done
