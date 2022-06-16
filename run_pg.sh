#! /bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -o /scratch/fengdic/slurm-%j.out

module load python/3.6
source $HOME/Documents/ENV/bin/activate

mkdir $SLURM_TMPDIR/logs/
python PG/Components/hyperparam_tuning.py --log_dir=$SLURM_TMPDIR/logs/

cp -r $SLURM_TMPDIR/gifs/ $SCRATCH/avg_discount