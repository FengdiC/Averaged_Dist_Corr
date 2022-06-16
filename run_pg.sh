#! /bin/bash
#SBATCH --account=rrg-ashique
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH -o /scratch/fengdic/slurm-%j.out

module load python/3.6
source $HOME/projects/rpp-bengioy/fengdic/tf/bin/activate

cp Q-model/human_SeaquestDeterministic-v4_1 $SLURM_TMPDIR

mkdir $SLURM_TMPDIR/models/ $SLURM_TMPDIR/logs/ $SLURM_TMPDIR/gifs/
python Q-model/dqfd.py --max_frames=10000000 --agent='dqn' --expert_dir=$SLURM_TMPDIR/ --log_dir=$SLURM_TMPDIR/logs --checkpoint_dir=$SLURM_TMPDIR/models --gif_dir=$SLURM_TMPDIR/gifs

cp -r $SLURM_TMPDIR/models/ $SCRATCH/expert
cp -r $SLURM_TMPDIR/logs/ $SCRATCH/expert
cp -r $SLURM_TMPDIR/gifs/ $SCRATCH/expert