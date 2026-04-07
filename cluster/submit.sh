#!/bin/bash
#SBATCH --job-name=gpt_fit
#SBATCH --array=0-799             # 5 groups × 20 ranks × 8 folds = 800 tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

module load python/3.11 scipy-stack
mkdir -p logs results

# Task ID layout:
#   task_id = group_idx * (N_RANKS * N_FOLDS) + k_idx * N_FOLDS + fold
#   N_RANKS=20, N_FOLDS=8 → stride per group = 160, stride per rank = 8

GROUP_IDS=(0 1 2 3 4)
RANK_CONFIGS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

GROUP_IDX=$(( SLURM_ARRAY_TASK_ID / 160 ))
K_IDX=$(( (SLURM_ARRAY_TASK_ID % 160) / 8 ))
FOLD=$(( SLURM_ARRAY_TASK_ID % 8 ))

GROUP_ID=${GROUP_IDS[$GROUP_IDX]}
K=${RANK_CONFIGS[$K_IDX]}

echo "Job $SLURM_ARRAY_TASK_ID → group=$GROUP_ID  k=$K  fold=$FOLD"
python run_gpt.py --group_id $GROUP_ID --k $K --fold $FOLD --outdir results
