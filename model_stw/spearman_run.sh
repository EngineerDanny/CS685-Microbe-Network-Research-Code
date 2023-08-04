#!/bin/bash
#SBATCH --job-name=spearman           
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/da2343/spearman.out	
#SBATCH --error=/scratch/da2343/spearman.err

python /projects/genomic-ml/da2343/ml_project_1/model_stw/spearman_stw.py
