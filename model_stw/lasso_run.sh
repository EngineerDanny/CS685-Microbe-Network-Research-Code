#!/bin/bash
#SBATCH --job-name=lasso           
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/da2343/lasso.out	
#SBATCH --error=/scratch/da2343/lasso.err

python /projects/genomic-ml/da2343/ml_project_1/model_stw/lasso_stw.py
