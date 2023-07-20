#!/bin/bash
#SBATCH --job-name=pearson           
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=64
#SBATCH --output=/scratch/da2343/pearson.out	
#SBATCH --error=/scratch/da2343/pearson.err

python /projects/genomic-ml/da2343/ml_project_1/model_stw/pearson_stw.py
