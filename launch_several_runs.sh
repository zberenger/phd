#!/bin/bash

#SBATCH --mem=100G 
#SBATCH --array=0

srun -p P100 bash run.sh ${SLURM_ARRAY_TASK_ID}
