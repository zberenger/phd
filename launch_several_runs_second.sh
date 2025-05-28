#!/bin/bash

#SBATCH --mem=100G 
#SBATCH --array=0-5

srun -p P100 bash run2.sh ${SLURM_ARRAY_TASK_ID}
