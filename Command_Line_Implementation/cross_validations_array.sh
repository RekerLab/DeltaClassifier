#!/bin/bash

eval "$(conda shell.bash hook)"
source activate chemprop
inDir=$2
cd $inDir
file_list=($(ls -1 *-Curated.csv))
python $1 -i ${file_list[$SLURM_ARRAY_TASK_ID]%-*} 