#!/bin/bash

NUM_RUNS=$1
CMD=$2
FILE=$3

for i in $(seq 1 $NUM_RUNS)
do
  sbatch ${FILE}.sh "${CMD}"
done