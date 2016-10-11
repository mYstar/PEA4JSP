#!/bin/bash

#Variables
RUNS=10
OUTFILE="/scratch/s9547231/output.txt"
ERRFILE="/scratch/s9547231/error.txt"

#Submit this script with: ./all.sh <tasks>

sbatch \
 --time=0:2:00 \
 --ntasks=2 \
 --partition=haswell
 --mem-per-cpu=2048M \
 -J "multistart_test" \
 --mail-user=s9547231@tu-dresden.de \
 --mail-type=END \
 --mail-type=FAIL \
 -A p_jobshop \
 multistartnsga2.batch

exit 0
