#! /bin/bash

for int in 5 10 25 50
do
  for idx in {0..9}
  do
    sbatch dEAnsga2_topo.batch "$idx" "$int" 10
  done
done
