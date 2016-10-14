#! /bin/bash

for int in 5 10
do
  for idx in {0..1}
  do
    sbatch dEAnsga2_topo.batch "$idx" "$int" 10
  done
done
