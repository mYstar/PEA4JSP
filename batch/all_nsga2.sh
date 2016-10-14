#! /bin/bash

#GEN='/home/est/doc/jspeval/generated'
GEN='/home/s9547231/data/generated'
#TAI='/home/est/doc/jspeval/converted/Taillard_extended'
TAI='/home/s9547231/data/Taillard_extended'


for file in `ls ${GEN}`
do
  for idx in {0..9}
  do
    sbatch nsga2.batch "$idx" "${GEN}/${file}"
  done
done

for file in `ls ${TAI}`
do
  for idx in {0..9}
  do
    sbatch nsga2.batch "$idx" "${TAI}/${file}"
  done
done
