#! /bin/bash

#GEN='/home/est/doc/jspeval/generated'
GEN='/home/s9547231/data/generated'
#TAI='/home/est/doc/jspeval/converted/Taillard_extended'
TAI='/home/s9547231/data/Taillard_extended'

# choose algorithm
ALG='none'
if [ $1 == 'ms' ]
then
  ALG='/home/s9547231/PEA4JSP/msnsga2.py'
elif [ $1 == 'multi' ]
then
  ALG='/home/s9547231/PEA4JSP/multistartnsga2.py'
elif [ $1 == 'dea' ]
then
  ALG='/home/s9547231/PEA4JSP/dEAnsga2.py'
fi
# choose number of tasks
TASKS=$2


for file in `ls ${GEN}`
do
  for idx in {0..9}
  do
    sbatch "parallel${TASKS}.batch" "$idx" "$ALG" "${GEN}/${file}"
  done
done

for file in `ls ${TAI}`
do
  for idx in {0..9}
  do
    sbatch "parallel${TASKS}.batch" "$idx" "$ALG" "${TAI}/${file}"
  done
done
