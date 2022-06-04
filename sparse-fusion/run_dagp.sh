#!/bin/bash
#SBATCH --job-name="fusion"
#SBATCH --output="fusion_gs.%j.%N.out"
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --export=ALL
#SBATCH -t 11:00:05

# this is for stampede2, should change for other servers
#module load tacc-singularity/3.7.2

BINPATH="singularity exec artifact.sif /source/fusion/build/demo/"
LOGS=./logs/ 
SCRIPTPATH=./
UFDB=./mm/
THRD=24
export OMP_NUM_THREADS=24

mkdir $LOGS/dagp



# DAGP
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/dagp_demo" "$UFDB" 4 $THRD > $LOGS/dagp/dagp_kernels.csv

mkdir $LOGS/plots

 for f in $LOGS/*.csv; do
 	python3 graph_gen.py -i $f -o $LOGS/plots/
 done


 python3 graph_gen.py -d $LOGS/plots/







