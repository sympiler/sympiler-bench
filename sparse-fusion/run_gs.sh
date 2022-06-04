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

mkdir $LOGS





bash $SCRIPTPATH/run_exp.sh  "$BINPATH/k_kernel_demo" "$UFDB" 2 $THRD > $LOGS/k_kernel_gs4.csv


python3 gauss_seidel.py $LOGS/k_kernel_gs4.csv




