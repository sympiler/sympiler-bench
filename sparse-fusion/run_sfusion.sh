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
UFDB=$HOME/test/ # ./mm/
THRD=24
export OMP_NUM_THREADS=24

mkdir $LOGS


bash $SCRIPTPATH/run_exp.sh  "$BINPATH/sptrsv_sptrsv_demo" "$UFDB" 4  $THRD  > $LOGS/sptrsv_sptrsv.csv
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/sptrsv_spmv_demo" "$UFDB" 4  $THRD > $LOGS/sptrsv_spmv.csv
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/spmv_sptrsv_demo" "$UFDB" 4  $THRD > $LOGS/spmv_sptrsv.csv
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/spilu_sptrsv_demo" "$UFDB" 4  $THRD > $LOGS/spilu0_sptrsv.csv
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/scal_spilu_demo" "$UFDB" 4  $THRD > $LOGS/scal_spilu0.csv
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/scal_spic0_demo" "$UFDB" 4  $THRD > $LOGS/scal_spic0.csv
bash $SCRIPTPATH/run_exp.sh  "$BINPATH/spic0_sptrsv_demo" "$UFDB" 4  $THRD > $LOGS/spic0_sptrsv.csv



