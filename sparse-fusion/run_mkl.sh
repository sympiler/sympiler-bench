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
UFDB=/home/kazem/UFDB/SPD/ #./mm/
THRD=24
export OMP_NUM_THREADS=24

mkdir $LOGS


# MKL
   bash $SCRIPTPATH/run_exp.sh  $BINPATH/mkl_mv_trsv  $UFDB 4 $THRD > $LOGS/spmv_sptrsv_mkl.csv
   bash $SCRIPTPATH/run_exp.sh  $BINPATH/mkl_trsv_mv  $UFDB 4 $THRD > $LOGS/sptrsv_spmv_mkl.csv
   bash $SCRIPTPATH/run_exp.sh  $BINPATH/mkl_trsv_trsv  $UFDB 4 $THRD > $LOGS/sptrsv_sptrsv_mkl.csv
   bash $SCRIPTPATH/run_exp.sh  $BINPATH/mkl_demo $UFDB 4 $THRD > $LOGS/spilu0_sptrsv_mkl.csv
   bash $SCRIPTPATH/run_exp.sh  $BINPATH/mkl_dad_spilu0_demo $UFDB 4 $THRD > $LOGS/scal_spilu0_mkl.csv
   bash $SCRIPTPATH/run_exp.sh  $BINPATH/mkl_trsv $UFDB 4 $THRD > $LOGS/sptrsv_mkl.csv







