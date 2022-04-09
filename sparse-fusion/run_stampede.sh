#!/bin/bash
#SBATCH --job-name="fusion"
#SBATCH --output="fusion_gs.%j.%N.out"
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --export=ALL
#SBATCH -t 11:00:05
#SBATCH --mail-user=kazem.cheshmi@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes



## Paths for comet
#BINPATH=/home/kazem/development/fusion/build/demo/
#UFDB=/home/kazem/UFDB/
#LOGS=/home/kazem/development/fusion/build/logs/
#SCRIPTPATH=/home/kazem/development/fusion/scripts/
#THRD=12


## Paths for stampede2
module load cmake
module load gcc
module load metis
BINPATH=/work2/04218/tg835174/stampede2/developement/fusion/build/demo/
UFDB=/work2/04218/tg835174/stampede2/UFDB/SPD/
LOGS=/work2/04218/tg835174/stampede2/developement/fusion/build/logs/
SCRIPTPATH=/work2/04218/tg835174/stampede2/developement/fusion/scripts/
THRD=24
export OMP_NUM_THREADS=24

## Paths for my PC
#BINPATH=/home/kazem/development/fusion/build/demo/
#UFDB=/home/kazem/UFDB/LFAT5/
#LOGS=/home/kazem/development/fusion/build/logs/
#SCRIPTPATH=/home/kazem/development/fusion/scripts/
#THRD=6

#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 2 > $LOGS/spmv_spmv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_diff_demo $UFDB 2 > $LOGS/spmv_spmv_diff.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_ind_demo $UFDB 2 > $LOGS/spmv_spmv_ind.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/sptrsv_sptrsv_demo $UFDB 1 > $LOGS/sptrsv_sptrsv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/sptrsv_spmv_demo $UFDB 4 > $LOGS/sptrsv_spmv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_sptrsv_demo $UFDB 4 > $LOGS/spmv_sptrsv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spilu_sptrsv_demo $UFDB 4 > $LOGS/spilu0_sptrsv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/scal_spilu_demo $UFDB 1 > $LOGS/scal_spilu0.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/scal_spic0_demo $UFDB 1 > $LOGS/scal_spic0.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spic0_sptrsv_demo $UFDB 1 > $LOGS/spic0_sptrsv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spic0_sptrsv_demo $UFDB 4 > $LOGS/spic0_sptrsv.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spldl_sptrsv_demo $UFDB 1 > $LOGS/spldl_sptrsv.csv

#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_dnn_demo $UFDB 5 > $LOGS/spmv_spmv_dnn.csv

### Profiling
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 4 $THRD > $LOGS/spmv_spmv_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_diff_demo $UFDB 4 $THRD > $LOGS/spmv_spmv_diff_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_ind_demo $UFDB 4  $THRD > $LOGS/spmv_spmv_ind_prof.csv

#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/sptrsv_sptrsv_demo $UFDB 4 $THRD > $LOGS/sptrsv_sptrsv_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/sptrsv_spmv_demo $UFDB 4 $THRD > $LOGS/sptrsv_spmv_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_sptrsv_demo $UFDB 4 $THRD > $LOGS/spmv_sptrsv_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spic0_sptrsv_demo $UFDB 4 $THRD > $LOGS/spic0_sptrsv_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spilu_sptrsv_demo $UFDB 4 $THRD > $LOGS/spilu0_sptrsv_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/scal_spilu_demo $UFDB 4 $THRD > $LOGS/scal_spilu0_prof.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/scal_spic0_demo $UFDB 4 $THRD > $LOGS/scal_spic0_prof.csv

   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/k_kernel_demo $UFDB 4 $THRD > $LOGS/k_kernel_gs4.csv

#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spldl_sptrsv_demo $UFDB 4 $THRD > $LOGS/spldl_sptrsv_prof.csv



###libraries
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_mv_mv $UFDB 2 $THRD > $LOGS/spmv_spmv_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_mv_mv_diff $UFDB 2 $THRD > $LOGS/spmv_spmv_diff_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_mv_mv_ind $UFDB 2 $THRD > $LOGS/spmv_spmv_ind_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_mv_trsv  $UFDB 2 $THRD > $LOGS/spmv_sptrsv_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_trsv_mv  $UFDB 2 $THRD > $LOGS/sptrsv_spmv_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_trsv_trsv  $UFDB 2 $THRD > $LOGS/sptrsv_sptrsv_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_demo $UFDB 2 $THRD > $LOGS/spilu0_sptrsv_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_dad_spilu0_demo $UFDB 2 $THRD > $LOGS/scal_spilu0_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/mkl_trsv $UFDB 2 $THRD > $LOGS/sptrsv_mkl.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/dagp_demo $UFDB 2 $THRD > $LOGS/dagp_kernels.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spldl_sptrsv_demo $UFDB 4 > $LOGS/spldl_sptrsv_serial.csv

#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 4 > $LOGS/spmv_spmv_new.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/sptrsv_spmv_demo $UFDB 4 > $LOGS/sptrsv_spmv_slacked.csv


### Joint DAG
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/etree_demo $UFDB 2 > $LOGS/spmv_sptrsv_joint_dag.csv
#bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/sptrsv_demo $UFDB 2 > $LOGS/sptrsv_all.csv



#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 6 2 > $LOGS/spmv_spmv_cm2.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 6 4 > $LOGS/spmv_spmv_cm4.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 6 8 > $LOGS/spmv_spmv_cm8.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_demo $UFDB 6 100 > $LOGS/spmv_spmv_cm100.csv
#
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_diff_demo $UFDB 6 2 > $LOGS/spmv_spmv_diff_cm2.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_diff_demo $UFDB 6 4 > $LOGS/spmv_spmv_diff_cm4.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_diff_demo $UFDB 6 8 > $LOGS/spmv_spmv_diff_cm8.csv
#   bash $SCRIPTPATH/run_exp_comet.sh $BINPATH/spmv_spmv_diff_demo $UFDB 6 100 > $LOGS/spmv_spmv_diff_cm100.csv
