#!/bin/bash

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="sym"
#SBATCH --nodes=1
#SBATCH --output="symbench.%j.%N.out"
#SBATCH -t 24:00:00

#module load NiaEnv/2019b
#module load cmake/3.17.3
#module load intel
#module load intel/2019u4
#module load intel/2020u4

#export SUITEROOT=/home/m/mmehride/kazem/programs/SuiteSparse/
#export SUITEROOT=/Users/kazem/programs/SuiteSparse/
#export SUITEROOT=/home/kazem/programs/SuiteSparse/
#export METISROOT=/home/m/mmehride/kazem/programs/SuiteSparse/lib/
export METISROOT=/scinet/niagara/software/2019b/opt/intel-2019u4/metis/5.1.0/lib/
THREADS=8
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

Kernel=0
Kernel=$1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPORT_DIR=$SCRIPT_DIR/../reports
BIN_DIR=$SCRIPT_DIR/../build/
DATA_DIR=$SCRIPT_DIR/../DB/



#### Building the benchmark
echo "Building ..."
mkdir $BIN_DIR
cd $BIN_DIR
rm -rf  CMakeCache*
#cmake  -DCMAKE_PREFIX_PATH="$MKLROOT/lib/intel64;/home/m/mmehride/kazem/programs/SuiteSparse/;$MKLROOT/include;/home/m/mmehride/kazem/programs/SuiteSparse/include/;/home/m/mmehride/kazem/programs/SuiteSparse/lib/" -DCMAKE_CXX_COMPILER=/scinet/niagara/software/2019b/opt/base/gcc/8.3.0/bin/g++  -DCMAKE_C_COMPILER=/scinet/niagara/software/2019b/opt/base/gcc/8.3.0/bin/gcc  -DCMAKE_BUILD_TYPE=Release ../
#cmake   -DCMAKE_CXX_COMPILER=/scinet/niagara/software/2019b/opt/base/gcc/8.3.0/bin/g++  -DCMAKE_C_COMPILER=/scinet/niagara/software/2019b/opt/base/gcc/8.3.0/bin/gcc  -DCMAKE_BUILD_TYPE=Release ../
cmake  -DSYMPILER_BLAS_BACKEND=Apple -DSYMPILER_USE_TBB=ON -DCMAKE_BUILD_TYPE=Release ../
#cmake   -DCMAKE_BUILD_TYPE=Release ../
make -j 8 sym_cholesky suite_cholmod


#### Running the benchmark
echo "Running ..."
cd $SCRIPT_DIR
mkdir $REPORT_DIR

## CHOLESKY
if [ $Kernel -eq 1 ]; then
 bash run_tool.sh $BIN_DIR/sym_interface/sym_cholesky $DATA_DIR  $THREADS 1 > ${REPORT_DIR}/sympiler_serial.csv
 bash run_tool.sh $BIN_DIR/sym_interface/sym_cholesky $DATA_DIR  $THREADS 4 > ${REPORT_DIR}/sympiler_parallel.csv
 bash run_tool.sh $BIN_DIR/sym_interface/sym_cholesky $DATA_DIR  $THREADS 1 1 > ${REPORT_DIR}/sympiler_serial_metis.csv
 bash run_tool.sh $BIN_DIR/sym_interface/sym_cholesky $DATA_DIR  $THREADS 4 1 > ${REPORT_DIR}/sympiler_parallel_metis.csv
 bash run_tool.sh $BIN_DIR/tools_interface/suite_cholmod $DATA_DIR  $THREADS 1 > ${REPORT_DIR}/cholmod_parallel.csv
 bash run_tool.sh $BIN_DIR/tools_interface/mkl_pardiso_cholesky $DATA_DIR  $THREADS 1> ${REPORT_DIR}/mkl_pardiso_parallel.csv
fi

if [ $Kernel -eq 2 ]; then
 for i in {3,2,1,0,-1,-2,-3}; do
   for j in {5,10,100,1000,4000}; do
     bash run_tool.sh $BIN_DIR/sym_interface/sym_cholesky $DATA_DIR  $THREADS 4 1 $i $j >> ${REPORT_DIR}/sympiler_tuned.csv
   done
 done
fi
