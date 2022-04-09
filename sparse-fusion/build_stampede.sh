#!/bin/sh

#SBATCH --job-name="fusion"
#SBATCH --output="fusion_gs.%j.%N.out"
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --export=ALL
#SBATCH -t 11:00:05
#SBATCH --mail-user=kazem.cheshmi@gmail.com
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finish

module load NiaEnv/2019b
module load cmake/3.17.3
#module load intel
module load intel/2019u4
#module load intel
#module load gcc
module load metis/5.1.0

#export MKLROOT=/scinet/intel/2019u4/compilers_and_libraries_2019.4.243/linux/mkl 
export METISROOT=/scinet/niagara/software/2019b/opt/intel-2019u4/metis/5.1.0/
#export METISROOT=/home/m/mmehride/kazem/programs/metis-5.1.0/
#export OSQPROOT=/home/m/mmehride/kazem/programs/osqp-0.6.0/ # /programs/osqp/
#export MOSEKROOT=/home/m/mmehride/kazem/programs/mosek/8/tools/platform/linux64x86/
#export EIGENROOT=/home/m/mmehride/kazem/programs/eigen-eigen-323c052e1731/
#export QLROOT=/home/m/mmehride/kazem/programs/

make clean
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 20

