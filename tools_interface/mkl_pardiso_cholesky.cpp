//
// Created by kazem on 6/7/21.
//

#define CSV_LOG 1
#include <iostream>
#include <aggregation/sparse_io.h>
#include <aggregation/test_utils.h>
//#include <omp.h>
#include <aggregation/metis_interface.h>
#include <sympiler/parsy/cholesky_solver.h>
#include <mkl_types.h>
#include <mkl_service.h>
#include <mkl.h>
#include <sympiler/parsy/matrixVector/spmv_CSC.h>
#include <sympiler/parsy/common/Norm.h>
#include "../common/FusionDemo.h"
#include "../common/label.h"
using namespace sym_lib;

/// Evaluate cholesky
/// \return
int mkl_cholesky_demo(int argc, char *argv[]);




int main(int argc, char *argv[]){
 int ret_val;
 ret_val = mkl_cholesky_demo(argc,argv);
 return ret_val;
}


int mkl_cholesky_demo(int argc, char *argv[]){
 CSC *L1_csc, *A = NULLPNTR;
 CSR *L2_csr;
 int n;
 int num_threads = 6;
 int p2 = -1, p3 = 4000; // LBC params
 int header = 0;
 int mode = 1;
 int *perm;
 std::string matrix_name;
 std::vector<timing_measurement> time_array;
 if (argc < 2) {
  PRINT_LOG("Not enough input args, switching to random mode.\n");
  n = 16;
  double density = 0.2;
  matrix_name = "Random_" + std::to_string(n);
  A = random_square_sparse(n, density);
  if (A == NULLPNTR)
   return -1;
  L1_csc = make_half(A->n, A->p, A->i, A->x);
  delete A;
 } else {
  std::string f1 = argv[1];
  matrix_name = f1;
  A = read_mtx(f1);
  if(A->stype == 0){
   L1_csc = make_half(A->n, A->p, A->i, A->x);
   delete A;
  } else {
   L1_csc = A;
  }
  if (L1_csc == NULLPNTR)
   return -1;
  n = L1_csc->n;
 }
 if(argc > 2)
  num_threads = atoi(argv[2]);
 if(argc > 3)
  header = atoi(argv[3]);
 if(argc > 4)
  mode = atoi(argv[4]);

 //omp_set_num_threads(num_threads);
 mkl_set_num_threads(num_threads);
/// Method 1 of calling Cholesky
 auto *solution = new double[n];
 std::fill_n(solution, n, 1.0);

/// Solving the linear system

 double *a;
 int *ja;
 int *ia;
 a = L1_csc->x;
 ia = L1_csc->p;
 ja = L1_csc->i;
 n = L1_csc->n;

 MKL_INT mtype = 2;       /* Real unsymmetric matrix */
 /* RHS and solution vectors. */
 //double b[8], x[8];
 double *b = solution;
 double *x = (double *) mkl_calloc(n, sizeof(double), 64);
 MKL_INT nrhs = 1;     /* Number of right hand sides. */
 long int pt[64];
 /* Pardiso control parameters. */
 MKL_INT iparm[64];
 MKL_INT maxfct, mnum, phase, error, msglvl;
 /* Auxiliary variables. */
 MKL_INT i;
 double ddum;          /* Double dummy */
 MKL_INT idum;         /* Integer dummy. */


/* -------------------------------------------------------------------- */
/* .. Setup Pardiso control parameters. */
/* -------------------------------------------------------------------- */
 for (i = 0; i < 64; i++) {
  iparm[i] = 0;
 }
 iparm[0] = 1;         /* No solver default */
 iparm[1] = 2;         /* Fill-in reordering from METIS */
 iparm[3] = 0;         /* No iterative-direct algorithm */
 iparm[4] = 0;         /* No user fill-in reducing permutation */
// iparm[4] = 1;         /* user fill-in reducing permutation is used*/
 iparm[5] = 0;         /* Write solution into x */
 iparm[6] = 0;         /* Not in use */
 iparm[7] = 4;         /* Max numbers of iterative refinement steps */
 iparm[8] = 0;         /* Not in use */
 iparm[9] = 9;        /* Perturb the pivot elements with 1E-8 */
 iparm[10] = 0;        /* Use nonsymmetric permutation and scaling MPS */
 iparm[11] = 0;        /* A^TX=B */
 iparm[12] = 0;        /* Maximum weighted matching algorithm is switched-off (default for symmetric). Try iparm[12] = 1 in case of inappropriate accuracy */
 iparm[13] = 0;        /* Output: Number of perturbed pivots */
 iparm[14] = 0;        /* Not in use */
 iparm[15] = 0;        /* Not in use */
 iparm[16] = 0;        /* Not in use */
 iparm[17] = -1;       /* Output: Number of nonzeros in the factor LU */
 iparm[18] = -1;       /* Output: Mflops for LU factorization */
 iparm[19] = 0;        /* Output: Numbers of CG Iterations */
 iparm[20] = 1; /*using bunch kaufman pivoting*/
 iparm[55] = 0; /*Diagonal and pivoting control., default is zero*/

 //
 iparm[26] = 1;
 //iparm[23] = 1; //TODO: Later enable to se if the parallelism is better
 iparm[34] = 1;
 //Because iparm[4]==0 so:
 iparm[30] = 0;
 iparm[35] = 0;
 maxfct = 1;           /* Maximum number of numerical factorizations. */
 mnum = 1;         /* Which factorization to use. */
 msglvl = 0;           /* Print statistical information in file */
 error = 0;            /* Initialize error flag */
/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* -------------------------------------------------------------------- */
 for (i = 0; i < 64; i++) {
  pt[i] = 0;
 }



 timing_measurement symbolic_time, factor_time, solve_time;
 symbolic_time.start_timer();
/* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates */
/* all memory that is necessary for the factorization. */
/* -------------------------------------------------------------------- */

 phase = 11;
 PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, NULL, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
 if (error != 0) {
  printf("\nERROR during symbolic factorization: %d", error);
  exit(1);
 }
 symbolic_time.measure_elapsed_time();

 factor_time.start_timer();
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */
 phase = 22;
 PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, NULL, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
 if (error != 0) {
  printf("\nERROR during numerical factorization: %d", error);
  exit(2);
 }
 factor_time.measure_elapsed_time();

 solve_time.start_timer();
/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
 phase = 33;
 iparm[7] = 0;         /* Max numbers of iterative refinement steps. */
 PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, NULL, &nrhs, iparm, &msglvl, b, x, &error);
 if (error != 0) {
  printf("\nERROR during solution: %d", error);
  exit(3);
 }
 solve_time.measure_elapsed_time();

/*
 char uplo = 'L';
 for (int k = 0; k < n; ++k) {
  std::cout<<x[k]<<",";
 }
 std::cout<<"\n";
 double *ax = new double[n]();
 mkl_cspblas_dcsrsymv(&uplo, &n, a, ia, ja, x, ax);
 for (int j = 0; j < n; ++j) {
   std::cout<<ax[j]<<";";
   ax[j]-=b[j];
 }

*/

 double alp=1, bet=-1;
 sym_lib::parsy::spmv_csc_sym_one_int(L1_csc->n, L1_csc->p, L1_csc->i, L1_csc->x, -1,
   &alp, &bet, 1, x, b);
 //print_vec("res: ",0,TMP2->ncol,res);

 double res_l1 = sym_lib::parsy::norm_dense(n, 1, b, 0);


 int nnz_l = iparm[17];
 double fl = iparm[18]*1e6;
 double total_flops = fl + (2*nnz_l) + n;

 if(header){
  print_common_header();
  std::cout<<TOOL<<",LBC P1,LBC P2,";
  std::cout<<SYM_TIME<<","<<FCT_TIME","<<SOLVE_TIME<<","<<RESIDUAL<<",";
  std::cout<<FLOPS<<",";
  std::cout<<"\n";
 }

 print_common(matrix_name, "Cholesky", "", L1_csc, (int)iparm[36], num_threads);
 PRINT_CSV("MKL Pardiso");
 PRINT_CSV(p2);
 PRINT_CSV(p3);
 PRINT_CSV(symbolic_time.elapsed_time);
 PRINT_CSV(factor_time.elapsed_time);
 PRINT_CSV(solve_time.elapsed_time);
 PRINT_CSV(res_l1);
 PRINT_CSV(total_flops);


/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
 phase = -1;           /* Release internal memory. */
 PARDISO(pt, &maxfct, &mnum, &mtype, &phase,
         &n, &ddum, ia, ja, NULL, &nrhs,
         iparm, &msglvl, &ddum, &ddum, &error);


 delete []solution;
 delete L1_csc;

 mkl_free(x);


 return 0;
}



