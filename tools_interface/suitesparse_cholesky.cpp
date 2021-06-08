//
// Created by kazem on 6/7/21.
//

#include <sparse_io.h>
#include <test_utils.h>
#include <sparse_utilities.h>
#include <omp.h>
#include "cholmod.h"
#include "FusionDemo.h"
#include "../common/label.h"
using namespace sym_lib;

/// Evaluate cholesky
/// \return
int suite_cholesky_demo(int argc, char *argv[]);


int main(int argc, char *argv[]){
 int ret_val;
 ret_val = suite_cholesky_demo(argc,argv);
 return ret_val;
}


int suite_cholesky_demo(int argc, char *argv[]){
 CSC *L1_csc, *A = NULLPNTR;
 CSR *L2_csr;
 size_t n;
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
 } else {
  std::string f1 = argv[1];
  matrix_name = f1;
  L1_csc = read_mtx(f1);
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
 omp_set_num_threads(num_threads);
/// Method 1 of calling Cholesky


 std::string f1 = argv[1];

 cholmod_common cm;
 cholmod_sparse* Ac;
 cholmod_factor* L;
 cholmod_dense *b;
 cholmod_start(&cm);
 FILE *ff = fopen(f1.c_str(),"r");
 Ac = cholmod_read_sparse(ff,&cm);

/// Solving the linear system

  b = cholmod_allocate_dense(L1_csc->m, 1, L1_csc->m, CHOLMOD_REAL, &cm);
  double *solution = (double *)b->x;
  std::fill_n(solution, n, 1.0);
  cholmod_dense* x;

 timing_measurement symbolic_time, factor_time, solve_time;
 symbolic_time.start_timer();
 L = cholmod_analyze(Ac, &cm);
 symbolic_time.measure_elapsed_time();

 factor_time.start_timer();
 cholmod_factorize(Ac, L, &cm);
 factor_time.measure_elapsed_time();

 solve_time.start_timer();
 x = cholmod_solve(CHOLMOD_A, L, b, &cm);
 solve_time.measure_elapsed_time();

 double alpha = 1.0, beta=-1.0;
/*
 cholmod_print_dense(x, "X:",&cm);
 double *tmp = (double*) x->x;
 for (int i = 0; i < 14; ++i) {
  std::cout<<tmp[i]<<",";
 }
 std::cout<<"\n";
*/
 cholmod_sdmult(Ac,0,&alpha,&beta,x,b,&cm);
 double res = cholmod_norm_dense(b,0,&cm);

 if(header){
  print_common_header();
  std::cout<<TOOL<<",LBC P1,LBC P2,";
  std::cout<<SYM_TIME<<","<<FCT_TIME","<<SOLVE_TIME<<","<<RESIDUAL<<",";
  std::cout<<"\n";
 }

 print_common(matrix_name, "Cholesky", "", L1_csc, L->nzmax, num_threads);
 if(mode == 1)
  PRINT_CSV("Sequential Sympiler");
 else
  PRINT_CSV("CHOLMOD");
 PRINT_CSV(p2);
 PRINT_CSV(p3);
 PRINT_CSV(symbolic_time.elapsed_time);
 PRINT_CSV(factor_time.elapsed_time);
 PRINT_CSV(solve_time.elapsed_time);
 PRINT_CSV(res);

#ifdef MRHS
 PRINT_CSV(solve_time1.elapsed_time);
 PRINT_CSV(sym_chol1->res_l1);
 delete sym_chol1;
 delete []rhs;
#endif

// delete []solution;
 delete A;
 delete L1_csc;

 cholmod_free_sparse(&Ac, &cm);
 cholmod_free_factor(&L, &cm);
 cholmod_free_dense(&b, &cm);
 cholmod_free_dense(&x, &cm);
 cholmod_finish(&cm);

 return 0;
}
