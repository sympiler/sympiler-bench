//
// Created by kazem on 4/30/21.
//

#include <iostream>
#include <sparse_io.h>
#include <test_utils.h>
#include <omp.h>
#include <metis_interface.h>
#include <linear_solver_wrapper.h>
#include "FusionDemo.h"

using namespace sym_lib;

/// Evaluate sptrsv
/// \return
int sym_cholesky_demo(int argc, char *argv[]);


int main(int argc, char *argv[]){
 int ret_val;
 ret_val = sym_cholesky_demo(argc,argv);
 return ret_val;
}


int sym_cholesky_demo(int argc, char *argv[]){
 CSC *L1_csc, *A = NULLPNTR;
 CSR *L2_csr;
 size_t n;
 int num_threads = 6;
 int p2 = -1, p3 = 4000; // LBC params
 int header = 0;
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
 if(argc >= 3)
  p2 = atoi(argv[2]);
 omp_set_num_threads(num_threads);
 if(argc >= 4)
  p3 = atoi(argv[3]);
 /// Re-ordering L matrix


 auto *solution = new double[n];
 std::fill_n(solution, n, 1.0);

 auto *H = new sym_lib::parsy::CSC;

 H->nzmax = L1_csc->nnz; H->ncol= H->nrow =L1_csc->n;
 H->p = L1_csc->p; H->i = L1_csc->i; H->x = L1_csc->x; H->stype=-1;
 H->packed = 1;
/// Solving the linear system
 auto *sym_chol = new sym_lib::parsy::SolverSettings(H, solution);
 sym_chol->ldl_variant = 4;
 sym_chol->req_ref_iter = 0;
 sym_chol->solver_mode = 0;
// sym_chol->sym_order = sym_lib::parsy::S_METIS;
 //sym_chol->sym_order = sym_lib::parsy::SYM_ORDER::S_METIS;
 timing_measurement symbolic_time, factor_time, solve_time;
 symbolic_time.start_timer();
 sym_chol->symbolic_analysis();
 symbolic_time.measure_elapsed_time();
 factor_time.start_timer();
 sym_chol->numerical_factorization();
 factor_time.measure_elapsed_time();
 solve_time.start_timer();
 double *x = sym_chol->solve_only();
 solve_time.measure_elapsed_time();

 sym_chol->compute_norms();


 if(header){
  print_common_header();
  std::cout<<"Tool Name,LBC P1,LBC P2,";
  std::cout<<"Symbolic Analysis Time (sec),Factorization Time (sec),Solve Time (sec),";
  std::cout<<"Residual,"
  "\n";
 }

 print_common(matrix_name, "Cholesky", "", L1_csc, L1_csc, num_threads);
 PRINT_CSV("Sympiler");
 PRINT_CSV(p2);
 PRINT_CSV(p3);
 PRINT_CSV(symbolic_time.elapsed_time);
 PRINT_CSV(factor_time.elapsed_time);
 PRINT_CSV(solve_time.elapsed_time);
 PRINT_CSV(sym_chol->res_l1);

 delete []solution;
 delete A;
 delete L1_csc;
 delete H;
 delete sym_chol;

 return 0;
}
