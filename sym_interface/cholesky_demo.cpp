//
// Created by kazem on 4/30/21.
//
#define CSV_LOG 1
#include <iostream>
#include <sparse_io.h>
#include <test_utils.h>
#ifdef OPENMP
#include <omp.h>
#endif
#include <metis_interface.h>
#include <parsy/cholesky_solver.h>
#include "../common/FusionDemo.h"
#include "../common/label.h"
using namespace sym_lib;

/// Evaluate cholesky
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
 int mode = 1; int ord = 0;
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
 if(argc > 5)
  ord = atoi(argv[5]);
 if(argc > 6)
  p2 = atoi(argv[6]);
 if(argc > 7)
  p3 = atoi(argv[7]);

#ifdef OPENMP
 omp_set_num_threads(num_threads);
#endif
 /// Method 1 of calling Cholesky
 auto *solution = new double[n];
 std::fill_n(solution, n, 1.0);

 auto *H = new sym_lib::parsy::CSC;

 H->nzmax = L1_csc->nnz; H->ncol= H->nrow =L1_csc->n;
 H->p = L1_csc->p; H->i = L1_csc->i; H->x = L1_csc->x; H->stype=-1;
 H->packed = 1;
/// Solving the linear system
 auto *sym_chol = new sym_lib::parsy::SolverSettings(H, solution);
 sym_chol->ldl_variant = mode;
 sym_chol->req_ref_iter = 0;
 sym_chol->solver_mode = 0;
// sym_chol->sym_order = sym_lib::parsy::S_METIS;
 if(ord != 0) // selecting metis ordering
   sym_chol->sym_order = sym_lib::parsy::SYM_ORDER::S_METIS;
 // changing LBC params
 sym_chol->level_param = p2;
 sym_chol->final_seq_node = p3;
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
 size_t  nnz_l = sym_chol->L->p[n];
 double total_flops = sym_chol->L->fl + (2*nnz_l) + n;

#ifdef MRHS
/// Method 2 of calling Cholesky with multiple RHS
 auto *rhs = new double[3*n];
 std::fill_n(rhs, 3*n, 1.0);
 auto *sym_chol1 = new sym_lib::parsy::SolverSettings(H);
 sym_chol1->ldl_variant = 1;
 sym_chol1->req_ref_iter = 0;
 sym_chol1->solver_mode = 0;
// sym_chol->sym_order = sym_lib::parsy::S_METIS;
 //sym_chol->sym_order = sym_lib::parsy::SYM_ORDER::S_METIS;
 timing_measurement symbolic_time1, factor_time1, solve_time1;
 symbolic_time1.start_timer();
 sym_chol1->symbolic_analysis();
 symbolic_time1.measure_elapsed_time();
 factor_time1.start_timer();
 sym_chol1->numerical_factorization();
 factor_time1.measure_elapsed_time();
 solve_time1.start_timer();
 double *x1 = sym_chol1->solve_only(rhs, 3);
 solve_time1.measure_elapsed_time();
 sym_chol1->compute_norms(rhs);
 //print_vec("x: ", 0, 10, x); std::cout<<"\n";
 //print_vec("x[0]: ", 0, 10, x1); std::cout<<"\n";
 //print_vec("x[1]: ", 0, 10, x1 + n); std::cout<<"\n";
 //print_vec("x[2]: ", 0, 10, x1 + 2*n); std::cout<<"\n";
#endif

 if(header){
  print_common_header();
  std::cout<<TOOL<<",LBC P1,LBC P2,";
  std::cout<<SYM_TIME<<","<<FCT_TIME","<<SOLVE_TIME<<","<<RESIDUAL<<",";
  std::cout<<FLOPS<<",";
  std::cout<<"\n";
 }

 print_common(matrix_name, "Cholesky", "", L1_csc, sym_chol->L->p[n], num_threads);

 if(mode == 1)
  PRINT_CSV("Sequential Sympiler");
 else
  PRINT_CSV("Parallel Sympiler");
 PRINT_CSV(p2);
 PRINT_CSV(p3);
 PRINT_CSV(symbolic_time.elapsed_time);
 PRINT_CSV(factor_time.elapsed_time);
 PRINT_CSV(solve_time.elapsed_time);
 PRINT_CSV(sym_chol->res_l1);
 PRINT_CSV(total_flops);

#ifdef MRHS
 PRINT_CSV(solve_time1.elapsed_time);
 PRINT_CSV(sym_chol1->res_l1);
 delete sym_chol1;
 delete []rhs;
#endif

 delete []solution;
 //delete A;
 delete L1_csc;
 delete H;
 delete sym_chol;


 return 0;
}
