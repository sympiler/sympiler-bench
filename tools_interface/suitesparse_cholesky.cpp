//
// Created by kazem on 6/7/21.
//

#define CSV_LOG 1
#include <aggregation/sparse_io.h>
#ifdef OPENMP
#include <omp.h>
#endif
#include "cholmod.h"
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
 size_t n;
 int num_threads = 6;
 int p2 = -1, p3 = 4000; // LBC params
 int header = 0;
 int mode = 1;
 int *perm;
 std::string matrix_name;
 std::vector<timing_measurement> time_array;
 if(argc > 2)
  num_threads = atoi(argv[2]);
 if(argc > 3)
  header = atoi(argv[3]);
 if(argc > 4)
  mode = atoi(argv[4]);
#ifdef OPENMP
 omp_set_num_threads(num_threads);
#endif
 /// Method 1 of calling Cholesky


 std::string f1 = argv[1];

 cholmod_common cm;
 cholmod_start(&cm);
 cholmod_sparse* Ac;
 cholmod_factor* L;
 cholmod_dense *b;
 FILE *ff = fopen(f1.c_str(),"r");
 Ac = cholmod_read_sparse(ff,&cm);
 n = Ac->nrow;

  b = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &cm);
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

 double total_flops = cm.fl + (2*cm.lnz) + n;//+ solve

 if(header){
  PRINT_CSV("Matrix Name,A Dimension,A Nonzero,L Nonzero,Code Type,Data Type,"
            "Metis Enabled,Number of Threads");
  std::cout<<TOOL<<",LBC P1,LBC P2,";
  std::cout<<SYM_TIME<<","<<FCT_TIME","<<SOLVE_TIME<<","<<RESIDUAL<<",";
  std::cout<<FLOPS<<","<<"Supernodal Enabled,";
  std::cout<<"\n";
 }


 PRINT_CSV(f1);
 PRINT_CSV(n);
 PRINT_CSV(Ac->nzmax);
 PRINT_CSV(cm.lnz);
 PRINT_CSV("CHOLESKY");
 PRINT_CSV("");
 PRINT_CSV("");
 PRINT_CSV(num_threads);
 PRINT_CSV("CHOLMOD");
 PRINT_CSV(p2);
 PRINT_CSV(p3);
 PRINT_CSV(symbolic_time.elapsed_time);
 PRINT_CSV(factor_time.elapsed_time);
 PRINT_CSV(solve_time.elapsed_time);
 PRINT_CSV(res);
 PRINT_CSV(total_flops);
 PRINT_CSV(L->is_super);

#ifdef MRHS
 PRINT_CSV(solve_time1.elapsed_time);
 PRINT_CSV(sym_chol1->res_l1);
 delete sym_chol1;
 delete []rhs;
#endif

// delete []solution;

 cholmod_free_sparse(&Ac, &cm);
 cholmod_free_factor(&L, &cm);
 cholmod_free_dense(&b, &cm);
 cholmod_free_dense(&x, &cm);
 cholmod_finish(&cm);

 return 0;
}
