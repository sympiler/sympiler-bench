#!/bin/bash
BINLIB=$1
TUNED=$2
THRDS=$3
MAT_DIR=$4
SPD_MAT_DIR=$5

export OMP_NUM_THREADS=$THRDS
export MKL_NUM_THREADS=$THRDS

header=1

MATS=($(ls  $MAT_DIR))
SPD_MATS=($(ls $SPD_MAT_DIR))

if [ "$TUNED" ==  1 ]; then
  for bp in {0,1}; do
    for mat in "${SPD_MATS[@]}"; do
      for k in {2,3,4,5,6,7,8,9}; do
        # for cparm in {1,2,3,4,5,10,20}; do
        if [ $header -eq 1 ]; then
          $BINLIB  -m $SPD_MAT_DIR/$mat/$mat.mtx -n SPTRS -s CSR -t $THRDS -c $k -d -p $bp -u 1
          header=0
        else
          $BINLIB  -m $SPD_MAT_DIR/$mat/$mat.mtx -n SPTRS -s CSR -t $THRDS -c $k -p $bp -u 1
        fi
      done
    done
  done
fi

MAT_SPD_NAME="bcsstk15/bcsstk15 bcsstk16/bcsstk16 bcsstk17/bcsstk17 bcsstk18/bcsstk18 bcsstk24/bcsstk24 bcsstk25/bcsstk25 bcsstk28/bcsstk28 bcsstk36/bcsstk36 bcsstk38/bcsstk38 crystm01/crystm01 crystm02/crystm02 crystm03/crystm03 ct20stif/ct20stif msc10848/msc10848 msc23052/msc23052 pwtk/pwtk finan512/finan512 nasa2910/nasa2910 nasa4704/nasa4704 nasasrb/nasasrb aft01/aft01 cfd1/cfd1 cfd2/cfd2 olafu/olafu raefsky4/raefsky4 qa8fm/qa8fm bodyy4/bodyy4 bodyy5/bodyy5 bodyy6/bodyy6 Andrews/Andrews nd3k/nd3k nd6k/nd6k nd12k/nd12k nd24k/nd24k af_shell3/af_shell3 af_shell4/af_shell4 af_shell7/af_shell7 af_shell8/af_shell8 Pres_Poisson/Pres_Poisson gyro_k/gyro_k gyro_m/gyro_m t2dah_e/t2dah_e audikw_1/audikw_1 bmw7st_1/bmw7st_1 bmwcra_1/bmwcra_1 crankseg_1/crankseg_1 crankseg_2/crankseg_2 hood/hood inline_1/inline_1 ldoor/ldoor m_t1/m_t1 oilpan/oilpan s3dkq4m2/s3dkq4m2 s3dkt3m2/s3dkt3m2 ship_001/ship_001 ship_003/ship_003 shipsec1/shipsec1 shipsec5/shipsec5 shipsec8/shipsec8 thread/thread vanbody/vanbody wathen100/wathen100 wathen120/wathen120 x104/x104 cvxbqp1/cvxbqp1 gridgena/gridgena jnlbrng1/jnlbrng1 minsurfo/minsurfo obstclae/obstclae torsion1/torsion1 Kuu/Kuu Muu/Muu bundle1/bundle1 thermal1/thermal1 thermal2/thermal2 ted_B/ted_B ted_B_unscaled/ted_B_unscaled G2_circuit/G2_circuit G3_circuit/G3_circuit apache1/apache1 apache2/apache2 gyro/gyro bone010/bone010 boneS01/boneS01 boneS10/boneS10 af_0_k101/af_0_k101 af_1_k101/af_1_k101 af_2_k101/af_2_k101 af_3_k101/af_3_k101 af_4_k101/af_4_k101 af_5_k101/af_5_k101 s1rmq4m1/s1rmq4m1 s2rmq4m1/s2rmq4m1 s3rmq4m1/s3rmq4m1 s1rmt3m1/s1rmt3m1 s2rmt3m1/s2rmt3m1 s3rmt3m1/s3rmt3m1 s3rmt3m3/s3rmt3m3 msdoor/msdoor Dubcova1/Dubcova1 Dubcova2/Dubcova2 Dubcova3/Dubcova3 BenElechi1/BenElechi1 parabolic_fem/parabolic_fem ecology2/ecology2 denormal/denormal tmt_sym/tmt_sym smt/smt cbuckle/cbuckle 2cubes_sphere/2cubes_sphere Trefethen_20000b/Trefethen_20000b Trefethen_20000/Trefethen_20000 thermomech_TC/thermomech_TC thermomech_TK/thermomech_TK thermomech_dM/thermomech_dM shallow_water1/shallow_water1 shallow_water2/shallow_water2 offshore/offshore pdb1HYS/pdb1HYS consph/consph cant/cant Serena/Serena Emilia_923/Emilia_923 Fault_639/Fault_639 Flan_1565/Flan_1565 Geo_1438/Geo_1438 Hook_1498/Hook_1498 StocF-1465/StocF-1465 Bump_2911/Bump_2911 Queen_4147/Queen_4147 PFlow_742/PFlow_742 bundle_adj/bundle_adj"


if [ "$TUNED" ==  5 ]; then
  for bp in {0,1}; do
    for mat in ${MAT_SPD_NAME}; do
      for k in {2,3,4,5,6,7,8,9}; do
        # for cparm in {1,2,3,4,5,10,20}; do
        if [ $header -eq 1 ]; then
          $BINLIB  -m $SPD_MAT_DIR/$mat.mtx -n SPTRS -s CSR -t $THRDS -c $k -d -p $bp -u 1
          header=0
        else
          $BINLIB  -m $SPD_MAT_DIR/$mat.mtx -n SPTRS -s CSR -t $THRDS -c $k -p $bp -u 1
        fi
      done
    done
  done
fi


prefer_fsc=( "" )
clt_min_widths=( 2 4 8 16 )
clt_max_distances=( 2 4 8 )

### SPMV
if [ "$TUNED" == 3 ]; then
  for mat in "${MATS[@]}"; do
    for md in "${clt_max_distances[@]}"; do
      for mw in "${clt_min_widths[@]}"; do
        for pf in "${prefer_fsc[@]}"; do
          if [ $header -eq 1 ]; then
            $BINLIB  -m $MAT_DIR/$mat/$mat.mtx -n SPMV -s CSR -t $THRDS $pf --clt_width=$mw --col_th=$md -d
            header=0
          else
            $BINLIB  -m $MAT_DIR/$mat/$mat.mtx -n SPMV -s CSR -t $THRDS $pf --clt_width=$mw --col_th=$md
          fi
        done
      done
    done
  done
fi

M_TILE_SIZES=( 4 8 16 32 )
N_TILE_SIZES=( 4 8 16 32 )
B_MAT_COL=( 256 )
### SPMM
if [ "$TUNED" == 4 ]; then
  for mat in "${MATS[@]}"; do
    for md in "${clt_max_distances[@]}"; do
      for mw in "${clt_min_widths[@]}"; do
        for mtile in "${M_TILE_SIZES[@]}"; do
          for ntile in "${N_TILE_SIZES[@]}"; do
            for bcol in "${B_MAT_COL[@]}"; do
              if [ "$ntile" -gt "$bcol" ]; then
                continue
              fi
              if [ $header -eq 1 ]; then
                $BINLIB -m $MAT_DIR/$mat/$mat.mtx -n SPMM -s CSR -t $THRDS --m_tile_size=$mtile --n_tile_size=$ntile --b_matrix_columns=$bcol -d --clt_width=$mw --col_th=$md
                header=0
              else
                $BINLIB -m $MAT_DIR/$mat/$mat.mtx -n SPMM -s CSR -t $THRDS --b_matrix_columns=$bcol --m_tile_size=$mtile --n_tile_size=$ntile --clt_width=$mw --col_th=$md
              fi
            done
          done
        done
      done
    done
  done
fi
