#!/bin/sh

THRDS=24
BINLIB=$1
PATHMAIN=$2
TUNED=$3
THRDS=$4
CMETIS=$5


#echo $BINLIB $PATHMAIN
#load module intel
export OMP_NUM_THREADS=$THRDS
export MKL_NUM_THREADS=$THRDS


MATSPD="Flan_1565.mtx Bump_2911.mtx Queen_4147.mtx audikw_1.mtx Serena.mtx Geo_1438.mtx Hook_1498.mtx bone010.mtx ldoor.mtx boneS10.mtx Emilia_923.mtx PFlow_742.mtx inline_1.mtx nd24k.mtx Fault_639.mtx StocF-1465.mtx bundle_adj.mtx msdoor.mtx af_shell7.mtx af_shell8.mtx af_shell4.mtx af_shell3.mtx af_3_k101.mtx af_1_k101.mtx af_4_k101.mtx af_5_k101.mtx af_0_k101.mtx af_2_k101.mtx nd12k.mtx crankseg_2.mtx BenElechi1.mtx pwtk.mtx bmwcra_1.mtx crankseg_1.mtx hood.mtx m_t1.mtx x104.mtx thermal2.mtx G3_circuit.mtx bmw7st_1.mtx nd6k.mtx consph.mtx boneS01.mtx tmt_sym.mtx ecology2.mtx apache2.mtx shipsec5.mtx thread.mtx s3dkq4m2.mtx pdb1HYS.mtx offshore.mtx cant.mtx ship_001.mtx ship_003.mtx smt.mtx s3dkt3m2.mtx parabolic_fem.mtx Dubcova3.mtx shipsec1.mtx shipsec8.mtx nd3k.mtx cfd2.mtx nasasrb.mtx ct20stif.mtx vanbody.mtx oilpan.mtx cfd1.mtx qa8fm.mtx 2cubes_sphere.mtx thermomech_dM.mtx raefsky4.mtx msc10848.mtx denormal.mtx bcsstk36.mtx msc23052.mtx Dubcova2.mtx gyro.mtx gyro_k.mtx olafu.mtx"

if [ "$TUNED" == 2 ]; then
  k=4
for mat in $MATSPD; do
#for lparm in {1..10}; do
#	for cparm in {1,2,3,4,5,10,20}; do
	$BINLIB  $PATHMAIN/$mat $k $header $THRDS
	echo ""
	if [ "$header" == 1 ]; then
     header=0
  fi
#done
#done
done

fi



MATSP="Flan_1565.mtx bone010.mtx Hook_1498.mtx af_shell10.mtx Emilia_923.mtx StocF-1465.mtx af_0_k101.mtx ted_B_unscaled.mtx"
if [ "$TUNED" ==  4 ]; then
for mat in $MATSP; do
k=4
	$BINLIB  "$PATHMAIN/${mat}" $k $header $THRDS
	echo ""
	if [ "$header" == 1 ]; then
          header=0
        fi
done
fi


