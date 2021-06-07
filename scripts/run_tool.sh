#!/bin/bash

TOOLBIN=$1
DATAPATH=$2
THREADS=$3
VAR=$4
header=1
#echo "Running $TOOLBIN for dataset in $DATAPATH ..."

MTXS=$(find $DATAPATH -name "*.mtx"  -type f | sort -t '\0')


for f in $MTXS; do
 if [ $header -eq 1 ]; then
  $TOOLBIN $f $THREADS 1 $VAR
  header=0
 else
  $TOOLBIN $f $THREADS 0 $VAR
 fi
 echo ""

done

