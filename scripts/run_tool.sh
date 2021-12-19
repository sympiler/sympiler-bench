#!/bin/bash

TOOLBIN=$1
DATAPATH=$2
THREADS=$3
VAR=$4
VAR5=$5
VAR6=$6
VAR7=$7
header=1
#echo "Running $TOOLBIN for dataset in $DATAPATH ..."

MTXS=$(find $DATAPATH -name "*.mtx"  -type f | sort -t '\0')


for f in $MTXS; do
 if [ $header -eq 1 ]; then
  $TOOLBIN $f $THREADS 1 $VAR $VAR5 $VAR6 $VAR7
  header=0
 else
  $TOOLBIN $f $THREADS 0 $VAR $VAR5 $VAR6 $VAR7
 fi
 echo ""

done

