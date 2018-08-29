#!/bin/bash

set -e
set -x

export BLIS_IC_NT=2
export BLIS_JC_NT=1
export BLIS_IR_NT=1
export BLIS_JR_NT=1

make testblis
$DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
make testblas
$DIST_PATH/blastest/check-blastest.sh

