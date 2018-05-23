#!/bin/bash

set -e
set -x

export BLIS_IC_NT=2
export BLIS_JC_NT=1
export BLIS_IR_NT=1
export BLIS_JR_NT=1

travis_wait 30 make BLIS_ENABLE_TEST_OUTPUT=yes testblis
$DIST_PATH/build/check-blistest.sh ./output.testsuite
make BLIS_ENABLE_TEST_OUTPUT=yes testblas
$DIST_PATH/build/check-blastest.sh

