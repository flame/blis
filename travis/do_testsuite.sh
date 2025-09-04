#!/bin/bash

set -e
set -x

export BLIS_JC_NT=1
export BLIS_IC_NT=2
export BLIS_JR_NT=1
export BLIS_IR_NT=1

if [ "$TEST" = "FAST" -o "$TEST" = "ALL" ]; then
    make testblis-fast
    $DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
fi

if [ "$TEST" = "MD" -o "$TEST" = "ALL" ]; then
	make testblis-md
    $DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
fi

if [ "$TEST" = "SALT" -o "$TEST" = "ALL" ]; then
	# Disable multithreading within BLIS.
	export BLIS_JC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
	make testblis-salt
    $DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
fi

if [ "$TEST" = "1" -o "$TEST" = "ALL" ]; then
    make testblis
    $DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
fi

make testblas
$DIST_PATH/blastest/check-blastest.sh

