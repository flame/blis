#!/bin/bash

set -e
set -x

export BLIS_JC_NT=1
export BLIS_IC_NT=2
export BLIS_JR_NT=1
export BLIS_IR_NT=1
export BLIS_THREAD_IMPL="single"

TO_TEST="$CONF"
ARM64_QEMU="no"
if [ "$CONF" = "arm64" ] && [ "$TESTSUITE_WRAPPER" != "" ]; then
	TO_TEST="armsve firestorm thunderx2 cortexa57 cortexa53 m4sme_p m4sme_e"
	ARM64_QEMU="yes"
	TESTSUITE_WRAPPER_SAVE="$TESTSUITE_WRAPPER"
fi

for THIS_CONF in $TO_TEST; do

	echo "Testing configuration: $THIS_CONF"

	if [ "$ARM64_QEMU" = "yes" ]; then
		case $THIS_CONF in
			armsve)
				CPU="max,sve512=on,sme=off"
				;;
			firestorm)
			thunderx2)
			cortexa57)
			cortexa53)
				CPU="max,sve=off,sme=off"
				;;
			m4sme_p)
			m4sme_e)
				CPU="max,sve=off,sme512=on"
				;;
			*)
				echo "Unknown arm64 configuration: $THIS_CONF"
				exit 1
				;;
		esac
		export BLIS_ARCH_TYPE="$THIS_CONF"
		export TESTSUITE_WRAPPER=$(echo "$TESTSUITE_WRAPPER_SAVE" | sed "s/XXX/$CPU/")
	fi

	if [ "$TEST" = "FAST" -o "$TEST" = "ALL" ]; then
		make T=1 testblis-fast
		$DIST_PATH/testsuite/check-blistest.sh ./output.testsuite

		for impl in $(echo $THR | sed 's/none//' | tr , ' '); do
			export BLIS_THREAD_IMPL="$impl"
			make T=1 testblis-fast
			$DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
		done
	fi

	if [ "$TEST" = "MD" -o "$TEST" = "ALL" ]; then
		make T=1 testblis-md
		$DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
	fi

	if [ "$TEST" = "SALT" -o "$TEST" = "ALL" ]; then
		# Disable multithreading within BLIS.
		export BLIS_JC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
		make T=1 testblis-salt
		$DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
	fi

	if [ "$TEST" = "1" -o "$TEST" = "ALL" ]; then
		make T=1 testblis
		$DIST_PATH/testsuite/check-blistest.sh ./output.testsuite
	fi

	export BLIS_THREAD_IMPL="single"
	make T=1 testblas
	$DIST_PATH/blastest/check-blastest.sh

done

