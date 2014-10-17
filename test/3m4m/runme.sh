#!/bin/bash

#export BLIS_IC_NT=8
#export BLIS_JC_NT=2

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15"
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

# File pefixes.
exec_root="test"
out_root="output"

# Threadedness to test.
threads="mt"

# Threading scheme to use when multithreading
ic_nt=8
jc_nt=2
nt=16

# Datatypes to test.
dts="z c"

# Operations to test.
l3_ops="gemm"
test_ops="${l3_ops}"

# Implementations to test
test_impls="openblas asm_blis 4m1_blis 4mhw_blis 3m1_blis 3mhw_blis"


for th in ${threads}; do

	for dt in ${dts}; do

		for im in ${test_impls}; do

			for op in ${test_ops}; do

				# Set the number of threads according to th.
				if [ ${th} = "mt" ]; then

					export BLIS_IC_NT=${ic_nt}
					export BLIS_JC_NT=${jc_nt}
					export OMP_NUM_THREADS=${nt}

				else

					export BLIS_IC_NT=1
					export BLIS_JC_NT=1
					export OMP_NUM_THREADS=1
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${dt}${op}_${im}_${th}.x"

				# Construct the name of the output file.
				out_file="${out_root}_${th}_${dt}${op}_${im}.m"

				echo "Running (nt = ${OMP_NUM_THREADS}) ./${exec_name} > ${out_file}"

				# Run executable.
				./${exec_name} > ${out_file}

				sleep 1

			done
		done
	done
done
