#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

sys="blis"
#sys="stampede2"
#sys="lonestar5"

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 2 4 6 1 3 5 7"
#export GOMP_CPU_AFFINITY="0 4 1 5 2 6 3 7"
#export GOMP_CPU_AFFINITY="0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 32 33 36 37 40 41 44 45"
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 16 18 20 22 1 3 5 7 9 11 13 15 17 19 21 23"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
export GOMP_CPU_AFFINITY="0 1 2 3"

# Modify LD_LIBRARY_PATH.
if [ ${sys} = "blis" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

elif [ ${sys} = "stampede2" ]; then

	:

elif [ ${sys} = "lonestar5" ]; then

	# A hack to use libiomp5 with gcc.
	#export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64"
	:

fi

# Threading scheme to use when multithreading
if [ ${sys} = "blis" ]; then

	jc_nt=2 # 5th loop
	ic_nt=2 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "stampede2" ]; then

	jc_nt=2 # 5th loop
	ic_nt=8 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=16

elif [ ${sys} = "lonestar5" ]; then

	jc_nt=4 # 5th loop
	ic_nt=6 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=24

fi

# Complex domain implementations to test.
if [ ${sys} = "blis" ]; then

	test_impls="openblas asm_blis"

elif [ ${sys} = "stampede2" ]; then

	test_impls="openblas asm_blis mkl"

elif [ ${sys} = "lonestar5" ]; then

	test_impls="openblas mkl asm_blis"
fi

# Datatypes to test.
#dts="s d c z"

# Operations to test.
l3_ops="gemm"
test_ops="${l3_ops}"

# Define the list of datatype chars and precision chars.
dt_chars="s d c z"
pr_chars="s d"

# Construct the datatype combination strings.
dt_combos=""
for dtc in ${dt_chars}; do
	for dta in ${dt_chars}; do
		for dtb in ${dt_chars}; do
			for pre in ${pr_chars}; do
				dt_combos="${dt_combos} ${dtc}${dta}${dtb}${pre}"
			done
		done
	done
done

# Threadedness to test.
threads="mt"
#threads="st"

test_impls="openblas"

#dt_combos="ssss sssd ssds sdss dsss ddds dddd"
#dt_combos="csss csds cdss cdds zsss zsds zdss zdds cssd csdd cdsd cddd zssd zsdd zdsd zddd"
#dt_combos="cssd csdd cdsd cddd zsss zsds zdss zdds"
#dt_combos="cdsd cddd zsss zsds zdss zdds"
#test_impls="asm_blis"

# Now perform complex test cases.
for th in ${threads}; do

	for dt in ${dt_combos}; do

		for im in ${test_impls}; do

			for op in ${test_ops}; do

				# Set the number of threads according to th.
				if [ ${th} = "mt" ]; then

					export BLIS_JC_NT=${jc_nt}
					export BLIS_IC_NT=${ic_nt}
					export BLIS_JR_NT=${jr_nt}
					export BLIS_IR_NT=${ir_nt}
					export OMP_NUM_THREADS=${nt}
					export OPENBLAS_NUM_THREADS=${nt}

					# Unset GOMP_CPU_AFFINITY for OpenBLAS, as it causes the library
					# to execute sequentially.
					if [ ${im} = "openblas" ]; then
						unset GOMP_CPU_AFFINITY
					else
						export GOMP_CPU_AFFINITY="0 1 2 3"
					fi
				else

					export BLIS_JC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
					export OMP_NUM_THREADS=1
					export OPENBLAS_NUM_THREADS=1
				fi

				# Construct the name of the test executable.
				exec_name="${exec_root}_${dt}${op}_${im}_${th}.x"

				# Construct the name of the output file.
				out_file="${out_root}_${th}_${dt}${op}_${im}.m"

				echo "Running (nt = ${OMP_NUM_THREADS}) ./${exec_name} > ${out_file}"

				# Run executable.
				./${exec_name} > ${out_file}

				#sleep 1

			done
		done
	done
done
