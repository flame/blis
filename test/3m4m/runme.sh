#!/bin/bash

# File pefixes.
exec_root="test"
out_root="output"

#sys="blis"
#sys="stampede"
sys="lonestar"
#sys="wahlberg"

# Bind threads to processors.
#export OMP_PROC_BIND=true
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7"
#export GOMP_CPU_AFFINITY="0 2 4 6 1 3 5 7"
#export GOMP_CPU_AFFINITY="0 4 1 5 2 6 3 7"
#export GOMP_CPU_AFFINITY="0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 32 33 36 37 40 41 44 45"
#export GOMP_CPU_AFFINITY="0 2 4 6 8 10 12 14 16 18 20 22 1 3 5 7 9 11 13 15 17 19 21 23"
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"

# Modify LD_LIBRARY_PATH.
if [ ${sys} = "blis" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

elif [ ${sys} = "stampede" ]; then

	# A hack to use libiomp5 with gcc.
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/13/composer_xe_2013.2.146/compiler/lib/intel64"

elif [ ${sys} = "lonestar" ]; then

	# A hack to use libiomp5 with gcc.
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/apps/intel/16.0.1.150/compilers_and_libraries_2016.1.150/linux/compiler/lib/intel64"

elif [ ${sys} = "wahlberg" ]; then

	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/flame/lib/acml/5.3.1/gfortran64_int64/lib"
	export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/flame/lib/acml/5.3.1/gfortran64_mp_int64/lib"
fi

# Threading scheme to use when multithreading
if [ ${sys} = "blis" ]; then

	jc_nt=1 # 5th loop
	ic_nt=4 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4

elif [ ${sys} = "stampede" ]; then

	jc_nt=2 # 5th loop
	ic_nt=8 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=16

elif [ ${sys} = "lonestar" ]; then

	jc_nt=2 # 5th loop
	ic_nt=12 # 3rd loop
	jr_nt=1 # 2nd loop
	ir_nt=1 # 1st loop
	nt=24

elif [ ${sys} = "wahlberg" ]; then

	jc_nt=1 # 5th loop
	ic_nt=2 # 3rd loop
	jr_nt=2 # 2nd loop
	ir_nt=1 # 1st loop
	nt=4
fi

# Threadedness to test.
#threads="mt"
#threads_r="mt"
threads="st"
threads_r="st"

# Datatypes to test.
dts="z c"
dts_r="d s"

# Operations to test.
l3_ops="gemm"
test_ops="${l3_ops}"
test_ops_r="${l3_ops}"

# Complex domain implementations to test.
if [ ${sys} = "blis" ]; then

	#test_impls="openblas mkl 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis"
	test_impls="openblas 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"

elif [ ${sys} = "stampede" ]; then

	test_impls="openblas mkl asm_blis 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"
	#test_impls="openblas mkl asm_blis"

elif [ ${sys} = "lonestar" ]; then

	test_impls="asm_blis 4mhw_blis 4m1a_blis 1m_blis 3m1_blis"
	#test_impls="1m_blis 3m1_blis"
	#test_impls="4m1a_blis"
	#test_impls="mkl"
	#test_impls="openblas mkl asm_blis"

elif [ ${sys} = "wahlberg" ]; then

	test_impls="openblas acml asm_blis 3mhw_blis 3m3_blis 3m2_blis 3m1_blis 4mhw_blis 4m1b_blis 4m1a_blis 1m_blis"
	test_impls="openblas acml asm_blis"
fi

# Real domain implementations to test.
#test_impls_r="openblas mkl asm_blis"
test_impls_r="asm_blis"
#test_impls_r=""

# First perform real test cases.
for th in ${threads_r}; do

	for dt in ${dts_r}; do

		for im in ${test_impls_r}; do

			for op in ${test_ops_r}; do

				# Set the number of threads according to th.
				if [ ${th} = "mt" ]; then

					export BLIS_JC_NT=${jc_nt}
					export BLIS_IC_NT=${ic_nt}
					export BLIS_JR_NT=${jr_nt}
					export BLIS_IR_NT=${ir_nt}
					export OMP_NUM_THREADS=${nt}

					# Unset GOMP_CPU_AFFINITY for MKL when using mkl_intel_thread.
					#if [ ${im} = "mkl" ]; then

					#	export GOMP_CPU_AFFINITY=""
					#	export MKL_NUM_THREADS=${nt}
					#else
					#	export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
					#fi
				else

					export BLIS_JC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
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

# Now perform complex test cases.
for th in ${threads}; do

	for dt in ${dts}; do

		for im in ${test_impls}; do

			for op in ${test_ops}; do

				# Set the number of threads according to th.
				if [ ${th} = "mt" ]; then

					export BLIS_JC_NT=${jc_nt}
					export BLIS_IC_NT=${ic_nt}
					export BLIS_JR_NT=${jr_nt}
					export BLIS_IR_NT=${ir_nt}
					export OMP_NUM_THREADS=${nt}

					# Unset GOMP_CPU_AFFINITY for MKL when using mkl_intel_thread.
					#if [ ${im} = "mkl" ]; then

					#	export GOMP_CPU_AFFINITY=""
					#else
					#	export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
					#fi
				else

					export BLIS_JC_NT=1
					export BLIS_IC_NT=1
					export BLIS_JR_NT=1
					export BLIS_IR_NT=1
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
