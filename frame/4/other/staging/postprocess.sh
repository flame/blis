#!/bin/bash

main()
{
	# local variables
	script_name=${0##*/}

	src_dir="."
	prefix="f2c_"

	# check src directory
	if [ ! -d "${src_dir}" ]; then
		echo "${script_name}: Source directory does not exist (${src_dir})."
		exit 1
	fi

	declare -a words=( \
	  'f90_cycle__()'              'continue' \
	  'f90_exit__()'               'break' \
	  'include \"f2c.h\"'          'include \"blis.h\"' \
	  'max('                       'bla_a_max(' \
	  'min('                       'bla_a_min(' \
	  'pow_ri('                    'bla_pow_ri(' \
	  'pow_di('                    'bla_pow_di(' \
	  'cdotu_'                     'f2c_cdotu_' \
	  'cdotc_'                     'f2c_cdotc_' \
	  'zdotu_'                     'f2c_zdotu_' \
	  'zdotc_'                     'f2c_zdotc_' \
	  'lsame_'                     'bla_lsame_' \
	  'slamch_'                    'bla_slamch_' \
	  'dlamch_'                    'bla_dlamch_' \
	  'r_abs'                      'bla_r_abs' \
	  'd_abs'                      'bla_d_abs' \
	  'c_abs'                      'bla_c_abs' \
	  'z_abs'                      'bla_z_abs' \
	  'i_len'                      'bla_i_len' \
	  'i_nint'                     'bla_i_nint' \
	  's_cmp'                      'bla_s_cmp' \
	  's_copy'                     'bla_s_copy' \
	  'integer'                    'bla_integer' \
	  'doublereal'                 'bla_double' \
	  'doublecomplex'              'bla_dcomplxx' \
	  'complex'                    'bla_scomplex' \
	  'bla_dcomplxx'               'bla_dcomplex' \
	  'logical'                    'bla_logical' \
	  '\/\* Subroutine \*\/ '      '' \
	  '\/\* Complex \*\/ '         '' \
	  '\/\* Double Complex \*\/ '  '' \
	  'VOID'                       'void' \
	)

	# create a sed stack
	files="$(find ${src_dir} -maxdepth 1 -name "${prefix}*.c")"
	stack="sed \"s/${words[0]}/${words[1]}/g\""
	for (( counter=2; counter<${#words[@]}; counter+=2 )); do
		stack="${stack} | sed \"s/${words[${counter}]}/${words[${counter}+1]}/g\""
	done
	echo "${script_name}: Filter on '*.c':                        "
	echo "${script_name}: ${stack}"

	# execute the stacked filter
	for file in ${files}; do
		echo -ne "${script_name}: Replacing ... ${file}                 "\\r
		tmp_file="${file}.back"
		( cp -f ${file} ${tmp_file} ; 
		  eval "cat ${file} | ${stack}"  > ${tmp_file} ;
		  mv ${tmp_file} "${file}" ; rm -f ${tmp_file} ) 
	done


	declare -a words_sc=( \
	  ' abs('        ' bla_r_abs(' \
	  '(abs('        '(bla_r_abs(' \
	  'aimag('       'r_imag(' \
	  'real('        'r_real(' \
	)

	# create a sed stack
	files_c="$(find ${src_dir} -maxdepth 1 -name "${prefix}[sc]*.c")"
	stack_c="sed \"s/${words_sc[0]}/${words_sc[1]}/g\""
	for (( ii=2; ii<${#words_sc[@]}; ii+=2 )); do
		stack_c="${stack_c} | sed \"s/${words_sc[${ii}]}/${words_sc[${ii}+1]}/g\""
	done
	echo "${script_name}: Filter on '[sc]*.c':                        "
	echo "${script_name}: ${stack_c}"

	# execute the stacked filter
	for file in ${files_c}; do
		echo -ne "${script_name}: Replacing ... ${file}                 "\\r
		tmp_file="${file}.back"
		( cp -f ${file} ${tmp_file} ; 
		  eval "cat ${file} | ${stack_c}"  > ${tmp_file} ;
		  mv ${tmp_file} "${file}" ; rm -f ${tmp_file} ) 
	done


	declare -a words_dz=( \
	  ' abs('        ' bla_d_abs(' \
	  '(abs('        '(bla_d_abs(' \
	  'aimag('       'd_imag(' \
	  'real('        'd_real(' \
	)

	# create a sed stack
	files_z="$(find ${src_dir} -maxdepth 1 -name "${prefix}[dz]*.c")"
	stack_z="sed \"s/${words_dz[0]}/${words_dz[1]}/g\""
	for (( ii=2; ii<${#words_dz[@]}; ii+=2 )); do
		stack_z="${stack_z} | sed \"s/${words_dz[${ii}]}/${words_dz[${ii}+1]}/g\""
	done
	echo "${script_name}: Filter on '[dz]*.c':                        "
	echo "${script_name}: ${stack_z}"

	# execute the stacked filter
	for file in ${files_z}; do
		echo -ne "${script_name}: Replacing ... ${file}                 "\\r
		tmp_file="${file}.back"
		( cp -f ${file} ${tmp_file} ; 
		  eval "cat ${file} | ${stack_z}"  > ${tmp_file} ;
		  mv ${tmp_file} "${file}" ; rm -f ${tmp_file} ) 
	done


	return 0
}

main "$@"
