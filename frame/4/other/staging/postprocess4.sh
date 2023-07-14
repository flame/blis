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
	  ' real '                     ' bla_real ' \
	  '(real'                      '(bla_real' \
	  'r_real('                    'bla_r_real(' \
	  'd_real('                    'bla_d_real(' \
	  'r_imag('                    'bla_r_imag(' \
	  'd_imag('                    'bla_d_imag(' \
	  'r_cnjg('                    'bla_r_cnjg(' \
	  'd_cnjg('                    'bla_d_cnjg(' \
	  'r_sign('                    'bla_r_sign(' \
	  'd_sign('                    'bla_d_sign(' \
	  '\.r'                        '\.real' \
	  '\.i'                        '\.imag' \
	  '->r'                        '->real' \
	  '->i'                        '->imag' \
	  # Remove ftnlen arguments from BLAS calls (but NOT calls to other
	  # LAPACK functions).
	  '([ ]*ftnlen[ ]*)[ ]*\([0-9]*\)' '(ftnlen)\1' \
	  ' \([sdcz]gemm_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]hemm_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]herk_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]her2k_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]symm_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]syrk_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]syr2k_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]trmm_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*, (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]trsm_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*, (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]gemv_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]hemv_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]her_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]her2_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]symv_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]syr_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]syr2_.*\), (ftnlen)[0-9]*)'                   ' \1)' \
	  ' \([sdcz]trmv_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	  ' \([sdcz]trsv_.*\), (ftnlen)[0-9]*, (ftnlen)[0-9]*, (ftnlen)[0-9]*)'   ' \1)' \
	)

	# create a sed stack
	files="$(find ${src_dir} -maxdepth 1 -name "${prefix}*.c")"
	stack="sed \"s/${words[0]}/${words[1]}/g\""
	for (( ii=2; ii<${#words[@]}; ii+=2 )); do
		stack="${stack} | sed \"s/${words[${ii}]}/${words[${ii}+1]}/g\""
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

	return 0
}

main "$@"
