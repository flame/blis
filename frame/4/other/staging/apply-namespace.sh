#!/bin/bash

main()
{
	# local variables
	script_name=${0##*/}

	src_dir="."
	prefix="f2c_"

	header="f2c_lapack.h"
	tmpfile="${header}.tmp"


	funcs="$(cat f2c_lapack.h | cut -f1 -d'(' | cut -f2 -d' ')"

	stack="cat"
	for func in ${funcs}; do
		newfunc="${prefix}${func}"
		stack="${stack} | sed \"s/${func}/${newfunc}/g\""
	done
	echo "${script_name}: Filter on '${prefix}*.c':                        "
	echo "${script_name}: ${stack}"


	files="$(find ${src_dir} -maxdepth 1 -name "${prefix}*.c")"
	files="${files} ${header}"

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
