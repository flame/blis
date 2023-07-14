#!/bin/bash

main()
{
	# local variables
	script_name=${0##*/}

	src_dir="."
	prefix="f2c_"

	declare -a words=( \
	  'ssyev.c' 's/bla_integer \*lwork,/bla_integer \*lwork, bla_real \*rwork,/g'
	  'dsyev.c' 's/bla_integer \*lwork,/bla_integer \*lwork, bla_double \*rwork,/g'
	)

	for (( ii=0; ii<${#words[@]}; ii+=2 )); do

		file="${src_dir}/${prefix}${words[${ii}]}"
		sedcmd=${words[${ii}+1]}

		echo "${script_name}: Harmonizing real/complex signatures in ${file}"
		tmp_file="${file}.back"

		( cp -f ${file} ${tmp_file} ; 
		  eval "cat ${file} | sed -e \"${sedcmd}\""  > ${tmp_file} ;
		  mv ${tmp_file} "${file}" ; rm -f ${tmp_file} ) 
	done

	return 0
}

main "$@"
