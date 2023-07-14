#!/bin/bash

main()
{
	# local variables
	script_name=${0##*/}

	declare -a words=( \
	  ' CYCLE'               ' CALL F90_CYCLE' \
	  ' EXIT'                ' CALL F90_EXIT' \
	  ', INTENT(IN) ::'      ' ' \
	  ', IPARAM2STAGE'       '' \
	)

	src_dir="netlib"

	# check src directory
	if [ ! -d "${src_dir}" ]; then
		echo "${script_name}: Source directory does not exist (${src_dir})."
		exit 1
	fi
	
	# create a sed stack
	files="$(find ${src_dir} -maxdepth 1 -name "*.f")"
	stack="sed \"s/${words[0]}/${words[1]}/g\""
	for (( counter=2; counter<${#words[@]}; counter+=2 )); do
		stack="${stack} | sed \"s/${words[${counter}]}/${words[${counter}+1]}/g\""
	done
	echo "${script_name}: Filter: "
	echo "${script_name}: ${stack}"
	echo " " 

	# execute the stacked filter
	for file in ${files}; do
		echo -ne "   Replacing ... ${file}                    "\\r
		tmp_file=$(echo "${file}.back")

		( cp -f ${file} ${tmp_file} ; 
		  eval "cat ${file} | ${stack}"  > ${tmp_file} ;
		  mv ${tmp_file} "${file}" ; 
		  rm -f ${tmp_file} ) 
	done

	return 0
}

main "$@"
