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

	declare -a words=(
	  'extern void'
	  'extern bla_logical'
	  'extern bla_integer'
	  'extern bla_real'
	  'extern bla_double'
	  'extern int'
	  'extern real'
	  '^ double sqrt('
	  '^ void r_cnjg('
	  '^ void d_cnjg('
	  '^ double r_imag('
	  '^ double d_imag('
	  '^ double r_sign('
	  '^ double d_sign('
	  '^ double log('
	  '^ double bla_c_abs('
	  '^ double bla_z_abs('
	  '^ bla_integer bla_i_len('
	  '^ bla_integer bla_i_nint('
	  '^ bla_integer bla_s_cmp('
	  '^ int bla_s_copy('
	)

	# create a sed stack
	files="$(find ${src_dir} -maxdepth 1 -name "${prefix}*.c")"

	for (( wi=0; wi<${#words[@]}; wi+=1 )); do

		pattern=${words[${wi}]}

		echo "${script_name}: Applying pattern \"${pattern}\""

		for file in ${files}; do

			cat ${file} | sed -e "/${pattern}/d" \
			            > "${file}.tmp" ;
			mv "${file}.tmp" ${file}

		done
	done

	return 0
}

main "$@"

