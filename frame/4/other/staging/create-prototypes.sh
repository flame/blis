#!/bin/bash

main()
{
    # Get the script name
    script_name=${0##*/}

	src_dir="."
	prefix="f2c_"

	header="./f2c_lapack.h"
	tmpfile="${src_dir}/${header}.tmp"
	touch ${tmpfile}

	rtypes="void bla_logical bla_integer bla_real bla_double int"

	for rt in ${rtypes}; do

		echo "${script_name}: Generating prototypes for functions that return '${rt}'"

		cat ${src_dir}/*.c \
		| grep  "^ ${rt} [a-z0-9_]*_(" \
		| sed "s/^ ${rt} /${rt} /g"  \
		>> ${tmpfile}

	done

	cat ${tmpfile} \
	| sed "s/) {/ );/g" \
	| sed "s/(/( /g" \
	> ${header}

	rm -f ${tmpfile}

	return 0
}

main "$@"
