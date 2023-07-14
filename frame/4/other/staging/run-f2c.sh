#!/bin/bash

main()
{
    echo "Running 'f2c' on current directory."
    echo ""

	src_dir="netlib"
	prefix="f2c_"
	
	files=$(ls ${src_dir}/*.f)

    for file in ${files}; do

		filename=${file##*/}

		#tmpfile=$(echo "${file}" | sed -e "s/\//\/${prefix}/g" )
		#tmpfilec=$(echo "${tmpfile}" | sed -e "s/\.f/\.c/g")
		tmpfile=${prefix}${filename}

		# Make a temporary local copy of the .f file.
		cp ${file} ${tmpfile}

		# Convert the Fortran file to C.
		echo "Running f2c on ${tmpfile}"
		f2c -A -R -a ${tmpfile}

		# Remove the temporary local .f file.
		echo "Removing ${tmpfile}"
		rm -f ${tmpfile}
	done

    return 0
}

main "$@"
