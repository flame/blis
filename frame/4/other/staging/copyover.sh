#!/bin/bash

print_usage()
{
    local script_name

    # Get the script name
    script_name=${0##*/}

    # Echo usage info
    echo " "
    echo " "${script_name}
    echo " "

    # Exit with non-zero exit status
    exit 1
}

main()
{
	if [ $# != "1" ]; then
		print_usage
	fi

	srcdirpath="$1"
	destdirpath="."

	# Acquire a list of the directory's contents.
	ffiles=$(ls ${destdirpath}/*.f)

	for f in ${ffiles}; do

		echo "Copying ${f} from ${destdirpath}"

		# Copy the file.
		cp ${srcdirpath}/${f} ${destdirpath}/${f}
	done
}

main "$@"
