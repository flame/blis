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
	#if [ $# != "1" ]; then
	#	print_usage
	#fi

	cp *.[ch] ../../f2c/

}

main "$@"
