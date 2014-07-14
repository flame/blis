#!/bin/bash
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name of The University of Texas at Austin nor the names
#     of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

#
# update-version-file.sh
#
# Field G. Van Zee
#


print_usage()
{
	#local script_name
	
	# Get the script name
	#script_name=${0##*/}
	
	# Echo usage info
	echo " "
	echo " "$script_name
	echo " "
	echo " Field G. Van Zee"
	echo " "
	echo " Checks whether the current BLIS distribution is a git clone: if so,"
	echo " it queries git to update the 'version' file with the latest version"
	echo " string; otherwise, leaves the contents of 'version' unchanged."
	echo " "
	echo " Usage:"
	echo "   ${script_name} [options] versfile"
	echo " "
	echo " Arguments:"
	echo " "
	echo "   versfile    The file where the version string is stored. If versfile is"
	echo "               is not specified, then it defaults to 'version'."
	echo " "
	echo " Options:"
	echo " "
	echo "   -o SCRIPT   output script name"
	echo "                 Use SCRIPT when outputting messages instead of the script's"
	echo "                 actual name."
	echo " "
	
	# Exit with non-zero exit status
	exit 1
}


main()
{
	# -- BEGIN GLOBAL VARIABLE DECLARATIONS --

	# The name of the script, stripped of any preceeding path.
	script_name=${0##*/}

	# The name of the default version file.
	version_file_def='version'

	# The name of the specified version file.
	version_file=''

	# Strings used during version query.
	git_describe_str=''
	new_version_str=''

	# The script name to use instead of the $0 when outputting messages.
	output_name=''

	# The git directory.
	gitdir='.git'
	
	# -- END GLOBAL VARIABLE DECLARATIONS --


	# Process our command line options.
	while getopts ":ho:" opt; do
		case $opt in
			o  ) output_name=$OPTARG ;;
			h  ) print_usage ;;
			\? ) print_usage
		esac
	done
	shift $(($OPTIND - 1))


	# If an output script name was given, overwrite script_name with it.
	if [ -n "${output_name}" ]; then

		script_name="${output_name}"
	fi


	echo "${script_name}: checking whether we need to update the version file."

	
	
	# Check the number of arguments after command line option processing.
	if [ $# = "0" ]; then

		version_file=${version_file_def}
		echo "${script_name}: not sure which version file to update; defaulting to '${version_file}'."

	elif [ $# = "1" ]; then

		version_file=$1
		echo "${script_name}: checking version file '${version_file}'."

	else
		print_usage
	fi


	# Check if the .git dir exists; if it does not, we do nothing.
	if [ -d "${gitdir}" ]; then

		echo "${script_name}: found '${gitdir}' directory; assuming git clone."

		echo "${script_name}: executing: git describe --tags."

		# Query git for the version string, which is simply the current tag,
		# followed by a number signifying how many commits have transpired
		# since the tag, followed by a 'g' and a shortened hash tab.
		git_describe_str=$(git describe --tags)

		echo "${script_name}: got back ${git_describe_str}."

		# Strip off the commit hash label.
		new_version_str=$(echo ${git_describe_str} | cut -d- -f-2)

		echo "${script_name}: truncating to ${new_version_str}."
		echo "${script_name}: updating version file '${version_file}'."

		# Write the new version string to the version file.
		echo "${new_version_str}" > ${version_file}

	fi


	# Exit peacefully.
	return 0
}


# The script's main entry point, passing all parameters given.
main "$@"
