#!/usr/bin/env bash
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
# -- Helper functions ----------------------------------------------------------
#

print_usage()
{
	# Echo usage info.
	echo " "
	echo " ${script_name}"
	echo " "
	echo " Field G. Van Zee"
	echo " "
	echo " Generate a monolithic header by recursively replacing all #include"
	echo " directives in a selected file with the contents of the header files"
	echo " they reference."
	echo " "
	echo " Usage:"
	echo " "
	echo "   ${script_name} header header_out root_dir"
	echo " "
	echo " Arguments:"
	echo " "
	echo "   header        The filepath to the top-level header, which is file that"
	echo "                 will #include all other header files. NOTE: It is okay if"
	echo "                 this file resides somewhere in root_dir, described below."
	echo " "
	echo "   header_out    The filepath of the file into which the script will output"
	echo "                 the monolithic header."
	echo " "
	echo "   root_dir      The path to the root of the directory tree containing the"
	echo "                 headers that will be #included by 'header'. 'root_dir' is"
	echo "                 searched recursively for any directory that contains .h"
	echo "                 files, and the resulting list of directories is then"
	echo "                 searched whenever a #include directive is encountered in"
	echo "                 'header' (or any file subsequently #included). If a"
	echo "                 referenced header file is not found within 'root_dir',"
	echo "                 the #include directive is left untouched and translated"
	echo "                 directly into 'header_out'."
	echo " "
	echo " The following options are accepted:"
	echo " "
	echo "   -o SCRIPT   output script name"
	echo "                 Use SCRIPT as a prefix when outputting messages instead"
	echo "                 the script's actual name. Useful when the current script"
	echo "                 is going to be called from within another, higher-level"
	echo "                 driver script and seeing the current script's name might"
	echo "                 unnecessarily confuse the user."
	echo " "
	echo "   -r          remove C-style comments"
	echo "                 Strip comments enclosed in /* */ delimiters from the"
	echo "                 output, including multi-line comments. By default, these"
	echo "                 comments are not stripped."
	echo " "
	echo "   -q          quiet"
	echo "                 Suppress informational output. By default, the script is"
	echo "                 verbose."
	echo " "
	echo "   -h          help"
	echo "                 Output this information and exit."
	echo " "

	# Exit with non-zero exit status
	exit 1
}

canonicalize_ws()
{
	local str="$1"

	# Remove leading and trailing whitespace.
	str=$(echo -e "${str}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

	# Remove duplicate spaces between words.
	str=$(echo -e "${str}" | tr -s " ")

	# Update the input argument.
	echo "${str}"
}

echoinfo()
{
	if [ -z "${quiet_flag}" ]; then

		# Echo the argument string to stderr instead of stdout.
		echo "${output_name}: $1" 1>&2;
	fi
}

echoninfo()
{
	if [ -z "${quiet_flag}" ]; then

		# Echo the argument string to stderr instead of stdout.
		echo -n "${output_name}: $1" 1>&2;
	fi
}

echon2info()
{
	if [ -z "${quiet_flag}" ]; then

		# Echo the argument string to stderr instead of stdout.
		echo "$1" 1>&2;
	fi
}

find_header_dirs()
{
	local cur_dirpath sub_items result cur_list item child_list

	# Extract the argument: the current directory, and the list of
	# directories found so far that contain headers.
	cur_dirpath="$1"

	echoninfo "scanning contents of ${cur_dirpath}"

	# Acquire a list of the directory's contents.
	sub_items=$(ls ${cur_dirpath})

	# If there is at least one header present, add the current directory to
	# the list header of directories. Otherwise, the current directory does
	# not contribute to the list returned to the caller.
	result=$(echo ${sub_items} | grep "\.h")

	if [ -n "${result}" ]; then
		cur_list="${cur_dirpath}"
		echon2info " ...found headers"
	else
		cur_list=""
		echon2info ""
	fi

	# Iterate over the list of directory contents.
	for item in ${sub_items}; do

		# If the current item is a directory, recursively accumulate header
		# directories for that sub-directory.
		if [ -d "${cur_dirpath}/${item}" ]; then

			# Recursively find header directories within the sub-directory
			# ${item} and store the directory list to child_list.
			child_list=$(find_header_dirs "${cur_dirpath}/${item}")

			# Accumulate the sub-directory's header list with the running list
			# of header directories
			cur_list="${cur_list} ${child_list}"
		fi

	done

	# Return the list of header directories.
	echo "${cur_list}"
}

get_header_path()
{
	local filename dirpaths filepath

	filename="$1"
	dirpaths="$2"
	filepath=""

	# Search each directory path for the filename given.
	for dirpath in ${dirpaths}; do

		if [ -f "${dirpath}/${filename}" ]; then

			filepath="${dirpath}/${filename}"
			break
		fi
	done

	# Return the filepath that was found. Note that if no filepath was found
	# in the loop above, the empty string gets returned.
	echo "${filepath}"
}

replace_pass()
{
	local filename dirpaths result header headerlist

	filename="$1"
	dirpaths="$2"

	headerlist=""

	# This string is inserted after #include directives after having
	# determined that they are not present in the directory tree and should
	# be ignored when assessing whether there are still #include directives
	# that need to be expanded. Note that it is formatted as a comment and
	# thus will be ignored when the monolithic header is eventually read C
	# preprocessor and/or compiler.
	skipstr="\/\/skipped"

	# The way we (optionally) remove C-style comments results in a single
	# blank line in its place (regardless of how many lines the comment
	# spanned. When a comment is removed, it is replaced by this string
	# so that the line can be deleted with a subsequent sed command.
	commstr="DeLeTeDCsTyLeCoMmEnT"

	# Iterate through each line of the header file, accumulating the names of
	# header files referenced in #include directives.
	while read -r curline
	do

		# Check whether the line begins with a #include directive.
		result=$(echo ${curline} | grep '^[[:space:]]*#include ' )

		# If the #include directive was found...
		if [ -n "${result}" ]; then

			# Isolate the header filename.
			header=$(echo ${curline} | sed -e "s/#include [\"<]\([a-zA-Z0-9.]*\)[\">]/\1/g")

			# Add the header file to a list.
			headerlist=$(canonicalize_ws "${headerlist} ${header}")

		fi
	done < "${filename}"

	echoinfo "  found references to: ${headerlist}"

	# Initialize the return value to null.
	result=""

	# Iterate over each header file found in the previous loop.
	for header in ${headerlist}; do

		# Find the path to the header.
		header_filepath=$(get_header_path ${header} "${dirpaths}")

		# If the header file was not found, get_header_path() returns an
		# empty string. In this case, we assume the file is a system header
		# and thus we skip it since we don't want to inline the contents of
		# system headers anyway.
		if [ -z "${header_filepath}" ]; then

			echoinfo "  could not locate file '${header}'; marking to skip."

			# Insert a comment after the #include so we know to ignore it
			# later. Notice that we mimic the quotes or angle brackets
			# around the header name, whichever pair was used in the input.
			cat ${filename} \
			    | sed -e "s/^[[:space:]]*#include \([\"<]\)\(${header}\)\([\">]\)/#include \1\2\3 ${skipstr}/" \
			    > "${filename}.tmp"

			# Overwrite the original file with the updated copy.
			mv "${filename}.tmp" ${filename}

		else

			echoinfo "  located file '${header_filepath}'; inserting."

			# Strip C-style comments from the file, if requested.
			if [ -n "${strip_comments}" ]; then

				header_filename=${header_filepath##*/}

				# Make a temporary copy of ${header_filepath} stripped of its
				# C-style comments. This leaves behind a single blank line,
				# which is then deleted.
				cat ${header_filepath} \
				    | perl -0777 -pe "s/\/\*.*?\*\//${commstr}/gs" \
				    | sed -e "/${commstr}/d" \
				    > "${header_filename}.tmp"

				header_to_insert="${header_filename}.tmp"
			else
				header_to_insert="${header_filepath}"
			fi

			# Replace the #include directive for the current header file with the
			# contents of that header file, saving the result to a temporary file.
			# We also insert begin and end markers to allow for more readability.
			cat ${filename} \
			    | sed -e "/^[[:space:]]*#include \"${header}\"/ {" \
			          -e "i // begin ${header}" \
			          -e "r ${header_to_insert}" \
			          -e "a // end ${header}" \
			          -e "d" \
			          -e "}" \
			    > "${filename}.tmp"

			# Overwrite the original header file with the updated copy.
			mv "${filename}.tmp" ${filename}

			# If C-style comments were stripped, remove the temporary file.
			if [ -n "${strip_comments}" ]; then
				rm "${header_filename}.tmp"
			fi
		fi
	done

	# works, but leaves blank line:
	#cat "test.h" | sed -e "/^#include \"foo.h\"/r foo.h" -e "s///" > "test.new.h"
	# works:
	#cat "test.h" | sed -e '/^#include \"foo.h\"/ {' -e 'r foo.h' -e 'd' -e '}' > "test.new.h"
	# works:
	#cat "test.h" | sed -e '/^#include \"foo.h\"/r foo.h' -e '/^#include \"foo.h\"/d' > "test.new.h"
	#cat zorn/header.h | sed -e '/^#include \"header1.h\"/ {' -e 'i // begin insertion' -e 'r alice/header1.h' -e 'a // end insertion' -e 'd' -e '}'

	# Search the updated file for #include directives, but ignore any
	# hits that also contain the skip string (indicating that the header
	# file referenced by that #include could not be found).
	result=$(cat ${filename} | grep '^[[:space:]]*#include ' | grep -v "${skipstr}")

	# Return the result so the caller knows if we need to proceed with
	# another pass.
	echo ${result}
}

#
# -- main function -------------------------------------------------------------
#

main()
{
	# The name of the script, stripped of any preceeding path.
	script_name=${0##*/}

	# The script name to use in informational output. Defaults to ${script_name}.
	output_name=${script_name}

	# Whether or not we should suppress informational output. (Default is to
	# output messages.)
	quiet_flag=""

	# Whether or not we should strip C-style comments from the outout. (Default
	# is to not strip C-style comments.)
	strip_comments=""

	# Process our command line options.
	while getopts ":ho:qr" opt; do
	    case $opt in
	        o  ) output_name=$OPTARG ;;
	        q  ) quiet_flag="1" ;;
	        r  ) strip_comments="1" ;;
	        h  ) print_usage ;;
	        \? ) print_usage
	    esac
	done
	shift $(($OPTIND - 1))

	# Print usage if we don't have exactly two arguments.
	if [ $# != "3" ]; then

		print_usage
	fi

	# Acquire the two required arguments:
	# - the input header file,
	# - the output header file,
	# - the root directory to search when attempting to locate header files
	#   referenced in #include directives.
	inputfile="$1"
	outputfile="$2"
	rootdir="$3"

	# Starting with the root search directory, recursively search for all
	# directory paths that contains a header file.
	dirpaths=$(find_header_dirs ${rootdir})
	dirpaths=$(canonicalize_ws "${dirpaths}")

	echoinfo "scan summary:"
	echoinfo "  headers found in:"
	echoinfo "  ${dirpaths}"

	echoinfo "preparing to monolithify '${inputfile}'."

	# Make a copy of the inputfile.
	cp ${inputfile} ${outputfile}
	
	echoinfo "new header will be saved to '${outputfile}'."

	done_flag="0"
	while [ ${done_flag} == "0" ]; do

		echoinfo "starting new pass."

		# Perform a replacement pass. The return string is non-null if
		# additional passes are necessary and null otherwise.
		result=$(replace_pass ${outputfile} "${dirpaths}")

		if [ -n "${result}" ]; then

			echoinfo "pass finished; result: additional pass(es) needed."
		else
			echoinfo "pass finished; result: no further passes needed."
		fi

		# If the return value was null, then we're done.
		if [ -z "${result}" ]; then
			done_flag="1"
		fi
	done

	echoinfo "substitution complete."
	echoinfo "monolithic header saved as '${outputfile}'."

	# Exit peacefully.
	return 0
}


# The script's main entry point, passing all parameters given.
main "$@"

