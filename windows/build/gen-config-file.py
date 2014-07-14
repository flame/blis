#! /usr/bin/env python
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

# ------------------------------------------------------------------------------

# Import modules
import sys
import os
import os.path
import getopt
import re
import string

# Global variables for command line options, with default settings.
script_name  = ""
dry_run_flag = False
verbose_flag = False

# Global constants
config_dirname     = "config"
source_dirname     = "frame"
object_dirname     = "obj"
object_extension   = ".obj"
leaf_list_path     = "build/leaf_list"
revision_filename  = "revision"
rev_varname        = "REVISION"
pwd_varname        = "PWD"
arch_varname       = "ARCH_STR"
build_varname      = "BUILD_STR"
ccompiler_varname  = "CCOMPILER_STR"


# ------------------------------------------------------------------------------

def print_usage():
	
	# Print help information.
	print " "
	print " %s" % script_name
	print " "
	print " Field G. Van Zee"
	print " "
	print " Create a config.mk file that is to be included by the nmake Makefile."
	print " This config.mk file is based on a template, but also includes variable"
	print " definitions that are needed for the specific build were are performing."
	print " The variables which are currently appended to config.mk at runtime are:"
	print "   - the revision string"
	print "   - the path to the current working directory"
	print "   - the build string (e.g. debug, release)"
	print "   - the architecture string (e.g. x86, x64)"
	print "   - the C compiler to use (e.g. icl, cl)"
	print "   - a list of paths to the object files to be compiled"
	print " The config.mk file is placed within the config subdirectory." 
	print " "
	print " Usage:"
	print "   %s [options] flat_dir arch build ccompiler path\\to\\config.mk.in" % script_name
	print " "
	print " The following options are accepted:"
	print " "
	print "   -d          dry-run"
	print "                 Go through all the motions, but don't actually output"
	print "                 the nmake definition file."
	print "   -v          verbose"
	print "                 Be verbose about actions (one line of output her action)."
	print " "

	# Exit the script.
	sys.exit()

# ------------------------------------------------------------------------------

def main():

	# Extern our global veriables.	
	global script_name
	global dry_run_flag
	global verbose_flag

	# Get the script name so we can use it in our output.
	( script_dir, script_name ) = os.path.split( sys.argv[0] )
	
	try:
		
		# Get the command line options.
		options, args = getopt.getopt( sys.argv[1:], "dv")
	
	except getopt.GetoptError, err:
	
		# print help information and exit:
		print str( err ) # will print something like "option -a not recognized"
		print_usage()
	
	# Parse our expected command line options.
	for o, a in options:
		
		if o == "-d":
			dry_run_flag = True
		elif o == "-v":
			verbose_flag = True
		else:
			assert False, "unhandled option"
	
	# Check the number of arguments after command line option processing.
	n_args = len( args )
	if n_args != 5:
		print_usage() 

	# Acquire the non-optional arguments.
	flat_dir         = args[0]
	arch_string      = args[1]
	build_string     = args[2]
	ccompiler_string = args[3]
	input_filepath   = args[4]

	# Acquire the list of leaf-type directories we will descend into.
	leaf_list = read_leaf_list()

	# Read the contents of the template file.
	template_file_line_list = read_template_file( input_filepath )

	# Initialize a new list for the lines to be output
	output_file_line_list = template_file_line_list

	# Read the revision number from the revision file.
	rev_num_str = read_revision_file( revision_filename )

	# Add a variable for the revision number of the code we're working with.
	rev_var_value = rev_varname + " = " + rev_num_str + "\n"
	output_file_line_list.append( rev_var_value )
	
	# Add a variable for the path to the current working directory and append
	# it to our list.
	pwd_var_value = pwd_varname + " = " + os.getcwd() + "\n"
	output_file_line_list.append( pwd_var_value )
	
	# Add a variable for the architecture string and append it to our list.
	arch_var_value = arch_varname + " = " + arch_string + "\n"
	output_file_line_list.append( arch_var_value )
	
	# Add a variable for the build type string and append it to our list.
	build_var_value = build_varname + " = " + build_string + "\n"
	output_file_line_list.append( build_var_value )
	
	# Add a variable for the C compiler string and append it to our list.
	ccompiler_var_value = ccompiler_varname + " = " + ccompiler_string + "\n"
	output_file_line_list.append( ccompiler_var_value )
	
	# Walk the flat subdirectories for each of the leaves.
	for leaf_spec in leaf_list:
		
		# Unpack the leaf_spec tuple.
		src_exts, hdr_exts = leaf_spec

		# Create the paths to the source and object subdirectories.
		src_dirpath = os.path.join( flat_dir, source_dirname )
		obj_dirpath = os.path.join( flat_dir, object_dirname, arch_string, build_string )

		# Get a list of files from the leaf subdirectory.
		src_filenames = os.listdir( src_dirpath )
		
		# This will be the nmake variable name to which we will assign the list
		# of source files.
		nmake_varname = "BLIS_OBJS"
		
		# Generate the line to output.
		leaf_line = generate_object_list( nmake_varname, src_filenames, src_exts, obj_dirpath )

		# Accumulate the lines.
		output_file_line_list.append( leaf_line )
	
	# Get the filename part of the input filepath.
	input_filedir, input_filename = os.path.split( input_filepath )

	# Remove the .in extension in the output filename.
	output_filename = re.sub( '.mk.in', '.mk', input_filename )
	
	# Construct the filepath for the output file.
	output_filepath = os.path.join( flat_dir, config_dirname, output_filename )

	# Write the output lines.
	write_output_file( output_filepath, output_file_line_list )

# ------------------------------------------------------------------------------

def read_revision_file( filepath ):

	# Try to open the revision file.
	try:
		
		revision_file = open( filepath, 'r' )
	
	except IOError, err:
		
		print "%s: Couldn't open revision file %s" % ( script_name, filepath )
		sys.exit(1)

	# Read the first (and only) line.
	line = revision_file.readline()

	# Close the file.
	revision_file.close()

	# Grab the string and strip the it of whitespace (should just be a newline).
	rev_num_str = line.strip()

	# Return the revision number string.
	return rev_num_str

# ------------------------------------------------------------------------------

def generate_object_list( nmake_varname, src_filenames, src_exts, obj_dirpath ):

	# Initialize the string as an assignment operation.
	the_line = nmake_varname + " = "
	
	# Return early if there are no source extensions for this leaf spec.
	if src_exts == []:
		return ""

	# Construct a pattern to match any file ending with any of the source file
	# extensions given. This string is going to look something like ".[cf]".
	src_pattern = '\.['
	for src_ext in src_exts:
		src_pattern = src_pattern + src_ext
	src_pattern = src_pattern + ']'

	# Consider all source files.
	for src_filename in src_filenames:
		
		obj_filename = re.sub( src_pattern, '.obj', src_filename )
		
		# Create the full path to the file.
		obj_filepath = os.path.join( obj_dirpath, obj_filename )
		
		# Be verbose if verbosity was requested.
		if verbose_flag == True:
			print "%s: adding file %s" % ( script_name, obj_filepath )
				
		# And then add it to the list.
		the_line = the_line + obj_filepath + " "

	# Be verbose if verbosity was requested.
	if verbose_flag == True:
		print "%s: %s" % ( script_name, the_line )
	
	# Append a newline to the end of the line, for file.writelines().
	the_line = the_line + "\n"

	# Return the new line.
	return the_line

# ------------------------------------------------------------------------------

def read_template_file( template_filepath ):
	
	# Open the template file as read-only.
	template_file = open( template_filepath, 'r' )

	# Read all lines in the template file.
	template_file_lines = template_file.readlines()

	# Close the file.
	template_file.close()

	# Return the list of lines in the template file.
	return template_file_lines

# ------------------------------------------------------------------------------

def write_output_file( output_filepath, output_lines ):

	# Take action only if this is not a dry run.
	if dry_run_flag == False:

		# Open the template file as writable.
		output_file = open( output_filepath, 'w' )

		# Write the lines.
		output_file.writelines( output_lines )

		# Close the file.
		output_file.close()

# ------------------------------------------------------------------------------

def read_leaf_list():

	# Open the leaf list file.
	leaf_file = open( leaf_list_path, 'r' )

	# Read the lines in the file.
	line_list = leaf_file.readlines()

	# Start with a blank list.
	leaf_list = []

	# Iterate over the lines.
	for line in line_list:

		# Split the specification by colon to separate the fields.
		fields = string.split( string.strip( line ), ':' )

		# Get the individual fields of the specification.
		src_exts = string.split( fields[0], ',' )
		hdr_exts = string.split( fields[1], ',' )
		
		# If it's a singleton list of an empty string, make it an empty list.
		if len(src_exts) == 1:
			if src_exts[0] == '':
				src_exts = []
		
		# If it's a singleton list of an empty string, make it an empty list.
		if len(hdr_exts) == 1:
			if hdr_exts[0] == '':
				hdr_exts = []

		# Pack the fields into a tuple.
		leaf_spec = ( src_exts, hdr_exts )

		# Append the tuple to our list.
		leaf_list.append( leaf_spec )

	# Return the list.
	return leaf_list

# ------------------------------------------------------------------------------

# Begin by executing main().
main()
