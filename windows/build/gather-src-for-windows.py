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
import shutil
import string

# Global variables for command line options, with default settings.
script_name  = ""
dry_run_flag = False
verbose_flag = False

# Global constants
flat_config_dirname  = "config"
flat_header_dirname  = "include"
flat_source_dirname  = "frame"
leaf_list_path       = "build/leaf_list"
ignore_list_path     = "build/ignore_list"
ignore_list_win_path = "build/ignore_list.windows"

# ------------------------------------------------------------------------------

def print_usage():
	
	# Print help information.
	print " "
	print " %s" % script_name
	print " "
	print " Field G. Van Zee"
	print " "
	print " Walk the BLIS source tree and copy all sources necessary for"
	print " building BLIS under Windows into a single flat directory with"
	print " no subdirectory hierarchy."
	print " "
	print " Usage:"
	print "   %s [options] tree_dir flat_dir" % script_name
	print " "
	print " The following options are accepted:"
	print " "
	print "   -d          dry-run"
	print "                 Go through all the motions, but don't actually copy any"
	print "                 files."
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
	print 'checking options'
	for o, a in options:
		
		if o == "-d":
			print 'found dry run'
			dry_run_flag = True
		elif o == "-v":
			verbose_flag = True
		else:
			assert False, "unhandled option"
	
	# Check the number of arguments after command line option processing.
	n_args = len( args )
	if n_args != 2:
		print_usage() 

	# Acquire the non-optional arguments.
	tree_dir = args[0]
	flat_dir = args[1]

	# Acquire the list of directories we will ignore.
	ignore_list = read_ignore_list()

	# Acquire the list of leaf-type directories we will descend into.
	leaf_list = read_leaf_list()

	# Create strings for each of the base subdirectories in the flat
	# destination directory.
	flat_config_base_dirpath = os.path.join( flat_dir, flat_config_dirname )
	flat_header_base_dirpath = os.path.join( flat_dir, flat_header_dirname )
	flat_source_base_dirpath = os.path.join( flat_dir, flat_source_dirname )

	# Start a list of directories to create.
	dirs_to_create = []

	# Append the config directory. We do this outside of the for loop because
	# we don't need subdirectories for each leaf type.
	dirs_to_create.append( flat_config_base_dirpath )

	# For each of the leaf specifications, make the full pathnames of the
	# subdirectories that will reside within the root destination directory.
	for leaf_spec in leaf_list:
		
		# Unpack the leaf_spec tuple.
		src_exts, hdr_exts = leaf_spec
		
		# Append the directory path name to our list. 
		dirs_to_create.append( flat_header_base_dirpath )
		dirs_to_create.append( flat_source_base_dirpath )

	# Iterate over the directory list we just created.
	for dirpath in dirs_to_create:

		# Make the subdirectories within the root destination directory, but
		# only if they are not existing directories.
		if os.path.isdir( dirpath ) == False:
	
			# Take action only if this is not a dry run.
			if dry_run_flag == False:
	
				# Be verbose if verbosity was requested.
				if verbose_flag == True:
					print "%s: creating directory %s" % ( script_name, dirpath )
			
				# Make the directory, and parent directories, for dirpath.
				os.makedirs( dirpath )
	
			else:
	
				# Be verbose if verbosity was requested.
				if verbose_flag == True:
					print "%s: (dry-run) creating directory %s" % ( script_name, dirpath )


	# Walk the directory structure top-down.
	for dirpath, dirnames, filenames in os.walk( tree_dir ):
		
		# Remove directories that appear in the ignore list.
		for item in ignore_list:
			if item in dirnames:
				dirnames.remove( item )

		# Consider each leaf specification. If we find the name in the directory
		# path, then copy the files with its designated extensions into the flat
		# source directory.
		for leaf_spec in leaf_list:

			# Unpack the leaf_spec tuple.
			src_exts, hdr_exts = leaf_spec

			# At this point following line can probably be removed
			type_dir_name = os.sep + ''

			flat_source_leaf_dirpath = flat_source_base_dirpath
			flat_header_leaf_dirpath = flat_header_base_dirpath

			if dirpath.find( type_dir_name ) != -1:
				copy_files_to_flat_subdirs( dirpath, filenames, src_exts, hdr_exts,
				                            flat_source_leaf_dirpath,
				                            flat_header_leaf_dirpath )

# ------------------------------------------------------------------------------

def copy_files_to_flat_subdirs( dirpath, filenames, src_exts, hdr_exts, src_dirpath, hdr_dirpath ):

	# Consider all files in dirpath.
	for filename in filenames:
		
		# Construct the full file path for the current file.
		filepath = os.path.join( dirpath, filename )

		# Iterate over the valid source extensions for the current directory
		# path.
		for src_ext in src_exts:

			# If the filename/filepath ends with the source extension, copy it
			# to the source subdirectory within the flat destination directory.
			if filepath.endswith( src_ext ):
				
				# Take action only if this is not a dry run.
				if dry_run_flag == False:
	
					# Be verbose if verbosity was requested.
					if verbose_flag == True:
						print "%s: copying to %s from %s" % ( script_name, src_dirpath, filepath )
				
					# Copy the source file to the source subdirectory.
					shutil.copy2( filepath, src_dirpath )
	
				else:
	
					# Be verbose if verbosity was requested.
					if verbose_flag == True:
						print "%s: (dry-run) copying to %s from %s" % ( script_name, src_dirpath, filepath )
	
		# Iterate over the valid header extensions for the current directory
		# path.
		for hdr_ext in hdr_exts:

			# If the filename/filepath ends with the header extension, copy it
			# to the include subdirectory within the flat destination directory.
			if filepath.endswith( hdr_ext ):
	
				# Take action only if this is not a dry run.
				if dry_run_flag == False:
	
					# Be verbose if verbosity was requested.
					if verbose_flag == True:
						print "%s: copying to %s from %s" % ( script_name, hdr_dirpath, filepath )
				
					# Copy the header file to the header subdirectory.
					shutil.copy2( filepath, hdr_dirpath )
	
				else:

					# Be verbose if verbosity was requested.
					if verbose_flag == True:
						print "%s: (dry-run) copying to %s from %s" % ( script_name, hdr_dirpath, filepath )

# ------------------------------------------------------------------------------

def read_ignore_list():

	# Open the ignore list files as read-only.
	ignore_file     = open( ignore_list_path, 'r' )
	ignore_file_win = open( ignore_list_win_path, 'r' )

	# Read all lines in the ignore list files. The items in these lists contain
	# newlines, which we'll strip out shortly.
	raw_list     = ignore_file.readlines()
	raw_win_list = ignore_file_win.readlines()

	# Close the files.
	ignore_file.close()
	ignore_file_win.close()

	# Initialize an empty ignore list for the stripped version of the raw list.
	ignore_list = []

	# Iterate over the first raw list.
	for line in raw_list:
		
		# Append the stripped line to a new list.
		ignore_list.append( line.strip() )

	# Iterate over the second raw list.
	for line in raw_win_list:
		
		# Append the stripped line to a new list.
		ignore_list.append( line.strip() )

	# Return the list of stripped lines.
	return ignore_list

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
