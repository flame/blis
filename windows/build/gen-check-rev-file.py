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

# Global variables for command line options, with default settings.
script_name  = ""
verbose_flag = False

# Global constants
toplevel_dirpath  = "."
svn_dirname       = ".svn"
entries_filename  = "entries"
revision_filename = "revision"
dummy_rev_string  = "unknown"


# ------------------------------------------------------------------------------

def print_usage():
	
	# Print help information.
	print " "
	print " %s" % script_name
	print " "
	print " Field G. Van Zee"
	print " "
	print " This script ensures that a revision file exists so nmake can include the"
	print " revision number in the subdirectory paths to the build products."
	print " "
	print " If a .svn directory exists, the revision file is created (or updated)"
	print " to contain the revision number contained in .svn\entries file."
	print " Otherwise, if a .svn directory does not exist, the revision file is"
	print " left untouched if it exists, and created with a dummy value if it does"
	print " not."
	print " "
	print " This script is typically invoked by configure.cmd, but it can also be"
	print " run manually."
	print " "
	print " Usage:"
	print "   %s" % script_name
	print " "
	print " The following options are accepted:"
	print " "
	print "   -v          verbose"
	print "                 Be verbose. Output what's happening."
	print " "

	# Exit the script.
	sys.exit()

# ------------------------------------------------------------------------------

def main():

	# Extern our global veriables.
	global script_name
	global verbose_flag

	# Get the script name so we can use it in our output.
	( script_dir, script_name ) = os.path.split( sys.argv[0] )
	
	try:
		
		# Get the command line options.
		options, args = getopt.getopt( sys.argv[1:], "v")
	
	except getopt.GetoptError, err:
	
		# print help information and exit:
		print str( err ) # will print something like "option -a not recognized"
		print_usage()
	
	# Parse our expected command line options.
	for o, a in options:
		
		if o == "-v":
			verbose_flag = True
		else:
			assert False, "unhandled option"
	
	# Check the number of arguments after command line option processing.
	n_args = len( args )
	if n_args != 0:
		print_usage() 

	# Construct the filepaths to the entries and revision files.
	entries_filepath  = os.path.join( toplevel_dirpath, svn_dirname, entries_filename )
	revision_filepath = os.path.join( toplevel_dirpath, revision_filename )

	# Test for the existence of the entries file (and by proxy, a working copy).
	entries_file_exists = file_exists( entries_filepath )

	# If the entries file exists, we are in a working copy, and thus we can
	# overwrite the revision file with a potentially new value.
	if entries_file_exists == True:

		# Read the revision number from the entries file.
		rev_num_str = read_revision_from_entries( entries_filepath )

		# Be verbose if verbosity was requested.
		if verbose_flag == True:
			print "%s: Found working copy; writing revision string \"%s\" to %s" % ( script_name, rev_num_str, revision_filepath )
			
		# Write the revision number to the revision file.
		write_revision_to_file( rev_num_str, revision_filepath )

	# If we can't find the entries file, we probably are in an exported
	# copy: either an official snapshot, or a copy that someone exported
	# manually--hopefully (and likely) the former.
	else:

		# Be verbose if verbosity was requested.
		if verbose_flag == True:
			print "%s: Found export. Checking for revision file..." % ( script_name )
		
		# Test for the existence of the revision file.
		rev_file_exists = file_exists( revision_filepath )

		# If the revision file does not exist, create a dummy file so the
		# configure script has something to work with.
		if rev_file_exists == False:

			# Be verbose if verbosity was requested.
			if verbose_flag == True:
				print "%s: Revision file not found. Writing dummy revision string \"%s\" to %s" % ( script_name, dummy_rev_string, revision_filepath )
			
			# Write the dummy string to the revision file.
			write_revision_to_file( dummy_rev_string, revision_filepath )

		else:

			# Get the revision number from the file just for the purposes of
			# being verbose, if it was requested.
			rev_num_str = read_revision_file( revision_filepath )

			# Be verbose if verbosity was requested.
			if verbose_flag == True:
				print "%s: Revision file found containing revision string \"%s\". Export is valid snapshot!" % ( script_name, rev_num_str )


# ------------------------------------------------------------------------------

def file_exists( filepath ):

	# Try to open the file read-only.
	try:
		
		fp = open( filepath, 'r' )
		fp.close()
		exists = True
	
	except IOError, err:
		
		exists = False
	
	return exists


# ------------------------------------------------------------------------------

def read_revision_from_entries( entries_filepath ):

	# Open the ignore list files as read-only.
	entries_file = open( entries_filepath, 'r' )

	# Read all lines in the entries file.
	raw_list     = entries_file.readlines()

	# Close the file.
	entries_file.close()

	# Grab the fourth line, which is where the revision number lives, and strip
	# it of whitespace (probably just a newline).
	rev_num_str = raw_list[3].strip()

	# Return the revision number string.
	return rev_num_str

# ------------------------------------------------------------------------------

def write_revision_to_file( rev_string, revision_filepath ):

	# Open the revision file for writing.
	revision_file = open( revision_filepath, 'w' )

	# Write the revision string to the file.
	revision_file.write( rev_string )

	# Close the file.
	revision_file.close()

# ------------------------------------------------------------------------------

def read_revision_file( revision_filepath ):

	# Open the revision file.
	revision_file = open( revision_filepath, 'r' )

	# Read the first (and only) line.
	line = revision_file.readline()

	# Close the file.
	revision_file.close()

	# Grab the string and strip the it of whitespace (should just be a newline).
	rev_num_str = line.strip()

	# Return the revision number string.
	return rev_num_str

# ------------------------------------------------------------------------------

# Begin by executing main().
main()
