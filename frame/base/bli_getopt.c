/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"


char       *bli_optarg;
int         bli_optind = 1;

int         bli_opterr = 0;
int         bli_optopt;

static char OPT_MARKER = '-';


int bli_getopt( int argc, char** const argv, const char* optstring )
{
	static char* nextchar = NULL;

	char*        elem_str;
	char*        optstr_char;

	// If argv contains no more arguments to process, return.
	if ( bli_optind == argc ) return -1;

	// Get a pointer to the current argv element string to process. If
	// nextchar is non-NULL, then it means the previous call processed
	// an element of argv with more than one option character, in which
	// case we need to pick up where we left off (which is the address
	// contained in nextchar).
	if ( nextchar == NULL )
	{
		elem_str = argv[ bli_optind ];

		// elem_str[0] should be an OPT_MARKER if it is an option. In the
		// event that it is not an option, argv should be permuted so that
		// the non-option argument moves back toward the end of the list.
		// This functionality is not supported/implemented here. Therefore,
		// we require all of the program's option arguments to precede all of
		// its non-option arguments.
		if ( elem_str[0] != OPT_MARKER )
		{
			bli_optarg = NULL;
			//bli_optind += 1;
			return -1;
		}

		// Skip over the OPT_MARKER.
		elem_str++;
	}
	else
	{
		// Note we don't need to skip the OPT_MARKER here since we are
		// continuing processing of a string with more than one option
		// character.

		// Use the nextchar pointer as our element string.
		elem_str = nextchar;

		// Reset nextchar to NULL.
		nextchar = NULL;
	}

	// Find the first occurrence of elem_str[0] in optstring.
	optstr_char = strchr( optstring, elem_str[0] );

	// If the option character in elem_str[0] is absent from the option
	// string, store it and return '?'.
	if ( optstr_char == NULL )
	{
		if ( bli_opterr == 1 ) fprintf( stderr, "bli_getopt(): **error**: option character '%c' missing from option string \"%s\"\n", elem_str[0], optstring );

		// We can't dereference optstr_char since it is NULL, so we use
		// elem_str[0] instead.
		bli_optopt = elem_str[0];
		bli_optind += 1;
		return '?';
	}

	// We can now safely assume that an option characer was found in the
	// option string. Now we need to check if the option takes an argument.
	if ( optstr_char[1] == ':' )
	{
		// If the current element string ends after the option character,
		// then the companion argument must be stored in the next element
		// of argv. Otherwise, the argument begins immediately after the
		// option character.
		if ( elem_str[1] == '\0' )
		{
			// If there are no more elements in argv, the argument was
			// omitted. Store the corresponding option character and
			// return '?'.
			if ( bli_optind + 1 >= argc )
			{
				if ( bli_opterr == 1 ) fprintf( stderr, "bli_getopt(): **error**: option character '%c' is missing an argument (end of argv)\n", elem_str[0] );

				bli_optopt = *optstr_char;
				bli_optind += 1;
				return '?';
			}
			// If there are still more elements in argv yet to process AND
			// the next one is an option, then the argument was omitted.
			else if ( argv[ bli_optind + 1 ][0] == OPT_MARKER )
			{
				if ( bli_opterr == 1 ) fprintf( stderr, "bli_getopt(): **error**: option character '%c' is missing an argument (next element of argv is option '%c')\n", elem_str[0], argv[ bli_optind + 1 ][1] );

				bli_optopt = *optstr_char;
				bli_optind += 1;
				return '?';
			}

			// If no error was deteced above, we can safely assign optarg
			// to be the next element in argv and increment optind by two.
			bli_optarg = argv[ bli_optind + 1 ];
			bli_optind += 2;
		}
		else
		{
			// We don't need to check for missing arguments since we know
			// that because the char after the option character is not NULL,
			// the character(s) after it must constitute the argument.

			bli_optarg = &elem_str[1];
			bli_optind += 1;
		}

		return *optstr_char;
	}

	// The current option character does NOT take an argument. However, we
	// still need to check if the next char is an option argument (such as
	// occurs when the user runs "program -rv" instead of "program -r -v").
	if ( elem_str[1] != '\0' )
	{
		if ( strchr( optstring, elem_str[1] ) != NULL )
		{
			nextchar = &elem_str[1];
			return *optstr_char;
		}
	}

	bli_optarg = NULL;
	bli_optind += 1;
	return *optstr_char;
}

