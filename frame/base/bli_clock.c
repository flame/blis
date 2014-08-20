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

static double gtod_ref_time_sec = 0.0;

double bli_clock( void )
{
	return bli_clock_helper();
}

double bli_clock_min_diff( double time_min, double time_start )
{
	double time_min_prev;
	double time_diff;

	// Save the old value.
	time_min_prev = time_min;

	time_diff = bli_clock() - time_start;

	time_min = bli_fmin( time_min, time_diff );

	// Assume that anything:
	// - under or equal to zero,
	// - over an hour, or
	// - under a nanosecond
	// is actually garbled due to the clocks being taken too closely together.
	if      ( time_min <= 0.0    ) time_min = time_min_prev;
	else if ( time_min >  3600.0 ) time_min = time_min_prev;
	else if ( time_min <  1.0e-9 ) time_min = time_min_prev;

	return time_min;
}


// --- Begin Linux build definitions -------------------------------------------
#ifndef BLIS_ENABLE_WINDOWS_BUILD

double bli_clock_helper()
{
	double         the_time, norm_sec;
	struct timeval tv;

	gettimeofday( &tv, NULL );

	if ( gtod_ref_time_sec == 0.0 )
		gtod_ref_time_sec = ( double ) tv.tv_sec;

	norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

	the_time = norm_sec + tv.tv_usec * 1.0e-6;

	return the_time;
}

// --- End Linux build definitions ---------------------------------------------
#else
// --- Begin Windows build definitions -----------------------------------------

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>

double bli_clock_helper()
{
	LARGE_INTEGER clock_freq = {0};
	LARGE_INTEGER clock_val;
	BOOL          r_val;

	r_val = QueryPerformanceFrequency( &clock_freq );

	if ( r_val == 0 )
	{
		bli_print_msg( "QueryPerformanceFrequency() failed", __FILE__, __LINE__ );
		bli_abort();
	}

	r_val = QueryPerformanceCounter( &clock_val );

	if ( r_val == 0 )
	{
		bli_print_msg( "QueryPerformanceCounter() failed", __FILE__, __LINE__ );
		bli_abort();
	}

	return ( ( double) clock_val.QuadPart / ( double) clock_freq.QuadPart );
}

#endif
// --- End Windows build definitions -------------------------------------------

