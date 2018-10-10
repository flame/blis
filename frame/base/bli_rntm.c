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

// -----------------------------------------------------------------------------

void bli_rntm_set_ways_for_op
     (
       opid_t  l3_op,
       side_t  side,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     )
{
	// Set the number of ways for each loop, if needed, depending on what
	// kind of information is already stored in the rntm_t object.
	bli_rntm_set_ways_from_rntm( m, n, k, rntm );

#if 0
printf( "bli_rntm_set_ways_for_op()\n" );
bli_rntm_print( rntm );
#endif

	// Now modify the number of ways, if necessary, based on the operation.
	if ( l3_op == BLIS_TRMM ||
	     l3_op == BLIS_TRSM )
	{
		dim_t jc = bli_rntm_jc_ways( rntm );
		dim_t pc = bli_rntm_pc_ways( rntm );
		dim_t ic = bli_rntm_ic_ways( rntm );
		dim_t jr = bli_rntm_jr_ways( rntm );
		dim_t ir = bli_rntm_ir_ways( rntm );

		// Notice that, if we do need to update the ways, we don't need to
		// update the num_threads field since we only reshuffle where the
		// parallelism is extracted, not the total amount of parallelism.

		if ( l3_op == BLIS_TRMM )
		{
			// We reconfigure the parallelism extracted from trmm_r due to a
			// dependency in the jc loop. (NOTE: This dependency does not exist
			// for trmm3.)
			if ( bli_is_left( side ) )
			{
				bli_rntm_set_ways_only
				(
				  jc,
				  pc,
				  ic,
				  jr,
				  ir,
				  rntm
				);
			}
			else // if ( bli_is_right( side ) )
			{
				bli_rntm_set_ways_only
				(
				  1,
				  pc,
				  ic,
				  jr * jc,
				  ir,
				  rntm
				);
			}
		}
		else if ( l3_op == BLIS_TRSM )
		{
			// For trsm_l, we extract all parallelism from the jc and jr loops.
			// For trsm_r, we extract all parallelism from the ic loop.
			if ( bli_is_left( side ) )
			{
				bli_rntm_set_ways_only
				(
				  jc,
				  1,
				  1,
				  ic * pc * jr * ir,
				  1,
				  rntm
				);
			}
			else // if ( bli_is_right( side ) )
			{
				bli_rntm_set_ways_only
				(
				  1,
				  1,
				  ic * pc * jc * ir * jr,
				  1,
				  1,
				  rntm
				);
			}
		}
	}
}

void bli_rntm_set_ways_from_rntm
     (
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     )
{
	dim_t nt = bli_rntm_num_threads( rntm );

	dim_t jc = bli_rntm_jc_ways( rntm );
	dim_t pc = bli_rntm_pc_ways( rntm );
	dim_t ic = bli_rntm_ic_ways( rntm );
	dim_t jr = bli_rntm_jr_ways( rntm );
	dim_t ir = bli_rntm_ir_ways( rntm );

#ifdef BLIS_ENABLE_MULTITHREADING

	bool_t nt_set   = FALSE;
	bool_t ways_set = FALSE;

	// If the rntm was fed in as a copy of the global runtime via
	// bli_thread_init_rntm(), we know that either the num_threads
	// field will be set and all of the ways unset, or vice versa.
	// However, we can't be sure that a user-provided rntm_t isn't
	// initialized uncleanly. So here we have to enforce some rules
	// to get the rntm_t into a predictable state.

	// First, we establish whether or not the number of threads is set.
	if ( nt > 0 ) nt_set = TRUE;

	// Next, we establish whether or not any of the ways of parallelism
	// for each loop were set. If any of the ways are set (positive), we
	// then we assume the user wanted to use those positive values and
	// default the non-positive values to 1.
	if ( jc > 0 || pc > 0 || ic > 0 || jr > 0 || ir > 0 )
	{
		ways_set = TRUE;

		if ( jc < 1 ) jc = 1;
		if ( pc < 1 ) pc = 1;
		if ( ic < 1 ) ic = 1;
		if ( jr < 1 ) jr = 1;
		if ( ir < 1 ) ir = 1;
	}

	// Now we use the values of nt_set and ways_set to determine how to
	// interpret the original values we found in the rntm_t object.

	if ( ways_set == TRUE )
	{
		// If the ways were set, then we use the values that were given
		// and interpreted above (we set any non-positive value to 1).
		// The only thing left to do is calculate the correct number of
		// threads.

		nt = jc * pc * ic * jr * ir;
	}
	else if ( ways_set == FALSE && nt_set == TRUE )
	{
		// If the ways were not set but the number of threas was set, then
		// we attempt to automatically generate a thread factorization that
		// will work given the problem size. Thus, here we only set the
		// ways and leave the number of threads unchanged.

		pc = 1;

		bli_partition_2x2( nt, m*BLIS_THREAD_RATIO_M,
		                       n*BLIS_THREAD_RATIO_N, &ic, &jc );

		for ( ir = BLIS_THREAD_MAX_IR ; ir > 1 ; ir-- )
		{
			if ( ic % ir == 0 ) { ic /= ir; break; }
		}

		for ( jr = BLIS_THREAD_MAX_JR ; jr > 1 ; jr-- )
		{
			if ( jc % jr == 0 ) { jc /= jr; break; }
		}
	}
	else // if ( ways_set == FALSE && nt_set == FALSE )
	{
		// If neither the ways nor the number of threads were set, then
		// the rntm was not meaningfully changed since initialization,
		// and thus we'll default to single-threaded execution.

		nt = 1;
		jc = pc = ic = jr = ir = 1;
	}

#else

	// When multithreading is disabled, always set the rntm_t ways
	// values to 1.
	nt = 1;
	jc = pc = ic = jr = ir = 1;

#endif

	// Save the results back in the runtime object.
	bli_rntm_set_num_threads_only( nt, rntm );
	bli_rntm_set_ways_only( jc, pc, ic, jr, ir, rntm );
}

void bli_rntm_print
     (
       rntm_t* rntm
     )
{
	dim_t nt = bli_rntm_num_threads( rntm );

	dim_t jc = bli_rntm_jc_ways( rntm );
	dim_t pc = bli_rntm_pc_ways( rntm );
	dim_t ic = bli_rntm_ic_ways( rntm );
	dim_t jr = bli_rntm_jr_ways( rntm );
	dim_t ir = bli_rntm_ir_ways( rntm );

	printf( "rntm contents    nt  jc  pc  ic  jr  ir\n" );
	printf( "               %4d%4d%4d%4d%4d%4d\n", (int)nt, (int)jc, (int)pc,
	                                               (int)ic, (int)jr, (int)ir );
}

