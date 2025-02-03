/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021 - 2025, Advanced Micro Devices, Inc. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

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

// The global rntm_t structure, which holds the global thread settings
// along with a few other key parameters.
rntm_t global_rntm = BLIS_RNTM_INITIALIZER;

// Make thread settings local to each thread calling BLIS routines
BLIS_THREAD_LOCAL rntm_t tl_rntm = BLIS_RNTM_INITIALIZER;

// A mutex to allow synchronous access to global_rntm.
bli_pthread_mutex_t global_rntm_mutex = BLIS_PTHREAD_MUTEX_INITIALIZER;

// ----------------------------------------------------------------------------

void bli_rntm_init_from_global( rntm_t* rntm )
{
	// We must ensure that global_rntm and tl_rntm have been initialized
	bli_init_once();

	// Initialize supplied rntm from tl_rntm.
	*rntm = tl_rntm;

#ifdef BLIS_ENABLE_MULTITHREADING
	// Now update threading info to account for current OpenMP
	// number of threads and active levels.
	bli_thread_update_rntm_from_env( rntm );
#endif

#if 0
	printf( "bli_rntm_init_from_global()\n" );
	bli_rntm_print( rntm );
#endif
}

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

#ifdef PRINT_THREADING
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
//printf( "bli_rntm_set_ways_for_op(): jc%d ic%d jr%d\n", (int)jc, (int)ic, (int)jr );
			if ( bli_is_left( side ) )
			{
				bli_rntm_set_ways_only
				(
				  jc,
				  1,
				  ic * pc,
				  jr * ir,
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

	bool  auto_factor = FALSE;

#ifdef BLIS_ENABLE_MULTITHREADING

	bool  nt_set   = FALSE;
	bool  ways_set = FALSE;

	// If the rntm was fed in as a copy of the global runtime via
	// bli_rntm_init_from_global(), we know that either:
	// - the num_threads field is -1 and all of the ways are -1;
	// - the num_threads field is -1 and all of the ways are set;
	// - the num_threads field is set and all of the ways are -1.
	// However, we can't be sure that a user-provided rntm_t isn't
	// initialized uncleanly. So here we have to enforce some rules
	// to get the rntm_t into a predictable state.

	// First, we establish whether or not the number of threads is set.
	if ( nt > 0 ) nt_set = TRUE;

	// Take this opportunity to set the auto_factor field (when using
	// more than one thread).
	if ( nt_set && nt > 1 ) auto_factor = TRUE;

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

		// auto factorization is to be disabled if BLIS_IC_NT/BLIS_JC_NT env
		// variables are set irrespective of whether num_threads is modified
		// or not. This ensures that preset factorization is prioritized.
		auto_factor = FALSE;
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
		// If the ways were not set but the number of thread was set, then
		// we attempt to automatically generate a thread factorization that
		// will work given the problem size.

#ifdef BLIS_DISABLE_AUTO_PRIME_NUM_THREADS
		// If use of prime numbers is disallowed for automatic thread
		// factorizations, we first check if the number of threads requested
		// is prime. If it is prime, and it exceeds a minimum threshold, then
		// we reduce the number of threads by one so that the number is not
		// prime. This will allow for automatic thread factorizations to span
		// two dimensions (loops), which tends to be more efficient.
		if ( bli_is_prime( nt ) && BLIS_NT_MAX_PRIME < nt ) nt -= 1;
#endif

		pc = 1;

		bli_thread_partition_2x2( nt, m*BLIS_THREAD_RATIO_M,
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
	bli_rntm_set_auto_factor_only( auto_factor, rntm );
	bli_rntm_set_num_threads_only( nt, rntm );
	bli_rntm_set_ways_only( jc, pc, ic, jr, ir, rntm );

#ifdef PRINT_THREADING
	printf( "bli_rntm_set_ways_from_rntm()\n" );
	bli_rntm_print( rntm );
#endif
}

void bli_rntm_set_ways_from_rntm_sup
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

	bool  auto_factor = FALSE;

#ifdef BLIS_ENABLE_MULTITHREADING

	bool  nt_set   = FALSE;
	bool  ways_set = FALSE;

	// If the rntm was fed in as a copy of the global runtime via
	// bli_rntm_init_from_global(), we know that either:
	// - the num_threads field is -1 and all of the ways are -1;
	// - the num_threads field is -1 and all of the ways are set;
	// - the num_threads field is set and all of the ways are -1.
	// However, we can't be sure that a user-provided rntm_t isn't
	// initialized uncleanly. So here we have to enforce some rules
	// to get the rntm_t into a predictable state.

	// First, we establish whether or not the number of threads is set.
	if ( nt > 0 ) nt_set = TRUE;

	// Take this opportunity to set the auto_factor field (when using
	// more than one thread).
	if ( nt_set && nt > 1 ) auto_factor = TRUE;

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

		// auto factorization is to be disabled if BLIS_IC_NT/BLIS_JC_NT env
		// variables are set irrespective of whether num_threads is modified
		// or not. This ensures that preset factorization is prioritized.
		auto_factor = FALSE;
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
		// If the ways were not set but the number of thread was set, then
		// we attempt to automatically generate a thread factorization that
		// will work given the problem size.

#ifdef BLIS_DISABLE_AUTO_PRIME_NUM_THREADS
		// If use of prime numbers is disallowed for automatic thread
		// factorizations, we first check if the number of threads requested
		// is prime. If it is prime, and it exceeds a minimum threshold, then
		// we reduce the number of threads by one so that the number is not
		// prime. This will allow for automatic thread factorizations to span
		// two dimensions (loops), which tends to be more efficient.
		if ( bli_is_prime( nt ) && BLIS_NT_MAX_PRIME < nt ) nt -= 1;
#endif

		pc = 1;

		//bli_thread_partition_2x2( nt, m*BLIS_THREAD_SUP_RATIO_M,
		//							  n*BLIS_THREAD_SUP_RATIO_N, &ic, &jc );
		bli_thread_partition_2x2( nt, m,
								  n, &ic, &jc );

//printf( "bli_rntm_set_ways_from_rntm_sup(): jc = %d  ic = %d\n", (int)jc, (int)ic );
#if 0
		for ( ir = BLIS_THREAD_SUP_MAX_IR ; ir > 1 ; ir-- )
		{
			if ( ic % ir == 0 ) { ic /= ir; break; }
		}

		for ( jr = BLIS_THREAD_SUP_MAX_JR ; jr > 1 ; jr-- )
		{
			if ( jc % jr == 0 ) { jc /= jr; break; }
		}
#else
		ir = 1;
		jr = 1;

#endif
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
	bli_rntm_set_auto_factor_only( auto_factor, rntm );
	bli_rntm_set_num_threads_only( nt, rntm );
	bli_rntm_set_ways_only( jc, pc, ic, jr, ir, rntm );

#ifdef PRINT_THREADING
	printf( "bli_rntm_set_ways_from_rntm_sup()\n" );
	bli_rntm_print( rntm );
#endif
}

void bli_rntm_print
	 (
	   rntm_t* rntm
	 )
{
	dim_t af = bli_rntm_auto_factor( rntm );

	dim_t nt = bli_rntm_num_threads( rntm );

	bool mt = bli_rntm_blis_mt( rntm );

	dim_t jc = bli_rntm_jc_ways( rntm );
	dim_t pc = bli_rntm_pc_ways( rntm );
	dim_t ic = bli_rntm_ic_ways( rntm );
	dim_t jr = bli_rntm_jr_ways( rntm );
	dim_t ir = bli_rntm_ir_ways( rntm );

	printf( "rntm contents	       |   nt  jc  pc  ic  jr  ir\n" );
	printf( "autofac, blis_mt? %1d, %1d | %4d%4d%4d%4d%4d%4d\n", (int)af, (int)mt,
							   (int)nt, (int)jc, (int)pc,
							   (int)ic, (int)jr, (int)ir );
}

// -----------------------------------------------------------------------------

dim_t bli_rntm_calc_num_threads_in
	 (
	   bszid_t* restrict bszid_cur,
	   rntm_t*  restrict rntm
	 )
{
	/*									 // bp algorithm:
	   bszid_t bszids[7] = { BLIS_NC,	  // level 0: 5th loop
				 BLIS_KC,	  // level 1: 4th loop
				 BLIS_NO_PART, // level 2: pack B
				 BLIS_MC,	  // level 3: 3rd loop
				 BLIS_NO_PART, // level 4: pack A
				 BLIS_NR,	  // level 5: 2nd loop
				 BLIS_MR,	  // level 6: 1st loop
				 BLIS_KR	   // level 7: ukr loop

				 ...		   // pb algorithm:
				 BLIS_NR,	  // level 5: 2nd loop
				 BLIS_MR,	  // level 6: 1st loop
				 BLIS_KR	   // level 7: ukr loop
				   }; */
	dim_t n_threads_in = 1;

	// Starting with the current element of the bszids array (pointed
	// to by bszid_cur), multiply all of the corresponding ways of
	// parallelism.
	for ( ; *bszid_cur != BLIS_KR; bszid_cur++ )
	{
		const bszid_t bszid = *bszid_cur;

		//if ( bszid == BLIS_KR ) break;

		// We assume bszid is in {NC,KC,MC,NR,MR,KR} if it is not
		// BLIS_NO_PART.
		if ( bszid != BLIS_NO_PART )
		{
			const dim_t cur_way = bli_rntm_ways_for( bszid, rntm );

			n_threads_in *= cur_way;
		}
	}

	return n_threads_in;
}

#if 0
	for ( ; *bszid_cur != BLIS_KR; bszid_cur++ )
	{
		const bszid_t bszid = *bszid_cur;
		dim_t		cur_way = 1;

		// We assume bszid is in {NC,KC,MC,NR,MR,KR} if it is not
		// BLIS_NO_PART.
		if ( bszid != BLIS_NO_PART )
			cur_way = bli_rntm_ways_for( bszid, rntm );
		else
			cur_way = 1;

		n_threads_in *= cur_way;
	}
#endif


// Calculates the optimum number of threads using m, n, k dimensions.
// This function modifies only the local copy of rntm with optimum threads.
// tl_rntm will remain unchanged. As a result, num_threads set by
// application is available in tl_rntm data structure.

void bli_nthreads_optimum(
				   obj_t*  a,
				   obj_t*  b,
				   obj_t*  c,
				   opid_t  family,
				   rntm_t* rntm
				 )
{
#ifndef BLIS_ENABLE_MULTITHREADING
	return;
#endif
#ifndef AOCL_DYNAMIC
	return;
#endif

	dim_t n_threads = bli_rntm_num_threads(rntm);

	if(( n_threads == -1) || (n_threads == 1)) return;

	dim_t n_threads_ideal = n_threads;

	if( family == BLIS_GEMM && bli_obj_is_double(c))
	{
		dim_t m = bli_obj_length(c);
		dim_t n = bli_obj_width(c);
		dim_t k = bli_obj_width_after_trans(a);


		// Query the architecture ID
		arch_t id = bli_arch_query_id();
		if(id == BLIS_ARCH_ZEN5 || id == BLIS_ARCH_ZEN4)
		{
			if(n < m)
			{
				if(k <= 32)
				{
					if( m <= 1000 )
					{
						n_threads_ideal = 8;
					}
					else if( m <= 10000)
					{
						if( n <= 500 )
						{
							n_threads_ideal = 16;
						}
						else if( n <= 1000 )
						{
							n_threads_ideal = 64;
						}
						else
						{
							n_threads_ideal = 96;
						}
					}
					else
					{
						n_threads_ideal = 96;
					}
				}
				else if(k <= 64)
				{
					if( (m <= 100) || (m <= 500 && n <= 100))
					{
						n_threads_ideal = 8;
					}
					else if(m <= 500)
					{
						n_threads_ideal = 16;
					}
					else if(m <= 1000)
					{
						if(n <= 50)
						{
							n_threads_ideal = 8;
						}
						else if(n <= 250)
						{
							n_threads_ideal = 16;
						}
						else
						{
							n_threads_ideal = 24;
						}
					}
					else if(m <= 10000)
					{
						if(n <= 500)
						{
							n_threads_ideal = 24;
						}
						else if(n <= 1000)
						{
							n_threads_ideal = 64;
						}
					}
					else if( m <= 20000 && n <= 500)
					{
						n_threads_ideal = 96;
					}
					else if( m <= 30000)
					{
						if(n <= 1000)
						{
							n_threads_ideal = 144;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( m <= 40000 && n <= 1000)
					{
						n_threads_ideal = 168;
					}
					else
					{
						n_threads_ideal = 192;
					}
				}
				else if(k <= 128)
				{
					if( (m <= 100) || (m <= 500 && n <= 50))
					{
						n_threads_ideal = 8;
					}
					else if(m <= 500)
					{
						if(n <= 100)
						{
							n_threads_ideal = 16;
						}
						else
						{
							n_threads_ideal = 24;
						}
					}
					else if( m <= 1000 )
					{
						if(n <= 200)
						{
							n_threads_ideal = 24;
						}
						else
						{
							n_threads_ideal = 48;
						}
					}
					else if( m <= 10000 )
					{
						if(n <= 50)
						{
							n_threads_ideal = 32;
						}
						else if(n <= 500)
						{
							n_threads_ideal = 48;
						}
						else if(n <= 750)
						{
							n_threads_ideal = 96;
						}
						else if(n <= 1000)
						{
							n_threads_ideal = 128;
						}
						else if(n <= 5000)
						{
							n_threads_ideal = 144;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( m <= 30000 )
					{
						if(n <= 1000)
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( m <= 40000 )
					{
						if(n <= 600)
						{
							n_threads_ideal = 144;
						}
						else if(n <= 1000)
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else
					{
						n_threads_ideal = 192;
					}
				}
				else
				{
					if( m <= 100 )
					{
						n_threads_ideal = 8;
					}
					else if( m <= 500 )
					{
						if( n <= 50 )
						{
							n_threads_ideal = 16;
						}
						else if( n <= 200 )
						{
							n_threads_ideal = 32;
						}
						else
						{
							n_threads_ideal = 48;
						}
					}
					else if( m <= 1000 )
					{
						if(n <= 100 )
						{
							n_threads_ideal = 32;
						}
						else
						{
							n_threads_ideal = 48;
						}
					}
					else if( m <= 10000 )
					{
						if(n <= 200 )
						{
							n_threads_ideal = 48;
						}
						else if( n <= 500 )
						{
							n_threads_ideal = 96;
						}
						else if( n <= 600 )
						{
							n_threads_ideal = 144;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( m <= 20000 && n <= 750 )
					{
						n_threads_ideal = 168;
					}
					else
					{
						n_threads_ideal = 192;
					}
				}
			}
			else if(m < n)
			{
				if(k <= 32)
				{
					if( n <= 1000 )
					{
						n_threads_ideal = 8;
					}
					else if( n <= 10000 )
					{
						if( m <= 500 )
						{
							n_threads_ideal = 16;
						}
						else if( m <= 1000 )
						{
							n_threads_ideal = 32;
						}
						else
						{
							n_threads_ideal = 96;
						}
					}
					else
					{
						n_threads_ideal = 96;
					}
				}
				else if(k <= 64)
				{
					if( (n <= 100) || (n <= 500 && m <= 100) )
					{
						n_threads_ideal = 8;
					}
					else if(n <= 500)
					{
						n_threads_ideal = 16;
					}
					else if( n <= 1000 )
					{
						if( m <= 200)
						{
							n_threads_ideal = 16;
						}
						else
						{
							n_threads_ideal = 32;
						}
					}
					else if( n <= 10000 )
					{
						if( m <= 100)
						{
							n_threads_ideal = 32;
						}
						else if( m <= 500)
						{
							n_threads_ideal = 48;
						}
						else if( m <= 1000)
						{
							n_threads_ideal = 96;
						}
						else if(m <= 2500)
						{
							n_threads_ideal = 128;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( n <= 20000 )
					{
						if( m < 1000 )
						{
							n_threads_ideal = 128;
						}
						else if( m < 2500 )
						{
							n_threads_ideal = 144;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( n <= 30000)
					{
						if( m < 1000 )
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( n <= 40000 )
					{
						if(m < 600)
						{
							n_threads_ideal = 144;
						}
						else if(m < 750)
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else
					{
						n_threads_ideal = 192;
					}
				}
				else if(k <= 128)
				{
					if( (n <= 100) || (n <= 500 && m <= 50) )
					{
						n_threads_ideal = 8;
					}
					else if(n <= 500 )
					{
						if( m <= 100)
						{
							n_threads_ideal = 16;
						}
						else
						{
							n_threads_ideal = 32;
						}
					}
					else if( n <= 1000 )
					{
						if( m <= 50)
						{
							n_threads_ideal = 16;
						}
						else
						{
							n_threads_ideal = 32;
						}
					}
					else if( n <= 10000 )
					{
						if(m <= 100 )
						{
							n_threads_ideal = 32;
						}
						else if(m <= 200 )
						{
							n_threads_ideal = 64;
						}
						else if(m <= 500 )
						{
							n_threads_ideal = 72;
						}
						else if(m < 1000 )
						{
							n_threads_ideal = 96;
						}
						else if(m < 2500 )
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( n <= 20000 )
					{
						if(m <= 500 )
						{
							n_threads_ideal = 96;
						}
						else if(m < 1000 )
						{
							n_threads_ideal = 128;
						}
						else if(m < 2500 )
						{
							n_threads_ideal = 144;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( n <= 30000 )
					{
						if(m <= 500 )
						{
							n_threads_ideal = 96;
						}
						else if(m < 750 )
						{
							n_threads_ideal = 128;
						}
						else if(m < 1000 )
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else if( n <= 40000 )
					{
						if(m < 500 )
						{
							n_threads_ideal = 128;
						}
						else if(m < 600 )
						{
							n_threads_ideal = 144;
						}
						else if(m < 750 )
						{
							n_threads_ideal = 168;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else
					{
						n_threads_ideal = 192;
					}
				}
				else
				{
					if(n <= 100)
					{
						n_threads_ideal = 8;
					}
					else if( n <= 500 )
					{
						if( m <= 100)
						{
							n_threads_ideal = 16;
						}
						else
						{
							n_threads_ideal = 32;
						}
					}
					else if( n <= 1000 )
					{
						if( m <= 100)
						{
							n_threads_ideal = 32;
						}
						else
						{
							n_threads_ideal = 48;
						}
					}
					else if( n <= 10000 )
					{
						if( m <= 50)
						{
							n_threads_ideal = 48;
						}
						else if(m <= 100)
						{
							n_threads_ideal = 64;
						}
						else if(m < 750)
						{
							n_threads_ideal = 96;
						}
						else
						{
							n_threads_ideal = 192;
						}
					}
					else
					{
						n_threads_ideal = 192;
					}
				}
			}
			else if(m == n)
			{
				if(k <= 32)
				{
					if( m <= 20 )        n_threads_ideal = 1;
					else if( m <= 40 )   n_threads_ideal = 4;
					else if( m <= 800 )  n_threads_ideal = 8;
					else if( m <= 1000 ) n_threads_ideal = 16;
					else if( m <= 5000 ) n_threads_ideal = 64;
					else                 n_threads_ideal = 96;
				}
				else if(k <= 64)
				{
					if(m <= 150) n_threads_ideal = 8;
					else if(m <= 1000) n_threads_ideal = 16;
					else if( m <= 2500) n_threads_ideal = 96;
					else if( m <= 5000) n_threads_ideal = 128;
					else if( m <= 6000) n_threads_ideal = 128;
					else n_threads_ideal = 192;
				}
				else if( k <= 128)
				{
					if( m <= 100) n_threads_ideal = 8;
					else if(m <= 500) n_threads_ideal = 32;
					else if( m <= 1000) n_threads_ideal = 64;
					else if( m <= 5000) n_threads_ideal = 144;
					else n_threads_ideal = 192;
				}
				else
				{
					if( m <= 100) n_threads_ideal = 8;
					else if( m <= 250 ) n_threads_ideal = 32;
					else if( m <= 500 ) n_threads_ideal = 48;
					else if( m <= 1000) n_threads_ideal = 96;
					else n_threads_ideal = 192;
				}
			}
		}
		else // Not BLIS_ARCH_ZEN5 or BLIS_ARCH_ZEN4
		{
			if( k >= 128)
			{
				if(n <= 15)
				{
					if(m < 128) 	 n_threads_ideal = 8;
					else if(m < 256) n_threads_ideal = 16;
					else if(m < 512) n_threads_ideal = 32;
					else 			 n_threads_ideal = 64;
				}
				else if (n <= 64)
				{
					if(m < 128) 	 n_threads_ideal = 16;
					else if(m < 256) n_threads_ideal = 32;
					else 			 n_threads_ideal = 64;
				}
				else
				{
					if(m < 256) n_threads_ideal = 32;
					else 		n_threads_ideal = 64;
				}
			}
			else
			{
				if(m > 10000)
				{
					// current logic is only limiting threads to
					// less or equal to 64 - limits performance.
					// To deal with larger matrix sizes we need to use
					// large number of threads to improve performance
					// Need to derive this upperTH - and
					// if matrix -sizes are larger and user wants
					// to use higher number of threads - that should be allowed.

					// if (n > UpperTH) n_threads_ideal = n_threads;
					if (n > 200 )	    n_threads_ideal = 64;
					else if ( n > 120 ) n_threads_ideal = 32;
					else if ( n > 40  ) n_threads_ideal = 16;
					else if ( n > 10  ) n_threads_ideal = 8;
					else 				n_threads_ideal = 4;
				}
				else if( m > 1000)
				{
					if (n <= 10) 		  n_threads_ideal = 4;
					else if ( n <= 512 )  n_threads_ideal = 8;
					else if ( n <= 1024 ) n_threads_ideal = 16;
					else if ( n <= 2048 ) n_threads_ideal = 32;
					else 				  n_threads_ideal = 64;
				}
				else if(m > 210)
				{
					if(n < 10)  	   n_threads_ideal = 4;
					else if(n <= 512)  n_threads_ideal = 8;
					else if(n <= 1024) n_threads_ideal = 16;
					else if(n <= 2048) n_threads_ideal = 32;
					else 			   n_threads_ideal = 64;
				}
				else if(m > 150)
				{
					if(n < 10)  	   n_threads_ideal = 2;
					else if(n <= 512)  n_threads_ideal = 8;
					else if(n <= 1024) n_threads_ideal = 16;
					else if(n <= 2048) n_threads_ideal = 32;
					else 			   n_threads_ideal = 64;
				}
				else if( ( m < 34) && (k < 68) && ( n < 34))
				{
					n_threads_ideal = 1;
				}
				else
				{	//(m<150 && k<128)
					if(n < 20) n_threads_ideal = 1;
					if(n < 64) n_threads_ideal = 4;
					else	   n_threads_ideal = 8;
				}
			}
		}
	}
	else if( family == BLIS_GEMM && bli_obj_is_dcomplex(c))
	{
		dim_t m = bli_obj_length(c);
		dim_t n = bli_obj_width(c);
		dim_t k = bli_obj_width_after_trans(a);

		if((m<=128 || n<=128 || k<=128) && ((m+n+k) <= 400))
		{
			n_threads_ideal = 8;
		}
		else if((m<=256 || n<=256 || k<=256) && ((m+n+k) <= 800))
		{
			n_threads_ideal = 16;
		}
		if((m<=48) || (n<=48) || (k<=48))
		{
			if((m+n+k) <= 840)
			{
				n_threads_ideal = 8;
			}
			else if((m+n+k) <= 1240)
			{
				n_threads_ideal = 16;
			}
			else if((m+n+k) <= 1540)
			{
				n_threads_ideal = 32;
			}
		}
	}
	else if( family == BLIS_SYRK && bli_obj_is_double(c))
	{
		dim_t n = bli_obj_length(c);
		dim_t k = bli_obj_width_after_trans(a);

		if( (( n <= 10) && ( k < 700))  ||
			(( n <= 20) && ( k <= 190)) ||
			(( n <= 40) && ( k <= 80))  ||
			(( n <= 50) && ( k <= 40))  ||
			(( n <= 60) && ( k <= 20))
		)
			n_threads_ideal = 1;
		else
			n_threads_ideal = n_threads;
	}
	else if( family == BLIS_TRSM && bli_obj_is_double(c) )
	{
		dim_t m = bli_obj_length(c);
		dim_t n = bli_obj_width(c);

		arch_t id = bli_arch_query_id();
		if (id == BLIS_ARCH_ZEN5)
		{
			if ( (m < 58 && n < 138) || (m < 1020 && n < 12))
				n_threads_ideal = 1;
			else if ((m < 137) && (n < 137))
				n_threads_ideal = 8;
			else if ((m < 324))
				n_threads_ideal = 32;
			else if ((n < 1020) || (n < 4294 && m < 1360))
				n_threads_ideal = 64;
			else
				n_threads_ideal = n_threads;
		}
		else
		{
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
			if ( (m <= 300) && (n <= 300) )
				n_threads_ideal = 8;
			else if ( (m <= 400) && (n <= 400) )
				n_threads_ideal = 16;
		  	else if ( (m <= 900) && (n <= 900) )
				n_threads_ideal = 32;
#else
			if ( (m <= 512) && (n <= 512) )
				n_threads_ideal = 4;
#endif
		}
	}
	else if( family == BLIS_TRSM && bli_obj_is_dcomplex(c))
	{
		dim_t m = bli_obj_length(c);
		dim_t n = bli_obj_width(c);
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
		if ( (m <= 300) && (n <= 300) )
			n_threads_ideal = 8;
		else if ( (m <= 400) && (n <= 400) )
			n_threads_ideal = 16;
		else if ( (m <= 900) && (n <= 900) )
			n_threads_ideal = 32;
#else
		if((m>=64) && (m<=256) && (n>=64) && (n<=256))
		{
			n_threads_ideal = 8;
		}
#endif
	}
	else if( family == BLIS_GEMMT && ( bli_obj_is_double(c) || bli_obj_is_dcomplex(c) ) )
	{
		dim_t n = bli_obj_length(c);
		dim_t k = bli_obj_width_after_trans(a);

		if ( n < 8 )
		{
			if ( k <= 512)
			{
				n_threads_ideal = 1;
			}
			else if ( k <= 1024 )
			{
				n_threads_ideal = 4;
			}
		}
		else if ( n < 32 )
		{
			if ( k < 128 )
			{
				n_threads_ideal = 1;
			}
			else if ( k <= 512 )
			{
				n_threads_ideal = 4;
			}
			else if ( k <= 1024 )
			{
				n_threads_ideal = 6;
			}
			else if ( k <= 1600 )
			{
				n_threads_ideal = 10;
			}
		}
		else if ( n <= 40 )
		{
			if ( k < 32 )
			{
				n_threads_ideal = 2;
			}
			else if ( k < 128 )
			{
				n_threads_ideal = 4;
			}
			else if ( k <= 256 )
			{
				n_threads_ideal = 8;
			}
		}
		else if ( n < 115 )
		{
			if ( k < 128 )
			{
				n_threads_ideal = 6;
			}
			else if ( k <= 216 )
			{
				n_threads_ideal = 8;
			}
		}
		else if ( n <= 160 )
		{
			if ( k <= 132 )
			{
				n_threads_ideal = 8;
			}
		}
		else if ( n < 176 )
		{
			if ( k < 128 )
			{
				n_threads_ideal = 8;
			}
			else if ( k <= 512 )
			{
				n_threads_ideal = 14;
			}
		}
		else if ( n <= 220 )
		{
			if ( k < 128 )
			{
				n_threads_ideal = 8;
			}
		}
	}
	else if( family == BLIS_TRMM && bli_obj_is_double(c))
	{
		dim_t m = bli_obj_length(c);
		dim_t n = bli_obj_width(c);

		if(( n <= 32) && (m <= 32))
		{
			n_threads_ideal=1;
		/*If Side is Left*/
		}else
		{
			//Left Side
			if(bli_obj_is_triangular(a))
			{
				if((m < 300))
				{
					if (n < 1000)
					{
						n_threads_ideal=8;
					}else if (n < 2000)
					{
						n_threads_ideal=16;
					}else if (n < 3000)
					{
						n_threads_ideal=32;
					}else
					{
						n_threads_ideal=64;
					}
				}else if(m < 600)
				{
					if (n < 2000)
					{
						n_threads_ideal=16;
					}else if (n < 3000)
					{
						n_threads_ideal=32;
					}else
					{
						n_threads_ideal=64;
					}
				}else
				{
					if(n < 1000)
					{
						n_threads_ideal=32;
					}else
					{
						n_threads_ideal=64;
					}
				}
			}else//Right Side
			{
				if((n < 300))
				{
					if (m < 1000)
					{
						n_threads_ideal=8;
					}else if (m < 2000)
					{
						n_threads_ideal=16;
					}else if (m < 3000)
					{
						n_threads_ideal=32;
					}else
					{
						n_threads_ideal=64;
					}
				}else if(n < 600)
				{
					if (m < 2000)
					{
						n_threads_ideal=16;
					}else if (m < 3000)
					{
						n_threads_ideal=32;
					}else
					{
						n_threads_ideal=64;
					}
				}else
				{
					if(m < 1000)
					{
						n_threads_ideal=32;
					}else
					{
						n_threads_ideal=64;
					}
				}
			}
		}
	}

	dim_t n_threads_opt = bli_min(n_threads, n_threads_ideal);

	// This modifies only local rntm - therefore doesn't require mutex locks
	// for updating rntm
	bli_rntm_set_num_threads_only( n_threads_opt, rntm );

#ifdef PRINT_THREADING
	printf( "bli_nthreads_optimum()\n" );
	bli_rntm_print( rntm );
#endif

	return;
}

#ifdef AOCL_DYNAMIC
// Calculates the optimum number of threads along with the factorization
// (ic, jc) using m, n, k dimensions. This function modifies only the local
// copy of rntm with optimum threads. Since tl_rntm remains unchanged the
// num_threads set by application is available in tl_rntm data structure.
err_t bli_smart_threading_sup
				(
				 obj_t*  a,
				 obj_t*  b,
				 obj_t*  c,
				 opid_t  family,
				 rntm_t* rntm,
				 cntx_t* cntx
				)
{
	// By default smart threading should be disabled.
	err_t ret_val = BLIS_FAILURE;

#ifndef BLIS_ENABLE_MULTITHREADING
	return ret_val;
#endif

	dim_t n_threads = bli_rntm_num_threads( rntm );

	// For non-openmp based threading, n_threads could be -1.
	if ( ( n_threads == -1 ) || ( n_threads == 1 ) ) return ret_val;

	dim_t ic_way = bli_rntm_ic_ways( rntm );
	dim_t jc_way = bli_rntm_jc_ways( rntm );

	// Dont enable smart threading if the user supplied the factorization.
	if( ( ic_way > 0 ) || ( jc_way > 0 ) ) return ret_val;

	// Only supporting sgemm for now.
	if ( ( family == BLIS_GEMM ) && bli_obj_is_float( c ) )
	{
		dim_t k = bli_obj_width_after_trans(a);
		dim_t m = 0;
		dim_t n = 0;

		bool trans_A_for_kernel = FALSE;

		const stor3_t stor_id = bli_obj_stor3_from_strides( c, a, b );
		const bool is_rrr_rrc_rcr_crr = (
										  stor_id == BLIS_RRR ||
										  stor_id == BLIS_RRC ||
										  stor_id == BLIS_RCR ||
										  stor_id == BLIS_CRR
										);

		// The A and B matrices are swapped based on the storage type in
		// var1n2m. Need to account for this when determining ic and jc
		// based on m and n dimensions of A and B.
		if ( is_rrr_rrc_rcr_crr )
		{
			m = bli_obj_length( c );
			n = bli_obj_width( c );
			trans_A_for_kernel = bli_obj_has_trans( a );
		}
		else
		{
			m = bli_obj_width( c );
			n = bli_obj_length( c );
			trans_A_for_kernel = bli_obj_has_trans( b );
		}

		// Take default path if transpose is enabled for A matrix.
		if ( trans_A_for_kernel == FALSE )
		{
			// A successfull call to smart threading api implies smart
			// factorization and possibly native -> SUP path conversion.
			// Optimal thread selection is not supported yet.
			ret_val = bli_gemm_smart_threading_sup( bli_obj_dt( c ),
						bli_obj_elem_size( c ),
						is_rrr_rrc_rcr_crr, m, n, k, n_threads,
						cntx, rntm );
		}
	}
	return ret_val;
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 dscalv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
BLIS_INLINE void aocl_dscalv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{

	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:
			if ( n_elem <= 63894 )
				*nt_ideal = 1;
			else if ( n_elem <= 145165 )
				*nt_ideal = 4;
			else if ( n_elem <= 4487626 )
				*nt_ideal = 8;
			else if ( n_elem <= 5773817 )
				*nt_ideal = 32;
			else if ( n_elem <= 10000000 )
				*nt_ideal = 64;
			else
				*nt_ideal = 96;

			break;

		case BLIS_ARCH_ZEN4:
			if ( n_elem <= 27500 )
				*nt_ideal = 1;
			else if ( n_elem <= 100000 )
				*nt_ideal = 2;
			else if ( n_elem <= 145000 )
				*nt_ideal = 4;
			else if ( n_elem <= 4944375 )
				*nt_ideal = 8;
			else if( n_elem <= 10000000 )
				*nt_ideal = 32;
			else
				*nt_ideal = 64;

			break;

		case BLIS_ARCH_ZEN3:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN:
			if ( n_elem <= 30000)
				*nt_ideal = 1;
			else if (n_elem <= 100000)
				*nt_ideal = 2;
			else if (n_elem <= 500000)
				*nt_ideal = 8;
			else if (n_elem <= 2500000)
				*nt_ideal = 12;
			else if (n_elem <= 4000000)
				*nt_ideal = 16;
			else if(n_elem <= 7000000)
				*nt_ideal = 24;
			else if(n_elem <= 10000000)
				*nt_ideal = 32;
			else
				*nt_ideal = 64;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 zdscalv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
BLIS_INLINE void aocl_zdscalv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{

	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem <= 10000)
				*nt_ideal = 1;
			else if (n_elem <= 20000)
				*nt_ideal = 4;
			else if (n_elem <= 1000000)
				*nt_ideal = 8;
			else if (n_elem <= 2500000)
				*nt_ideal = 12;
			else if (n_elem <= 5000000)
				*nt_ideal = 32;
			else
				*nt_ideal = 64;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 daxpyv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
BLIS_INLINE void aocl_daxpyv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:

			if ( n_elem <= 34000 )
				*nt_ideal = 1;
			else if ( n_elem <= 82000 )
				*nt_ideal = 4;
			else if ( n_elem <= 2330000 )
				*nt_ideal = 8;
			else if ( n_elem <= 4250000 )
				*nt_ideal = 16;
			else if ( n_elem <= 7000000 )
				*nt_ideal = 32;
			else if ( n_elem <= 21300000 )
				*nt_ideal = 64;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		case BLIS_ARCH_ZEN4:

			if ( n_elem <= 11000 )
				*nt_ideal = 1;
			else if ( n_elem <= 130000 )
				*nt_ideal = 4;
			else if ( n_elem <= 2230000 )
				*nt_ideal = 8;
			else if ( n_elem <= 3400000 )
				*nt_ideal = 16;
			else if ( n_elem <= 9250000 )
				*nt_ideal = 32;
			else if ( n_elem <= 15800000 )
				*nt_ideal = 64;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem <= 4000 )
				*nt_ideal = 1;
			else if (n_elem <= 11000)
				*nt_ideal = 4;
			else if (n_elem <= 300000)
				*nt_ideal = 8;
			else if (n_elem <= 750000)
				*nt_ideal = 16;
			else if (n_elem <= 2600000)
				*nt_ideal = 32;
			else if (n_elem <= 4000000)
				*nt_ideal = 64;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 zaxpyv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
BLIS_INLINE void aocl_zaxpyv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:

			if ( n_elem <= 16000 )
				*nt_ideal = 1;
			else if (n_elem <= 43000)
				*nt_ideal = 4;
			else if (n_elem <= 2300000)
				*nt_ideal = 8;
			else if (n_elem <= 4000000)
				*nt_ideal = 32;
			else if (n_elem <= 6600000)
				*nt_ideal = 64;
			else if (n_elem <= 6600000)
				*nt_ideal = 96;
			else
				*nt_ideal = 128;
			break;

		case BLIS_ARCH_ZEN4:

			if ( n_elem <= 4600 )
				*nt_ideal = 1;
			else if (n_elem <= 6700)
				*nt_ideal = 2;
			else if (n_elem <= 61500)
				*nt_ideal = 4;
			else if (n_elem <= 1200000)
				*nt_ideal = 8;
			else if (n_elem <= 4000000)
				*nt_ideal = 32;
			else
				*nt_ideal = 96;
			break;

		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem <= 2600 )
				*nt_ideal = 1;
			else if( n_elem <= 11000)
				*nt_ideal = 2;
			else if (n_elem <= 33000)
				*nt_ideal = 4;
			else
				// Performance does not scale with number of threads beyond 8 threads
				*nt_ideal = 8;
			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 ddotv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
BLIS_INLINE void aocl_ddotv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem <= 2500 )
				*nt_ideal = 1;
			else if (n_elem <= 5000)
				*nt_ideal = 4;
			else if (n_elem <= 15000)
				*nt_ideal = 8;
			else if (n_elem <= 40000)
				*nt_ideal = 16;
			else if (n_elem <= 200000)
				*nt_ideal = 32;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

BLIS_INLINE void aocl_zdotv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:
			// @note: Further tuning can be done.
			if ( n_elem <= 2080 )
				*nt_ideal = 1;
			else if (n_elem <= 3328 )
				*nt_ideal = 4;
			else if (n_elem <= 98304)
				*nt_ideal = 8;
			else if (n_elem <= 262144)
				*nt_ideal = 32;
			else if (n_elem <= 524288)
				*nt_ideal = 64;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 dcopyv API based on the
	architecture ID, input type and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/

BLIS_INLINE void aocl_dcopyv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	// Pick the AOCL dynamic logic based on the
	// architecture ID

	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:

			if ( n_elem <= 39000 )
				*nt_ideal = 1;
			else if ( n_elem <= 46000 )
				*nt_ideal = 2;
			else if (n_elem <= 160000)
				*nt_ideal = 4;
			else
				*nt_ideal = 8;
				// dcopy does not scale with more than 8 threads
			break;

		case BLIS_ARCH_ZEN4:

			if ( n_elem <= 17000 )
				*nt_ideal = 1;
			else if (n_elem <= 62000)
				*nt_ideal = 2;
			else if (n_elem <= 96000)
				*nt_ideal = 4;
			else
				*nt_ideal = 8;
				// dcopy does not scale with more than 8 threads
			break;
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem <= 17000 )
				*nt_ideal = 1;
			else if (n_elem <= 52200)
				*nt_ideal = 4;
			else
				*nt_ideal = 8;
				// dcopy does not scale with more than 8 threads
			break;

		default:
			// Without this default condition, compiler will throw
			// a warning saying other conditions are not handled
			// For other architectures, AOCL dynamic does not make any change
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 zcopyv API based on the
	architecture ID, input type and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	The function has been made static to restrict its scope.

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/

BLIS_INLINE void aocl_zcopyv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	// Pick the AOCL dynamic logic based on the
	// architecture ID

	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem <= 4600 )
				*nt_ideal = 1;
			else if (n_elem <= 5100)
				*nt_ideal = 2;
			else if (n_elem <= 22000)
				*nt_ideal = 4;
			else if (n_elem <= 240000)
				*nt_ideal = 8;
			else if (n_elem <=380000)
				*nt_ideal = 16;
			else if (n_elem <= 1700000)
				*nt_ideal = 32;
			else if (n_elem <= 3700000)
				*nt_ideal = 64;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		default:
			// Without this default condition, compiler will throw
			// a warning saying other conditions are not handled

			// For other architectures, AOCL dynamic does not make any change
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 dnormfv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
void aocl_dnormfv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch ( arch_id )
	{
		case BLIS_ARCH_ZEN5:

			#ifdef __clang__
				// Threshold setting based on LLVM's OpenMP
				if ( n_elem < 6000 )
					*nt_ideal = 1;
				else if ( n_elem < 16900 )
					*nt_ideal = 4;
				else if ( n_elem < 126000 )
					*nt_ideal = 8;
				else if ( n_elem < 200000 )
					*nt_ideal = 16;
				else if ( n_elem < 250000 )
					*nt_ideal = 32;
				else if ( n_elem < 500000 )
					*nt_ideal = 64;
				else
					// For sizes in this range, AOCL dynamic does not make any change
					*nt_ideal = -1;
			#else
				// Threshold setting based on GNU's OpenMP
				if ( n_elem < 4500 )
					*nt_ideal = 1;
				else if ( n_elem < 15400 )
					*nt_ideal = 4;
				else if ( n_elem < 285000 )
					*nt_ideal = 8;
				else if ( n_elem < 604000 )
					*nt_ideal = 16;
				else if ( n_elem < 2780000 )
					*nt_ideal = 32;
				else if ( n_elem < 10500000 )
					*nt_ideal = 64;
				else
					// For sizes in this range, AOCL dynamic does not make any change
					*nt_ideal = -1;
			#endif

			break;

		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem < 4000 )
				*nt_ideal = 1;
			else if ( n_elem < 17000 )
				*nt_ideal = 4;
			else if ( n_elem < 136000 )
				*nt_ideal = 8;
			else if ( n_elem < 365000 )
				*nt_ideal = 16;
			else if ( n_elem < 2950000 )
				*nt_ideal = 32;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

/*
	Functionality:
	--------------
	This function decides the AOCL dynamic logic for L1 znormfv API based on the
	architecture ID and size of the input variable.

	Function signature
	-------------------

	This function takes the following input:

	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	Exception
	----------

	1. For non-Zen architectures, return -1. The expectation is that this is handled
	   in the higher layer
*/
void aocl_znormfv_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{
	/*
		Pick the AOCL dynamic logic based on the
		architecture ID
	*/
	switch ( arch_id )
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN3:

			if ( n_elem < 2000 )
				*nt_ideal = 1;
			else if ( n_elem < 6500 )
				*nt_ideal = 4;
			else if ( n_elem < 71000 )
				*nt_ideal = 8;
			else if ( n_elem < 200000 )
				*nt_ideal = 16;
			else if ( n_elem < 1530000 )
				*nt_ideal = 32;
			else
				// For sizes in this range, AOCL dynamic does not make any change
				*nt_ideal = -1;

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

static void aocl_daxpyf_dynamic
     (
       arch_t arch_id,
       dim_t  n_elem,
       dim_t* nt_ideal
     )
{

	// Pick the AOCL dynamic logic based on the
	// architecture ID

	switch (arch_id)
	{
		case BLIS_ARCH_ZEN5:
		case BLIS_ARCH_ZEN4:
		case BLIS_ARCH_ZEN3:
		case BLIS_ARCH_ZEN2:
		case BLIS_ARCH_ZEN:

			if ( n_elem <= 128 )
				*nt_ideal = 1;
			// these nt_ideal sizes are tuned for trsv only,
			// when axpyf kernels are enabled for gemv, these might need
			// to be re tuned

			// else if ( n_elem <= 224)
			// 	*nt_ideal = 2;
			// else if ( n_elem <= 860)
			// 	*nt_ideal = 4;
			else
				*nt_ideal = 8;
				// axpyf does not scale with more than 8 threads

			break;

		default:
			/*
				Without this default condition, compiler will throw
				a warning saying other conditions are not handled
			*/

			/*
				For other architectures, AOCL dynamic does not make any change
			*/
			*nt_ideal = -1;
	}
}

#endif // AOCL_DYNAMIC

/*
	Functionality:
	--------------

	This function does the following:
	1. Reads the number of threads requested by the user from the rntm variable
	2. Acts as the gateway to the AOCL dynamic logic if AOCL dynamic is enabled
	   and alters the count of the number of threads accordingly

	Function signature
	-------------------

	This function takes the following input:

	* 'ker_id' - ID of kernel invoking this function
	* 'datatype_a' - Datatype 1 of kernel
	* 'datatype_b' - Datatype 2 of kernel
	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	Exception
	----------

	None
*/
void bli_nthreads_l1
     (
       l1vkr_t  ker_id,
       num_t    data_type_a,
       num_t    data_type_b,
       arch_t   arch_id,
       dim_t    n_elem,
       dim_t*   nt_ideal
     )
{
#ifdef AOCL_DYNAMIC
	/*
		This code sections dispatches the AOCL dynamic logic kernel for
		L1 APIs based on the kernel ID and the data type.
	*/
	// Function pointer to AOCL Dynamic logic kernel
	void (*aocl_dynamic_func_l1)(arch_t, dim_t, dim_t* ) = NULL;

	// Pick the aocl dynamic thread decision kernel based on the kernel ID
	switch (ker_id)
	{
		case BLIS_SCALV_KER:

			/*
				When input data types do not match the call is from mixed precision
			*/
			if (data_type_a != data_type_b)
			{
				// Function for ZDSCALV
				aocl_dynamic_func_l1 = aocl_zdscalv_dynamic;
			}
			else
			{
				// Function for DSCALV
				aocl_dynamic_func_l1 = aocl_dscalv_dynamic;
			}

			break;

		case BLIS_AXPYV_KER:

			if ( data_type_a == BLIS_DOUBLE )
			{
				// Function for DAXPYV
				aocl_dynamic_func_l1 = aocl_daxpyv_dynamic;
			}
			else if ( data_type_a == BLIS_DCOMPLEX )
			{
				// Function for ZAXPYV
				aocl_dynamic_func_l1 = aocl_zaxpyv_dynamic;
			}
			break;

		case BLIS_DOTV_KER:

			if ( data_type_a == BLIS_DOUBLE )
			{
				// Function for DDOTV
				aocl_dynamic_func_l1 = aocl_ddotv_dynamic;
			}
			else if ( data_type_a == BLIS_DCOMPLEX )
			{
				// Function for ZDOTV
				aocl_dynamic_func_l1 = aocl_zdotv_dynamic;
			}

			break;

		case BLIS_COPYV_KER:

			if ( data_type_a == BLIS_DOUBLE)
			{
				// Function for DCOPYV
				aocl_dynamic_func_l1 = aocl_dcopyv_dynamic;
			}
			else if ( data_type_a == BLIS_DCOMPLEX )
			{
				// Function for ZCOPYV
				aocl_dynamic_func_l1 = aocl_zcopyv_dynamic;
			}
					break;

		default:
			/*
				For kernels that do no have AOCL dynamic logic,
				use the number of threads requested by the user.
			*/
			*nt_ideal = -1;
	}

	/*
		For APIs that do not have AOCL dynamic
		logic, aocl_dynamic_func_l1 will be NULL.
	*/
	if( aocl_dynamic_func_l1 != NULL)
	{
		// Call the AOCL dynamic logic kernel
		aocl_dynamic_func_l1
		(
			arch_id,
			n_elem,
			nt_ideal
		);

		if (*nt_ideal == 1)
		{
			// Return early when the number of threads is 1
			return;
		}
	}

#endif
	// Initialized to avoid compiler warning
	rntm_t rntm_local;

	// Initialize a local runtime with global settings.
	bli_rntm_init_from_global(&rntm_local);

	// Query the total number of threads from the rntm_t object.
	dim_t nt_rntm = bli_rntm_num_threads(&rntm_local);

	if (nt_rntm <= 0)
	{
		// nt is less than one if BLIS manual setting of parallelism
		// has been used. Parallelism here will be product of values.
		nt_rntm = bli_rntm_calc_num_threads(&rntm_local);
	}

#ifdef AOCL_DYNAMIC

	// Calculate the actual number of threads that will be spawned
	if (*nt_ideal != -1)
	{
		// The if block is executed for all Zen architectures
		*nt_ideal = bli_min(nt_rntm, *nt_ideal);
	}
	else
	{
		/*
			For non-Zen architectures and very large sizes,
			spawn the actual number of threads requested
		*/
		*nt_ideal = nt_rntm;
	}

	/*
	  When the number of element to be processed is less
	  than the number of threads spawn n_elem number of threads.
	*/
	if (n_elem < *nt_ideal)
	{
		*nt_ideal = n_elem;
	}
#else

	// Calculate the actual number of threads that will be spawned
	*nt_ideal = nt_rntm;

#endif
}

/*
	Functionality:
	--------------

	This function does the following:
	1. Reads the number of threads requested by the user from the rntm variable
	2. Acts as the gateway to the AOCL dynamic logic if AOCL dynamic is enabled
	   and alters the count of the number of threads accordingly

	Function signature
	-------------------

	This function takes the following input:

	* 'ker_id' - ID of kernel invoking this function
	* 'datatype_a' - Datatype 1 of kernel
	* 'datatype_b' - Datatype 2 of kernel
	* 'arch_id' - Architecture ID of the system (copy of BLIS global arch id)
	* 'n_elem' - Number of elements in the vector
	* 'nt_ideal' - Ideal number of threads

	Exception
	----------

	None
*/
void bli_nthreads_l1f
	 (
       l1fkr_t  ker_id,
       num_t    data_type_a,
       num_t    data_type_b,
       arch_t   arch_id,
       dim_t    n_elem,
       dim_t*   nt_ideal
	 )
{
#ifdef AOCL_DYNAMIC
	/*
		This code sections dispatches the AOCL dynamic logic kernel for
		L1 APIs based on the kernel ID and the data type.
	*/
	// Function pointer to AOCL Dynamic logic kernel
	void (*aocl_dynamic_func_l1f)(arch_t, dim_t, dim_t* ) = NULL;

	// Pick the aocl dynamic thread decision kernel based on the kernel ID
	switch (ker_id)
	{
		case BLIS_AXPYF_KER:

			if ( data_type_a == BLIS_DOUBLE )
			{
				// Function for DAXPYF
				aocl_dynamic_func_l1f = aocl_daxpyf_dynamic;
			}
			break;

		default:
			/*
				For kernels that do no have AOCL dynamic logic,
				use the number of threads requested by the user.
			*/
			*nt_ideal = -1;
	}

	/*
		For APIs that do not have AOCL dynamic
		logic, aocl_dynamic_func_l1f will be NULL.
	*/
	if( aocl_dynamic_func_l1f != NULL)
	{
		// Call the AOCL dynamic logic kernel
		aocl_dynamic_func_l1f
		(
			arch_id,
			n_elem,
			nt_ideal
		);

		if (*nt_ideal == 1)
		{
			// Return early when the number of threads is 1
			return;
		}
	}

#endif
	// Initialized to avoid compiler warning
	rntm_t rntm_local;

	// Initialize a local runtime with global settings.
	bli_rntm_init_from_global(&rntm_local);

	// Query the total number of threads from the rntm_t object.
	dim_t nt_rntm = bli_rntm_num_threads(&rntm_local);

	if (nt_rntm <= 0)
	{
		// nt is less than one if BLIS manual setting of parallelism
		// has been used. Parallelism here will be product of values.
		nt_rntm = bli_rntm_calc_num_threads(&rntm_local);
	}

#ifdef AOCL_DYNAMIC

	// Calculate the actual number of threads that will be spawned
	if (*nt_ideal != -1)
	{
		// The if block is executed for all Zen architectures
		*nt_ideal = bli_min(nt_rntm, *nt_ideal);
	}
	else
	{
		/*
			For non-Zen architectures and very large sizes,
			spawn the actual number of threads requested
		*/
		*nt_ideal = nt_rntm;
	}

	/*
	  When the number of element to be processed is less
	  than the number of threads spawn n_elem number of threads.
	*/
	if (n_elem < *nt_ideal)
	{
		*nt_ideal = n_elem;
	}
#else

	// Calculate the actual number of threads that will be spawned
	*nt_ideal = nt_rntm;

#endif
}
