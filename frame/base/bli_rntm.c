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

// A mutex to allow synchronous access to global_rntm.
bli_pthread_mutex_t global_rntm_mutex = BLIS_PTHREAD_MUTEX_INITIALIZER;

// -----------------------------------------------------------------------------

void bli_rntm_init_from_global( rntm_t* rntm )
{
	// We must ensure that global_rntm has been initialized.
	bli_init_once();

	// Acquire the mutex protecting global_rntm.
	bli_pthread_mutex_lock( &global_rntm_mutex );

	*rntm = global_rntm;

	// Release the mutex protecting global_rntm.
	bli_pthread_mutex_unlock( &global_rntm_mutex );
}

// -----------------------------------------------------------------------------

void bli_rntm_set_num_threads
     (
       dim_t   nt,
       rntm_t* rntm
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING

	// Record the total number of threads to use.
	bli_rntm_set_num_threads_only( nt, rntm );

	// Set the individual ways of parallelism to default states. This
	// must be done before sanitization so that the .num_threads field
	// will prevail over any previous ways that may have been set.
	bli_rntm_clear_ways_only( rntm );

	// Ensure that the rntm_t is in a consistent state.
	bli_rntm_sanitize( rntm );

#else

	// When multithreading is disabled at compile time, ignore the user's
	// request. And just to be safe, reassert the default rntm_t values.
	bli_rntm_clear_num_threads_only( rntm );
	bli_rntm_clear_ways_only( rntm );

#endif
}

void bli_rntm_set_ways
     (
       dim_t   jc,
       dim_t   pc,
       dim_t   ic,
       dim_t   jr,
       dim_t   ir,
       rntm_t* rntm
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING

	// Record the number of ways of parallelism per loop.
	bli_rntm_set_jc_ways_only( jc, rntm );
	bli_rntm_set_pc_ways_only(  1, rntm ); // Disable pc_nt values.
	bli_rntm_set_ic_ways_only( ic, rntm );
	bli_rntm_set_jr_ways_only( jr, rntm );
	bli_rntm_set_ir_ways_only( ir, rntm );
	bli_rntm_set_pr_ways_only(  1, rntm );

	// Set the total number of threads to its default state. This isn't
	// strictly necessary, but is done in case the priority of nt vs.
	// ways ever changes. (Currently, the ways always prevail over the
	// number of threads, if both are set.)
	bli_rntm_clear_num_threads_only( rntm );

	// Ensure that the rntm_t is in a consistent state.
	bli_rntm_sanitize( rntm );

#else

	// When multithreading is disabled at compile time, ignore the user's
	// request. And just to be safe, reassert the default rntm_t values.
	bli_rntm_clear_num_threads_only( rntm );
	bli_rntm_clear_ways_only( rntm );

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
	bli_rntm_factorize( m, n, k, rntm );

	#if 0
	printf( "bli_rntm_set_ways_for_op()\n" );
	bli_rntm_print( rntm );
	#endif

	// Now modify the number of ways, if necessary, based on the operation.

	// Consider gemm (hemm, symm), gemmt (herk, her2k, syrk, syr2k), and
	// trmm (trmm, trmm3).
	if (
#ifdef BLIS_ENABLE_JRIR_TLB
	     l3_op == BLIS_GEMM  ||
	     l3_op == BLIS_GEMMT ||
	     l3_op == BLIS_TRMM  ||
#endif
	     FALSE
	   )
	{
		dim_t jc = bli_rntm_jc_ways( rntm );
		dim_t pc = bli_rntm_pc_ways( rntm );
		dim_t ic = bli_rntm_ic_ways( rntm );
		dim_t jr = bli_rntm_jr_ways( rntm );
		dim_t ir = bli_rntm_ir_ways( rntm );

		// If TLB is enabled for gemm or gemmt, redirect any ir loop parallelism
		// into the jr loop.
		bli_rntm_set_ways_only
		(
		  jc,
		  pc,
		  ic,
		  jr * ir,
		  1,
		  rntm
		);
	}

	// Consider trmm, trmm3, trsm.
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

void bli_rntm_sanitize
     (
       rntm_t* rntm
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING

	timpl_t ti = bli_rntm_thread_impl( rntm );
	dim_t   nt = bli_rntm_num_threads( rntm );
	dim_t   jc = bli_rntm_jc_ways( rntm );
	dim_t   pc = bli_rntm_pc_ways( rntm );
	dim_t   ic = bli_rntm_ic_ways( rntm );
	dim_t   jr = bli_rntm_jr_ways( rntm );
	dim_t   ir = bli_rntm_ir_ways( rntm );

	bool auto_factor = FALSE;

	bool nt_set   = FALSE;
	bool ways_set = FALSE;

	if ( ti == BLIS_SINGLE )
	{
		// If the threading implementation was set to BLIS_SINGLE, we ignore
		// everything else.

		nt = 1;
		jc = pc = ic = jr = ir = 1;
		auto_factor = FALSE;
	}
	else // if ( ti != BLIS_SINGLE )
	{
		// If the threading implementation was set to one of the true
		// multithreading implementations (e.g. BLIS_OPENMP, BLIS_POSIX),
		// we proceed to interpret and process the rntm_t's fields.

		// Some users are mischievous/dumb. Make sure they don't cause trouble.
		if ( nt < 1 ) nt = 1;
		if ( jc < 1 ) jc = 1;
		if ( pc < 1 ) pc = 1;
		if ( ic < 1 ) ic = 1;
		if ( jr < 1 ) jr = 1;
		if ( ir < 1 ) ir = 1;

		// Now establish whether or not the number of threads or ways of
		// parallelism were set to meaningful values.
		if ( nt > 1 ) { nt_set   = TRUE; }
		if ( jc > 1 ) { ways_set = TRUE; }
		if ( pc > 1 ) { ways_set = TRUE; pc = 1; } // Disable pc_nt values.
		if ( ic > 1 ) { ways_set = TRUE; }
		if ( jr > 1 ) { ways_set = TRUE; }
		if ( ir > 1 ) { ways_set = TRUE; }

		// Next, we use the values of nt_set and ways_set to determine how to
		// interpret the original values we found in the rntm_t object.

		if ( ways_set == TRUE )
		{
			// If the per-loop ways of parallelism were set, then we use the values
			// that were given and interpreted above. Since the per-loop ways are
			// known, we can calculate the total number of threads. Notice that if
			// the user also happened to set the total number of threads, that value
			// is discarded in favor of the implied value from the per-loop ways of
			// parallelism.

			nt = jc * pc * ic * jr * ir;
			auto_factor = FALSE;
		}
		else if ( ways_set == FALSE && nt_set == TRUE )
		{
			// If the ways were not set but the number of thread was set, then we
			// will attempt to automatically generate a thread factorization that
			// will work given the problem size. This happens later, in
			// bli_rntm_factorize().

			auto_factor = TRUE;
		}
		else // if ( ways_set == FALSE && nt_set == FALSE )
		{
			// If neither the ways nor the number of threads were set, then the
			// rntm_t was not meaningfully changed since initialization. This means
			// the ways are already 1, which will lead to the default behavior of
			// single-threaded execution.
		}
	}

	// Save the results back in the rntm_t object.
	// Note: We don't need to set the .thread_impl field of the rntm_t because
	// it was not changed in the sanitization process.
	//bli_rntm_set_thread_impl_only( ti, rntm );
	bli_rntm_set_num_threads_only( nt, rntm );
	bli_rntm_set_ways_only( jc, pc, ic, jr, ir, rntm );
	bli_rntm_set_auto_factor_only( auto_factor, rntm );

#else

	// When multithreading is disabled, always set the per-loop ways of
	// parallelism to 1.
	bli_rntm_set_thread_impl_only( BLIS_SINGLE, rntm );
	bli_rntm_set_num_threads_only( 1, rntm );
	bli_rntm_set_ways_only( 1, 1, 1, 1, 1, rntm );
	bli_rntm_set_auto_factor_only( FALSE, rntm );

#endif
}

void bli_rntm_factorize
     (
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING

	// The .auto_factor field would have been set either at initialization or
	// when the rntm_t was sanitized after being updated by the user.
	if ( bli_rntm_auto_factor( rntm ) )
	{
		dim_t nt = bli_rntm_num_threads( rntm );
		dim_t jc = bli_rntm_jc_ways( rntm );
		dim_t pc = bli_rntm_pc_ways( rntm );
		dim_t ic = bli_rntm_ic_ways( rntm );
		dim_t jr = bli_rntm_jr_ways( rntm );
		dim_t ir = bli_rntm_ir_ways( rntm );

		if ( 0 < m && 0 < n && 0 <= k )
		{
			#ifdef BLIS_DISABLE_AUTO_PRIME_NUM_THREADS
			// If use of prime numbers is disallowed for automatic thread
			// factorizations, we first check if the number of threads requested
			// is prime. If it is prime, and it exceeds a minimum threshold, then
			// we reduce the number of threads by one so that the number is not
			// prime. This will allow for automatic thread factorizations to span
			// two dimensions (loops), which tends to be more efficient.
			if ( bli_is_prime( nt ) && BLIS_NT_MAX_PRIME < nt ) nt -= 1;
			#endif

			//printf( "m n = %d %d  BLIS_THREAD_RATIO_M _N = %d %d\n",
			//         (int)m, (int)n, (int)BLIS_THREAD_RATIO_M,
			//                         (int)BLIS_THREAD_RATIO_N );

			bli_thread_partition_2x2( nt, m*BLIS_THREAD_RATIO_M,
			                              n*BLIS_THREAD_RATIO_N, &ic, &jc );

			//printf( "jc ic = %d %d\n", (int)jc, (int)ic );

			for ( ir = BLIS_THREAD_MAX_IR ; ir > 1 ; ir-- )
			{
				if ( ic % ir == 0 ) { ic /= ir; break; }
			}

			for ( jr = BLIS_THREAD_MAX_JR ; jr > 1 ; jr-- )
			{
				if ( jc % jr == 0 ) { jc /= jr; break; }
			}
		}

		// Save the results back in the rntm_t object.
		bli_rntm_set_num_threads_only( nt, rntm );
		bli_rntm_set_ways_only( jc, pc, ic, jr, ir, rntm );
	}

#else

	// When multithreading is disabled at compile time, the rntm can keep its
	// default initialization values since using one thread requires no
	// factorization.

#endif
}

void bli_rntm_factorize_sup
     (
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm
     )
{
#ifdef BLIS_ENABLE_MULTITHREADING

	// The .auto_factor field would have been set either at initialization or
	// when the rntm_t was sanitized after being updated by the user.
	if ( bli_rntm_auto_factor( rntm ) )
	{
		dim_t nt = bli_rntm_num_threads( rntm );
		dim_t jc = bli_rntm_jc_ways( rntm );
		dim_t pc = bli_rntm_pc_ways( rntm );
		dim_t ic = bli_rntm_ic_ways( rntm );
		dim_t jr = bli_rntm_jr_ways( rntm );
		dim_t ir = bli_rntm_ir_ways( rntm );

		if ( 0 < m && 0 < n && 0 <= k )
		{
			#ifdef BLIS_DISABLE_AUTO_PRIME_NUM_THREADS
			// If use of prime numbers is disallowed for automatic thread
			// factorizations, we first check if the number of threads requested
			// is prime. If it is prime, and it exceeds a minimum threshold, then
			// we reduce the number of threads by one so that the number is not
			// prime. This will allow for automatic thread factorizations to span
			// two dimensions (loops), which tends to be more efficient.
			if ( bli_is_prime( nt ) && BLIS_NT_MAX_PRIME < nt ) nt -= 1;
			#endif

			bli_thread_partition_2x2( nt, m,
										  n, &ic, &jc );
			ir = 1; jr = 1;
		}

		// Save the results back in the rntm_t object.
		bli_rntm_set_num_threads_only( nt, rntm );
		bli_rntm_set_ways_only( jc, pc, ic, jr, ir, rntm );
	}

#else

	// When multithreading is disabled at compile time, the rntm can keep its
	// default initialization values since using one thread requires no
	// factorization.

#endif
}

void bli_rntm_print
     (
       const rntm_t* rntm
     )
{
	timpl_t ti = bli_rntm_thread_impl( rntm );

	dim_t   af = bli_rntm_auto_factor( rntm );

	dim_t   nt = bli_rntm_num_threads( rntm );

	dim_t   jc = bli_rntm_jc_ways( rntm );
	dim_t   pc = bli_rntm_pc_ways( rntm );
	dim_t   ic = bli_rntm_ic_ways( rntm );
	dim_t   jr = bli_rntm_jr_ways( rntm );
	dim_t   ir = bli_rntm_ir_ways( rntm );

	printf( "thread impl: %d\n", ti );
	printf( "rntm contents    nt  jc  pc  ic  jr  ir\n" );
	printf( "autofac? %1d | %4d%4d%4d%4d%4d%4d\n", (int)af,
	                                               (int)nt, (int)jc, (int)pc,
	                                               (int)ic, (int)jr, (int)ir );
}

// -----------------------------------------------------------------------------

dim_t bli_rntm_calc_num_threads_in
     (
       const bszid_t* bszid_cur,
       const rntm_t*  rntm
     )
{
	/*                                     // bp algorithm:
	   bszid_t bszids[7] = { BLIS_NC,      // level 0: 5th loop
	                         BLIS_KC,      // level 1: 4th loop
	                         BLIS_NO_PART, // level 2: pack B
	                         BLIS_MC,      // level 3: 3rd loop
	                         BLIS_NO_PART, // level 4: pack A
	                         BLIS_NR,      // level 5: 2nd loop
	                         BLIS_MR,      // level 6: 1st loop
	                         BLIS_KR       // level 7: ukr loop

	                         ...           // pb algorithm:
	                         BLIS_NR,      // level 5: 2nd loop
	                         BLIS_MR,      // level 6: 1st loop
	                         BLIS_KR       // level 7: ukr loop
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
		dim_t         cur_way = 1;

		// We assume bszid is in {NC,KC,MC,NR,MR,KR} if it is not
		// BLIS_NO_PART.
		if ( bszid != BLIS_NO_PART )
			cur_way = bli_rntm_ways_for( bszid, rntm );
		else
			cur_way = 1;

		n_threads_in *= cur_way;
	}
#endif

