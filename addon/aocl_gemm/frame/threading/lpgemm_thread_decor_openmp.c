/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_config.h"
#include "lpgemm_thread_decor_openmp.h"
#include "lpgemm_types.h"
#include "lpgemm_5loop_interface_apis.h"

#ifdef BLIS_ENABLE_OPENMP

#define BLIS_LPGEMM_NUM_STATIC_COMMS 96

BLIS_INLINE dim_t next_factor
     (
       const dim_t nt,
       const dim_t part_nt
     )
{
	if ( part_nt == nt )
	{
		return part_nt;
	}

	dim_t nt_temp = part_nt + 1;
	while ( ( nt_temp <= nt ) && ( ( nt % nt_temp ) != 0 ) )
	{
		nt_temp++;
	}
	return nt_temp;
}

BLIS_INLINE dim_t prev_factor
     (
       const dim_t nt,
       const dim_t part_nt
     )
{
	if ( part_nt == 1 )
	{
		return part_nt;
	}

	dim_t nt_temp = part_nt - 1;
	while ( ( nt_temp >= 1 ) && ( ( nt % nt_temp ) != 0 ) )
	{
		nt_temp--;
	}
	return nt_temp;
}

BLIS_INLINE void lpgemm_pnl_wrk_heur_adjust_ic_jc_ways
     (
       dim_t MR,
       dim_t NR,
       dim_t m,
       dim_t n,
       dim_t* n_threads,
       dim_t* ic_ways,
       dim_t* jc_ways
     )
{
	// This function currently only increments ic and subsequently decrements
	// jc. Cannot proceed if all threads are allocated to ic.
	// The factorization adjustment here is based on improving the B NR panel
	// distribution among the jc threads.
	dim_t mu = ( m + MR - 1 ) / MR;
	dim_t nu = ( n + NR - 1 ) / NR;

	// The next 3 ic factors will be considered to see if it results in better
	// NR panel distribution and subsequently reduce the per thread panel work.
	dim_t nu_mod_jc_ways = nu % ( *jc_ways );
	if ( ( nu_mod_jc_ways != 0 ) && ( ( *ic_ways ) < ( *n_threads ) ) )
	{
		dim_t mu_ic_cur = ( mu + ( *ic_ways ) - 1 ) / ( *ic_ways );
		dim_t nu_jc_cur = ( nu + ( *jc_ways ) - 1 ) / ( *jc_ways );
		dim_t panel_work_cur = mu_ic_cur + nu_jc_cur;

		const dim_t next_ic = next_factor( ( *n_threads ), ( *ic_ways ) );
		const dim_t prev_jc = prev_factor( ( *n_threads ), ( *jc_ways ) );
		dim_t mu_ic_next = ( mu + next_ic - 1 ) / next_ic;
		dim_t nu_jc_prev = ( nu + prev_jc - 1 ) / prev_jc;
		dim_t panel_work_next = mu_ic_next + nu_jc_prev;

		if ( panel_work_next < panel_work_cur )
		{
			panel_work_cur = panel_work_next;
			( *ic_ways ) = next_ic;
			( *jc_ways ) = prev_jc;
		}

		nu_mod_jc_ways = nu % ( *jc_ways );
		if ( ( nu_mod_jc_ways != 0 ) && ( next_ic < ( *n_threads ) ) )
		{
			const dim_t next_next_ic = next_factor( ( *n_threads ), next_ic );
			const dim_t prev_prev_jc = prev_factor( ( *n_threads ), prev_jc );
			dim_t mu_ic_next_next = ( mu + next_next_ic - 1 ) / next_next_ic;
			dim_t nu_jc_prev_prev = ( nu + prev_prev_jc - 1 ) / prev_prev_jc;
			dim_t panel_work_next_next = mu_ic_next_next + nu_jc_prev_prev;

			if ( panel_work_next_next < panel_work_cur )
			{
				panel_work_cur = panel_work_next_next;
				( *ic_ways ) = next_next_ic;
				( *jc_ways ) = prev_prev_jc;
			}

			nu_mod_jc_ways = nu % ( *jc_ways );
			if ( ( nu_mod_jc_ways != 0 ) && ( next_next_ic < ( *n_threads ) ) )
			{
				const dim_t next_next_next_ic =
								next_factor
								(
								  ( *n_threads ), next_next_ic
								);
				const dim_t prev_prev_prev_jc =
								prev_factor
								(
								  ( *n_threads ), prev_prev_jc
								);
				dim_t mu_ic_next_next_next =
						( mu + next_next_next_ic - 1 ) / next_next_next_ic;
				dim_t nu_jc_prev_prev_prev =
						( nu + prev_prev_prev_jc - 1 ) / prev_prev_prev_jc;
				dim_t panel_work_next_next_next =
						mu_ic_next_next_next + nu_jc_prev_prev_prev;

				if ( panel_work_next_next_next < panel_work_cur )
				{
					( *ic_ways ) = next_next_next_ic;
					( *jc_ways ) = prev_prev_prev_jc;
				}
			}
		}
	}
}

BLIS_INLINE void lpgemm_adjust_ic_jc_ways
     (
       dim_t   m,
       dim_t   n,
       dim_t* n_threads,
       dim_t* ic_ways,
       dim_t* jc_ways
     )
{
	const dim_t m_ic = m / ( *ic_ways );
	const dim_t n_jc = n / ( *jc_ways );
	const int64_t cur_work_per_thread = m_ic + n_jc;

	const dim_t next_ic = next_factor( ( *n_threads ), ( *ic_ways ) );
	const dim_t prev_ic = prev_factor( ( *n_threads ), ( *ic_ways ) );
	const dim_t next_jc = next_factor( ( *n_threads ), ( *jc_ways ) );
	const dim_t prev_jc = prev_factor( ( *n_threads ), ( *jc_ways ) );

	const dim_t m_next_ic = m / next_ic;
	const dim_t m_prev_ic = m / prev_ic;
	const dim_t n_next_jc = n / next_jc;
	const dim_t n_prev_jc = n / prev_jc;

	const int64_t next_jc_work_per_thread = n_next_jc + m_prev_ic;
	const int64_t next_ic_work_per_thread = m_next_ic + n_prev_jc;

	bool can_increase_ic = FALSE;
	bool can_increase_jc = FALSE;

	if ( next_ic_work_per_thread <= cur_work_per_thread )
	{
		can_increase_ic = TRUE;
	}
	else if ( next_jc_work_per_thread < cur_work_per_thread )
	{
		can_increase_jc = TRUE;
	}

	if ( can_increase_ic )
	{
		( *ic_ways ) = next_ic;
		( *jc_ways ) = prev_jc;
	}
	else if ( can_increase_jc )
	{
		// Giving priority to ic and m dimensions, if m >= n, jc must be < ic.
		if ( ( ( m >= n ) && ( prev_ic >= next_jc ) ) ||
			 ( ( m < n ) && ( prev_ic <= next_jc ) ) )
		{
			( *ic_ways ) = prev_ic;
			( *jc_ways ) = next_jc;
		}
	}
}

BLIS_INLINE void lpgemm_u8s8s16o16_get_threading
     (
       dim_t*  n_threads,
       dim_t*  ic_ways,
       dim_t*  jc_ways,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm_g
     )
{
	*n_threads = bli_rntm_num_threads( rntm_g );
	*jc_ways = bli_rntm_jc_ways( rntm_g );
	*ic_ways = bli_rntm_ic_ways( rntm_g );

	if ( ( ( *ic_ways ) > 0 ) || ( ( *jc_ways ) > 0 ) )
	{
		// If BLIS_IC_NT or JC_NT are set.
		// Default cases.
 		*ic_ways = ( ( *ic_ways ) > 0 ) ? ( *ic_ways ) : 1;
		*jc_ways = ( ( *jc_ways ) > 0 ) ? ( *jc_ways ) : 1;

		*n_threads = ( *jc_ways ) * ( *ic_ways );
	}
	else if ( ( *n_threads ) > 1 )
	{

		dim_t NR = lpgemm_get_block_size_NR_global_cntx( U8S8S16OS16 );

		if ( n <= NR )
		{
			// If n is less than micro panel dimension, allocating all threads
			// to ic resulted in gains.
			( *ic_ways ) = ( *n_threads );
			( *jc_ways ) = 1;
		}
		else
		{
			// If BLIS_NUM_THREADS are set, generate jc,ic from the same.
			bli_thread_partition_2x2( ( *n_threads ), m, n, ic_ways, jc_ways );
		}
	}
	else
	{
		// Setting all the values to 1 in case n_threads <= 1. This ensures
		// the threading parameters are valid.
		*n_threads = 1;
		*jc_ways = 1;
		*ic_ways = 1;
	}
}

BLIS_INLINE void lpgemm_u8s8s32o32_get_threading
     (
       dim_t*  n_threads,
       dim_t*  ic_ways,
       dim_t*  jc_ways,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm_g
     )
{
	*n_threads = bli_rntm_num_threads( rntm_g );
	*jc_ways = bli_rntm_jc_ways( rntm_g );
	*ic_ways = bli_rntm_ic_ways( rntm_g );

	if ( ( ( *ic_ways ) > 0 ) || ( ( *jc_ways ) > 0 ) )
	{
		// If BLIS_IC_NT or JC_NT are set.
		// Default cases.
 		*ic_ways = ( ( *ic_ways ) > 0 ) ? ( *ic_ways ) : 1;
		*jc_ways = ( ( *jc_ways ) > 0 ) ? ( *jc_ways ) : 1;

		*n_threads = ( *jc_ways ) * ( *ic_ways );
	}
	else if ( ( *n_threads ) > 1 )
	{

		dim_t NR = lpgemm_get_block_size_NR_global_cntx( U8S8S32OS32 );
		dim_t MR = lpgemm_get_block_size_MR_global_cntx( U8S8S32OS32 );

		if ( n <= NR )
		{
			// If n is less than micro panel dimension, allocating all threads
			// to ic resulted in gains.
			( *ic_ways ) = ( *n_threads );
			( *jc_ways ) = 1;
		}
		else
		{
			// If BLIS_NUM_THREADS are set, generate jc,ic from the same.
			bli_thread_partition_2x2( ( *n_threads ), m, n, ic_ways, jc_ways );

			lpgemm_adjust_ic_jc_ways( m, n, n_threads, ic_ways, jc_ways );

			lpgemm_pnl_wrk_heur_adjust_ic_jc_ways
			(
			  MR, NR, m, n,
			  n_threads, ic_ways, jc_ways
			);
		}
	}
	else
	{
		// Setting all the values to 1 in case n_threads <= 1. This ensures
		// the threading parameters are valid.
		*n_threads = 1;
		*jc_ways = 1;
		*ic_ways = 1;
	}
}

BLIS_INLINE void lpgemm_bf16bf16f32of32_get_threading
     (
       dim_t*  n_threads,
       dim_t*  ic_ways,
       dim_t*  jc_ways,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm_g
     )
{
	*n_threads = bli_rntm_num_threads( rntm_g );
	*jc_ways = bli_rntm_jc_ways( rntm_g );
	*ic_ways = bli_rntm_ic_ways( rntm_g );

	if ( ( ( *ic_ways ) > 0 ) || ( ( *jc_ways ) > 0 ) )
	{
		// If BLIS_IC_NT or JC_NT are set.
		// Default cases.
 		*ic_ways = ( ( *ic_ways ) > 0 ) ? ( *ic_ways ) : 1;
		*jc_ways = ( ( *jc_ways ) > 0 ) ? ( *jc_ways ) : 1;

		*n_threads = ( *jc_ways ) * ( *ic_ways );
	}
	else if ( ( *n_threads ) > 1 )
	{

		dim_t NR = lpgemm_get_block_size_NR_global_cntx( BF16BF16F32OF32 );
		dim_t MR = lpgemm_get_block_size_MR_global_cntx( BF16BF16F32OF32 );

		if ( n <= NR )
		{
			// If n is less than micro panel dimension, allocating all threads
			// to ic resulted in gains.
			( *ic_ways ) = ( *n_threads );
			( *jc_ways ) = 1;
		}
		else
		{
			// If BLIS_NUM_THREADS are set, generate jc,ic from the same.
			bli_thread_partition_2x2( ( *n_threads ), m, n, ic_ways, jc_ways );
			lpgemm_adjust_ic_jc_ways( m, n, n_threads, ic_ways, jc_ways );
			lpgemm_pnl_wrk_heur_adjust_ic_jc_ways
			(
			  MR, NR, m, n,
			  n_threads, ic_ways, jc_ways
			);
		}
	}
	else
	{
		// Setting all the values to 1 in case n_threads <= 1. This ensures
		// the threading parameters are valid.
		*n_threads = 1;
		*jc_ways = 1;
		*ic_ways = 1;
	}
}

// Some aspects of sgemm smart threading incorporated here. Eventually this
// will be redirected to the sgemm smart threading API.
BLIS_INLINE void lpgemm_f32f32f32of32_get_threading
     (
       dim_t*  n_threads,
       dim_t*  ic_ways,
       dim_t*  jc_ways,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       rntm_t* rntm_g
     )
{
	// Query the global cntx.
	cntx_t* cntx = bli_gks_query_cntx();

	num_t dt = BLIS_FLOAT;

	// Query the context for SUP limits.
	const dim_t MT = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_MT, cntx );
	const dim_t NT = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_NT, cntx );
	const dim_t KT = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_KT, cntx );

	const dim_t MT_2 = MT / 2;

	*n_threads = bli_rntm_num_threads( rntm_g );
	*jc_ways = bli_rntm_jc_ways( rntm_g );
	*ic_ways = bli_rntm_ic_ways( rntm_g );

	if ( ( ( *ic_ways ) > 0 ) || ( ( *jc_ways ) > 0 ) )
	{
		// If BLIS_IC_NT or JC_NT are set.
		// Default cases.
 		*ic_ways = ( ( *ic_ways ) > 0 ) ? ( *ic_ways ) : 1;
		*jc_ways = ( ( *jc_ways ) > 0 ) ? ( *jc_ways ) : 1;

		*n_threads = ( *jc_ways ) * ( *ic_ways );
	}
	else if ( ( *n_threads ) > 1 )
	{
		// If BLIS_NUM_THREADS are set, generate jc,ic from the same.
		bli_thread_partition_2x2( ( *n_threads ), m, n, ic_ways, jc_ways );

		lpgemm_adjust_ic_jc_ways( m, n, n_threads, ic_ways, jc_ways );
	}
	else
	{
		// Setting all the values to 1 in case n_threads <= 1. This ensures
		// the threading parameters are valid.
		*n_threads = 1;
		*jc_ways = 1;
		*ic_ways = 1;
	}

	// Native -> SUP path.
	const dim_t m_ic = m / ( *ic_ways );
	const dim_t n_jc = n / ( *jc_ways );
	const dim_t page_size = bli_info_get_page_size();
	const dim_t page_size_b_floatx2 =
			2 * ( page_size / sizeof( float ) );

	if ( ( m >= MT ) && ( n >= NT ) && ( k >= KT ) )
	{
		if ( ( k > page_size_b_floatx2 ) ||
			 ( ( k <= page_size_b_floatx2 ) &&
				  ( m_ic > MT_2 ) && ( n_jc >= NT ) ) )
		{
			bli_rntm_set_pack_a( 1, rntm_g );
		}
	}
}

#define GEN_LPGEMM_OPENMP_DECORATOR(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _openmp_thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
       const AOCL_MEMORY_TAG mtag_b, \
       C_type*               c, \
       const dim_t           rs_c, \
       C_type                alpha, \
       C_type                beta, \
       rntm_t*               rntm_g, \
       lpgemm_post_op*       post_op_list, \
       bool                  c_downscale \
     ) \
{ \
	dim_t n_threads; \
 \
	/* Factorization of threads along m and n dimension respectively.*/ \
	dim_t ic_ways; \
	dim_t jc_ways; \
 \
	lpgemm_ ## LPGEMM_SFX ## _get_threading \
	( \
	  &n_threads, \
	  &ic_ways, &jc_ways, \
	  m, n, k, rntm_g \
	); \
 \
	/* Set the packing block allocator field of the rntm. This will be
	 * inherited by all of the child threads when they make local copies of
	 * the rntm below.*/ \
	bli_membrk_rntm_set_membrk( rntm_g ); \
 \
	thrcomm_t static_lpgemm_comms[BLIS_LPGEMM_NUM_STATIC_COMMS]; \
	thrcomm_t* cur_lpgemm_comms = static_lpgemm_comms; \
 \
	if ( jc_ways > BLIS_LPGEMM_NUM_STATIC_COMMS ) \
	{ \
		cur_lpgemm_comms = bli_malloc_intl( jc_ways * sizeof( thrcomm_t ) ); \
	} \
	for ( dim_t i = 0; i < jc_ways; ++i ) \
	{ \
		bli_thrcomm_init( ic_ways, &cur_lpgemm_comms[i] ); \
	} \
 \
	_Pragma( "omp parallel num_threads(n_threads)" ) \
	{ \
		/* Create a thread-local copy of the master thread's rntm_t. This is
		 * necessary since we want each thread to be able to track its own
		 * small block pool_t as it executes down the function stack.*/ \
		rntm_t rntm_l = *rntm_g; \
 \
		/* lpgemm_thrinfo_t object will be used to generate thrinfo_t objects
		 * for use in blis mt framework inside the respective mat mul driver
		 * functions.*/ \
		lpgemm_thrinfo_t thread; \
		thread.n_threads = n_threads; \
		thread.tid = omp_get_thread_num(); \
		thread.ic_ways = ic_ways; \
		thread.jc_ways = jc_ways; \
		thread.comm = cur_lpgemm_comms; \
 \
		lpgemm_rowvar_ ## LPGEMM_SFX \
		( \
		  m, n, k, \
		  a, rs_a, cs_a, mtag_a, \
		  b, rs_b, cs_b, mtag_b, \
		  c, rs_c, \
		  alpha, \
		  beta, \
		  &rntm_l, \
		  &thread, \
		  post_op_list, c_downscale \
		); \
	} \
	if ( jc_ways > BLIS_LPGEMM_NUM_STATIC_COMMS ) \
	{ \
		bli_free_intl( cur_lpgemm_comms ); \
	} \
} \

GEN_LPGEMM_OPENMP_DECORATOR(uint8_t,int8_t,int16_t,u8s8s16o16)
GEN_LPGEMM_OPENMP_DECORATOR(uint8_t,int8_t,int32_t,u8s8s32o32)
GEN_LPGEMM_OPENMP_DECORATOR(bfloat16,bfloat16,float,bf16bf16f32of32)
GEN_LPGEMM_OPENMP_DECORATOR(float,float,float,f32f32f32of32)

#else

#define GEN_LPGEMM_DECORATOR(A_type,B_type,C_type,LPGEMM_SFX) \
void lpgemm_ ## LPGEMM_SFX ## _thread_decorator \
     ( \
       const dim_t           m, \
       const dim_t           n, \
       const dim_t           k, \
       const A_type*         a, \
       const dim_t           rs_a, \
       const dim_t           cs_a, \
       const AOCL_MEMORY_TAG mtag_a, \
       const B_type*         b, \
       const dim_t           rs_b, \
       const dim_t           cs_b, \
       const AOCL_MEMORY_TAG mtag_b, \
       C_type*               c, \
       const dim_t           rs_c, \
       C_type                alpha, \
       C_type                beta, \
       rntm_t*               rntm_g, \
       lpgemm_post_op*       post_op_list, \
       bool                  c_downscale \
     ) \
{ \
	dim_t n_threads = 1; \
 \
	/* Factorization of threads along m and n dimension respectively.*/ \
	dim_t ic_ways = 1; \
	dim_t jc_ways = 1; \
 \
	/* Set the packing block allocator field of the rntm. This will be
	 * inherited by all of the child threads when they make local copies of
	 * the rntm below.*/ \
	bli_membrk_rntm_set_membrk( rntm_g ); \
 \
	thrcomm_t static_lpgemm_comm; \
	thrcomm_t* cur_lpgemm_comm = &static_lpgemm_comm; \
 \
	bli_thrcomm_init( ic_ways, cur_lpgemm_comm ); \
 \
	/* lpgemm_thrinfo_t object will be used to generate thrinfo_t objects
	 * for use in blis mt framework inside the respective mat mul driver
	 * functions.*/ \
	lpgemm_thrinfo_t thread; \
	thread.n_threads = n_threads; \
	thread.tid = 0; \
	thread.ic_ways = ic_ways; \
	thread.jc_ways = jc_ways; \
	thread.comm = cur_lpgemm_comm; \
 \
	lpgemm_rowvar_ ## LPGEMM_SFX \
	( \
	  m, n, k, \
	  a, rs_a, cs_a, mtag_a, \
	  b, rs_b, cs_b, mtag_b, \
	  c, rs_c, \
	  alpha, \
	  beta, \
	  rntm_g, \
	  &thread, \
	  post_op_list, c_downscale \
	); \
} \

GEN_LPGEMM_DECORATOR(uint8_t,int8_t,int16_t,u8s8s16o16)
GEN_LPGEMM_DECORATOR(uint8_t,int8_t,int32_t,u8s8s32o32)
GEN_LPGEMM_DECORATOR(bfloat16,bfloat16,float,bf16bf16f32of32)
GEN_LPGEMM_DECORATOR(float,float,float,f32f32f32of32)

#endif
