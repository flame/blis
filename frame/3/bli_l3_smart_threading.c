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
#include "bli_l3_smart_threading.h"

#ifdef AOCL_DYNAMIC

// Utility functions.
static inline dim_t next_factor
     (
       const dim_t nt,
       const dim_t part_nt
     )
{
	if ( part_nt == nt)
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

static inline dim_t prev_factor
     (
       const dim_t nt,
       const dim_t part_nt
     )
{
	if ( part_nt == 1)
	{
		return part_nt;
	}

	dim_t nt_temp = part_nt - 1;
	while ((nt_temp >= 1) && ((nt % nt_temp) != 0))
	{
		nt_temp--;
	}
	return nt_temp;
}
// End utility functions.

static err_t bli_gemm_ic_jc_optimum_sup_arch_dispatcher
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       const dim_t max_available_nt,
       cntx_t* cntx,
       rntm_t* rntm
     );

static err_t bli_gemm_ic_jc_optimum_sup_zen3
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       const dim_t max_available_nt,
       cntx_t* cntx,
       rntm_t* rntm
     );

static void bli_gemm_cache_heur_adjust_ic_jc_sup_zen3
     (
       const dim_t m,
       const dim_t n,
       const dim_t k,
       dim_t nt,
       dim_t* ic,
       dim_t* jc,
       const dim_t MR,
       const dim_t NR,
       const dim_t MC,
       const dim_t KC
     );

err_t bli_check_and_transform_native_to_SUP
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       dim_t ic,
       dim_t jc,
       const dim_t NR,
       const dim_t MC,
       const dim_t KC,
       cntx_t* cntx,
       rntm_t* rntm
     );

err_t bli_gemm_smart_threading_sup
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       const dim_t max_available_nt,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	err_t ret_val = BLIS_FAILURE;

	// Sanity check, max available threads should be atleast 4 for the
	// smart threading/factorization to be meaningful. For nt < 4 the
	// default ic,jc factorization holds good.
	if ( ( m <= 1 ) || ( n <= 1 ) ||  ( k <= 1 ) || ( max_available_nt < 4 ) )
	{
		return ret_val;
	}

	if ( bli_is_float( dt ) )
	{
		ret_val = bli_gemm_ic_jc_optimum_sup_arch_dispatcher
				  (
				    dt, elem_size, is_rrr_rrc_rcr_crr, m, n, k,
				    max_available_nt, cntx, rntm
				  );
	}
	else
	{
		// Other data types not supported for now.
	}

	if ( ret_val == BLIS_SUCCESS )
	{
		// This is a workaround to ensure that auto_factor attribute of rntm_t
		// is not set to TRUE inside bli_rntm_set_ways_from_rntm_sup. Also
		// the nt value will be properly set to ic*jc towards the end of
		// bli_rntm_set_ways_from_rntm_sup.
		bli_rntm_set_num_threads_only( -1, rntm );
	}

	return ret_val;
}

static err_t bli_gemm_ic_jc_optimum_sup_arch_dispatcher
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       const dim_t max_available_nt,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	err_t ret_val = BLIS_FAILURE;

	arch_t id = bli_arch_query_id();
	if ( id == BLIS_ARCH_ZEN3 )
	{
		ret_val = bli_gemm_ic_jc_optimum_sup_zen3
				  (
				    dt, elem_size, is_rrr_rrc_rcr_crr,  m, n, k,
				    max_available_nt, cntx, rntm
				  );
	}
	else
	{
		// Other architectures not supported for now.
	}

	return ret_val;
}

// open zen3 region.
#define NUM_CORES_PER_CCD_ZEN3 8

// Determines the optimal number of threads (nt) and corresponding work split
//  (ic,jc factorization of nt) for gemm on zen3 machines.
static err_t bli_gemm_ic_jc_optimum_sup_zen3
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       const dim_t max_available_nt,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	err_t ret_val = BLIS_SUCCESS;

	const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx );
	const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx );
	const dim_t MC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx );
	const dim_t NC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx );
	const dim_t KC = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx );

	dim_t ic = -1;
	dim_t jc = -1;

	bli_thread_partition_2x2( max_available_nt, m, n, &ic, &jc );

	dim_t jc_per_ccd = ( NUM_CORES_PER_CCD_ZEN3 + ic - 1 ) / ic ;
	dim_t b_mat_data_per_ccd = jc_per_ccd * ( n / jc );

	// All the cores (8) on a CCD share a L3 cache and hence total data
	// loaded by the cores on a CCD should be < NC to avoid L3 contention.
	// In cases where it is violated, it is better to increase ic and
	// reduce B data per CCD, using micro panels mu, nu for thread
	// partitioning can help achieve this. Avoiding further ic,jc
	// adjustment in this case.
	if ( b_mat_data_per_ccd > NC )
	{
		const dim_t mu = m / MR;
		const dim_t nu = n / NR;
		bli_thread_partition_2x2( max_available_nt, mu, nu, &ic, &jc );
	}
	else
	{
		// Adjust the ic,jc in the best match so that m_ic and n_jc
		// turns out to be more cache friendly.
		bli_gemm_cache_heur_adjust_ic_jc_sup_zen3
		(
		  m, n, k, max_available_nt, &ic, &jc, MR, NR, MC, KC
		);
	}

	ret_val = bli_check_and_transform_native_to_SUP
			  (
			    dt, elem_size, is_rrr_rrc_rcr_crr, m, n, k,
			    ic, jc, NR, MC, KC, cntx, rntm
			  );

	if ( ret_val == BLIS_SUCCESS )
	{
		bli_rntm_set_ic_ways_only( ic, rntm );
		bli_rntm_set_jc_ways_only( jc, rntm );
	}

	return ret_val;
}

// The factorization of nt into ic,jc is based on m and n values (for simplicity
// it can be assumed to be based on m:n ratio). It does not take into account
// how the matrices are loaded into cache or which matrix goes to the larger
// cache. Depending on the matrix dimensions, increasing the ic can result in
// reduced loads from main memory to L2 cache for A matrix without any impact on
// B matrix load (since B is streamed into L3, which is larger). Similary
// adjusting jc can result in B matrix panels fitting perfectly within the L1
// cache.This function makes these adjustments on ic,jc.
static void bli_gemm_cache_heur_adjust_ic_jc_sup_zen3
     (
       const dim_t m,
       const dim_t n,
       const dim_t k,
       dim_t nt,
       dim_t* ic,
       dim_t* jc,
       const dim_t MR,
       const dim_t NR,
       const dim_t MC,
       const dim_t KC
     )
{
	const dim_t m_ic = m / ( *ic );
	const dim_t n_jc = n / ( *jc );
	const int64_t cur_work_per_thread = m_ic + n_jc;

	// The next and prev factors are caluclated with respect to the current
	// factor part of nt. In effect
	// 1. next ic * prev jc = nt
	// 2. prev ic * next jc = nt
	// 3. ic * jc = nt
	const dim_t next_ic = next_factor( nt, ( *ic ) );
	const dim_t prev_ic = prev_factor( nt, ( *ic ) );
	const dim_t next_jc = next_factor( nt, ( *jc ) );
	const dim_t prev_jc = prev_factor( nt, ( *jc ) );

	const dim_t m_next_ic = m / next_ic;
	const dim_t m_prev_ic = m / prev_ic;
	const dim_t n_next_jc = n / next_jc;
	const dim_t n_prev_jc = n / prev_jc;
	const dim_t n_jc_modulo_NR = n_jc % NR;
	const dim_t n_prev_jc_modulo_NR = n_prev_jc % NR;

	const int64_t next_jc_work_per_thread = n_next_jc + m_prev_ic;
	const int64_t next_ic_work_per_thread = m_next_ic + n_prev_jc;

	const dim_t MCx2 = MC * 2;
	const dim_t NRx4 = NR * 4;
	const dim_t NRx8 = NR * 8;

	// MC will be reduced if the following mods are zero. Incrementing jc
	// helps in this case.
	const dim_t n_mod_256 = n % 256;
	const dim_t k_mod_256 = k % 256;

	const dim_t k_factor = k / KC;

	bool can_increase_jc = FALSE;
	bool can_increase_ic = FALSE;

	// jc adjustment towards next highest factor if it results in n_jc*KC
	// fittting completely within l1d cache. Only done if ic prev factor
	// does not move m_prev_ic out of good l2 load zone (MC).
	// Performance improvement also observed when n_jc is a multiple of NR.
	if ( ( ( *ic ) > 1 ) && ( ( *jc ) < nt ) )
	{
		// Check whether m_prev_ic remains in good l2 load zone.
		if ( ( ( ( m_ic <= MC ) && ( m_prev_ic <= MC ) ) ||
			   ( m_ic > MC ) ) &&
			 ( ( n_jc > NR ) && ( n_next_jc == NR ) ) )
		{
			can_increase_jc = TRUE;
		}
		// 2x2 factorization doesnt always give equal sum partition.
		else if ( next_jc_work_per_thread < cur_work_per_thread )
		{
			can_increase_jc = TRUE;
		}
	}

	// Favor jc if both n and k are multiples of 256 ( high cache line
	// replacement ).
	if ( ( ( *ic ) < nt ) && ( ( *jc ) > 1) )
	{
		// ic adjustment towards next highest factor if it results in
		// m_next_ic <= MC. This helps in reducing number of A matrix
		// loads per thread to l2 from main memory.
		if ( ( m_ic > MC ) && ( m_next_ic <= MC ) &&
			 ( m_next_ic >= MR ) && ( k_factor > 4 ) )
		{
			can_increase_ic = TRUE;
		}
		// ic adjustment towards next highest factor resulted in better
		// performance when m is sufficiently larger than n and jc prev
		// factor did not result in n_prev_jc moving out of good l2
		// load zone (n_jc < 64).
		else if ( ( m > ( 5 * n ) ) && ( m_ic >= MCx2 ) && ( k_factor > 4 ) &&
				  ( ( n_jc > NRx4 ) ||
					( ( n_jc <= NRx4 ) && ( n_prev_jc <= NRx4 ) ) ) )
		{
			can_increase_ic = TRUE;
		}
		// Performance improvement also observed when n_jc is a multiple
		// of NR.
		else if ( ( n_jc_modulo_NR != 0 ) && ( n_prev_jc_modulo_NR == 0 ) &&
				  ( k_factor > 4 ) )
		{
			can_increase_ic = TRUE;
		}
		// 2x2 factorization doesnt always give equal sum partition.
		else if ( next_ic_work_per_thread <= cur_work_per_thread )
		{
			can_increase_ic = TRUE;
		}
	}

	// Favor jc if both n and k are multiples of 256 ( high cache line
	// replacement ).
	if ( ( n_mod_256 == 0 ) && ( k_mod_256 == 0 ) && ( k > KC ) )
	{
		if ( can_increase_ic == TRUE )
		{
			can_increase_ic = FALSE;
		}
		else if ( can_increase_jc == FALSE )
		{
			can_increase_jc = TRUE;
		}
	}
	// If only one of either n or k is a multiple of 256, favour jc if n per
	// thread is within a heuristic factor of NR.
	else if ( ( ( n_mod_256 == 0 ) || ( k_mod_256 == 0 ) ) && ( k > KC ) )
	{
		if ( ( can_increase_ic == TRUE ) && ( n_jc <= NRx8 ) )
		{
			can_increase_ic = FALSE;
		}
		else if ( ( can_increase_jc == FALSE ) && ( n_next_jc <= NRx8 ) )
		{
			can_increase_jc = TRUE;
		}
	}

	// Increasing ic factor is given a higher priority compared to jc
	// since it was observed that the A matrix loads (main memory -> l2) had
	// more impact on perf compared to B matrix (main memory -> l3 -> l1)
	// for the sizes considered.
	if ( can_increase_ic )
	{
		// It is expected that the larger dimension (m or n) will be
		// allocated a larger share of the thread factorization.
		if ( ( ( m >= n ) && ( next_ic >= prev_jc ) ) ||
		     ( ( m <= n ) && ( next_ic <= prev_jc ) ) )
		{
			*ic = next_ic;
			*jc = prev_jc;
		}
	}
	else if ( can_increase_jc )
	{
		// It is expected that the larger dimension (m or n) will be
		// allocated a larger share of the thread factorization.
		if ( ( ( m >= n ) && ( prev_ic >= next_jc ) ) ||
		     ( ( m <= n ) && ( prev_ic <= next_jc ) ) )
		{
			*ic = prev_ic;
			*jc = next_jc;
		}
	}
}

// It was observed that the SUP thresholds can be lowered and applied on a
// per thread basis in multi threaded scenarios.
err_t bli_check_and_transform_native_to_SUP
     (
       num_t dt,
       siz_t elem_size,
       const bool is_rrr_rrc_rcr_crr,
       const dim_t m,
       const dim_t n,
       const dim_t k,
       dim_t ic,
       dim_t jc,
       const dim_t NR,
       const dim_t MC,
       const dim_t KC,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	err_t ret_val = BLIS_FAILURE;
	dim_t m_ic;
	dim_t n_jc;

	const dim_t MT = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_MT, cntx );
	const dim_t NT = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_NT, cntx );
	const dim_t KT = bli_cntx_get_l3_sup_thresh_dt( dt, BLIS_KT, cntx );

	const dim_t MT_2 = MT / 2;
	const dim_t NTx4 = NT * 4;
	const dim_t NRx8 = NR * 8;

	const dim_t page_size = bli_info_get_page_size();
	const dim_t page_size_b_float = page_size / ( dim_t ) elem_size;
	const dim_t page_size_b_floatx2 = page_size_b_float * 2;

	// Default SUP check without considering per thread dimensions.
	if ( ( k < KT ) || ( m < MT ) || ( n < NT ) )
	{
		ret_val = BLIS_SUCCESS;
	}
	// Per thread SUP limit checking. It was observed that when k is large,
	// (twice page size) moving native to SUP did not help even if m_ic or
	// n_jc were within SUP limits.
	else if ( ( m >= MT ) && ( n >= NT ) && ( k < page_size_b_floatx2 ) )
	{
		m_ic = m / ic;
		n_jc = n / jc;
		// In multi-threaded scenario, it was observed that if the per
		// thread m dimension(A matrix) and n dimension(B matrix) is
		// within a factor of SUP limits, SUP path without packing
		// resulted in gains. Along similar lines, if the B matrix is
		// large enough and reuse is good, packing B matrix alone in SUP
		// resulted in perf gains.
		if ( ( m_ic <= MT_2 ) && ( n_jc < NTx4 ) )
		{
			if ( ( k > KC ) &&
			     ( m_ic >= MC ) && ( n_jc >= NT ) )
			{
				if ( is_rrr_rrc_rcr_crr )
				{
					bli_rntm_set_pack_b( 1, rntm );
				}
				else
				{
					bli_rntm_set_pack_a( 1, rntm );
				}
			}
			ret_val = BLIS_SUCCESS;
		}
		else if ( ( n_jc < NT ) && ( m_ic <= MT ) )
		{
			if ( ( k > KC ) && ( m_ic >= MC ) && ( n_jc >= NRx8 ) )
			{
				if ( is_rrr_rrc_rcr_crr )
				{
					bli_rntm_set_pack_b( 1, rntm );
				}
				else
				{
					bli_rntm_set_pack_a( 1, rntm );
				}
			}
			ret_val = BLIS_SUCCESS;
		}
		else
		{
			ret_val = BLIS_FAILURE;
		}
	}
	else
	{
		ret_val = BLIS_FAILURE;
	}

	return ret_val;
}
// close zen3 region.

#endif
