/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "lpgemm_thread_utils.h"

static bli_pthread_once_t once_check_lpgemm_thread_topo_init = BLIS_PTHREAD_ONCE_INIT;

static lpgemm_thread_attrs_t lpgemm_thread_attrs;

#ifdef BLIS_ENABLE_OPENMP

static void lpgemm_detect_thread_topo()
{
	int nt_max = omp_get_max_threads();
	int num_procs = omp_get_num_procs();

	if ( nt_max > num_procs )
	{
		// Over subscription of threads, no more work distr.
		return;
	}

	lpgemm_thread_attrs.tid_cnt = nt_max;
	lpgemm_thread_attrs.openmp_enabled = TRUE;

	int** thread_core_bind_list = NULL;
	int* adj_tid_cnt_for_core_grps = NULL;
	int* tid_cnt_for_core_grps = NULL;

	thread_core_bind_list = malloc( nt_max * sizeof( int* ) );
	if ( thread_core_bind_list == NULL )
	{
		goto err_handle;
	}

	// Launch max threads to determine the core bininding for all threads
	// within the omp team.
	#pragma omp parallel num_threads(nt_max)
	{
		int thread_num = omp_get_thread_num();
		int thread_place = omp_get_place_num();
		int place_num_procs = omp_get_place_num_procs( thread_place );

		// 1 extra int for storing num_procs value.
		thread_core_bind_list[thread_num] = NULL;
		thread_core_bind_list[thread_num] = malloc( ( place_num_procs + 1 ) * sizeof( int ) );
		if ( thread_core_bind_list[thread_num] != NULL )
		{
			thread_core_bind_list[thread_num][0] = place_num_procs;
			omp_get_place_proc_ids( thread_place, &thread_core_bind_list[thread_num][1] );
		}
	}

	// When SMT is on, this should be 16. Need a way to dynamically retrieve it.
	const int core_grp_size = 8;
	bool can_detect_topo = TRUE;

	lpgemm_thread_attrs.tid_core_grp_id_list = malloc( nt_max * sizeof( int ) );
	if ( lpgemm_thread_attrs.tid_core_grp_id_list == NULL )
	{
		goto err_handle;
	}

	// TIDs are assigned from 0 to nt_max - 1.
	// OpenMP for close distribution need not pin threads to sequential cores
	// in the presence of CCD architecture. Like tid 0-7 will be on core 0-7
	// but tid 8-15 could be on core 96-103. So just checking for increasing
	// core id for corresponding tid wont get accurate core group load.
	// GOMP_CPU_AFFINITY however assigns cores sequentially.
	for ( int ii = 0; ii < nt_max; ++ii )
	{
		lpgemm_thread_attrs.tid_core_grp_id_list[ii] = -1;
		if ( thread_core_bind_list[ii] != NULL)
		{
			// Wrap around the proc/core ids based on number of cores used.
			int st_core_grp_id =
			 ( thread_core_bind_list[ii][1] % num_procs ) / core_grp_size;
			lpgemm_thread_attrs.tid_core_grp_id_list[ii] = st_core_grp_id;
			for ( int jj = 1; jj < thread_core_bind_list[ii][0]; ++jj )
			{
				int cur_core_grp_id =
				 ( thread_core_bind_list[ii][jj + 1] % num_procs ) / core_grp_size;
				if ( cur_core_grp_id != st_core_grp_id )
				{
					// Core binding spanning across core groups,
					// cannot detect topo.
					can_detect_topo = FALSE;
					break;
				}
			}
		}
		if ( can_detect_topo == FALSE )
		{
			// Revert tid_core_grp_list to -1s;
			for ( int jj = 0; jj < ii; ++jj )
			{
				lpgemm_thread_attrs.tid_core_grp_id_list[ii] = -1;
			}
			break;
		}
	}

	int num_core_grps = num_procs / core_grp_size;

	adj_tid_cnt_for_core_grps = malloc( num_core_grps * sizeof( int ) );
	if ( adj_tid_cnt_for_core_grps == NULL )
	{
		goto err_handle;
	}
	for ( int ii = 0; ii < num_core_grps; ++ii )
	{
		adj_tid_cnt_for_core_grps[ii] = 0;
	}

	tid_cnt_for_core_grps = malloc( num_core_grps * sizeof( int ) );
	if ( tid_cnt_for_core_grps == NULL )
	{
		goto err_handle;
	}
	for ( int ii = 0; ii < num_core_grps; ++ii )
	{
		tid_cnt_for_core_grps[ii] = 0;
	}

	// Get count of core groups that are loaded and not loaded with adj ranks.
	// This will give an approximation for thread pin distribution.
	if ( can_detect_topo == TRUE )
	{
		const int core_grp_loaded_thres = 3;
		int core_grp_adj_tid_thres_cnt = 0;
		int core_grp_adj_tid_cnt = 0;
		int core_grp_non_adj_tid_cnt = 0;

		int cur_core_grp_id = lpgemm_thread_attrs.tid_core_grp_id_list[0];
		tid_cnt_for_core_grps[cur_core_grp_id] += 1;

		for ( int ii = 1; ii < nt_max; ++ii )
		{
			if ( lpgemm_thread_attrs.tid_core_grp_id_list[ii] == cur_core_grp_id )
			{
				adj_tid_cnt_for_core_grps[cur_core_grp_id] += 1;
			}
			else
			{
				cur_core_grp_id = lpgemm_thread_attrs.tid_core_grp_id_list[ii];
			}
			tid_cnt_for_core_grps[lpgemm_thread_attrs.tid_core_grp_id_list[ii]] += 1;
		}

		for ( int ii = 0; ii < num_core_grps; ++ii )
		{
			if ( adj_tid_cnt_for_core_grps[ii] >= core_grp_loaded_thres )
			{
				core_grp_adj_tid_thres_cnt += 1;
				core_grp_adj_tid_cnt += 1;
			}
			else if ( adj_tid_cnt_for_core_grps[ii] > 0 )
			{
				core_grp_adj_tid_cnt += 1;
			}
			else if( tid_cnt_for_core_grps[ii] > 0)
			{
				core_grp_non_adj_tid_cnt += 1;
			}
		}

		if ( core_grp_adj_tid_cnt > ( 2 * core_grp_non_adj_tid_cnt ) )
		{
			lpgemm_thread_attrs.tid_distr_nearly_seq = TRUE;
		}

		if ( ( core_grp_adj_tid_thres_cnt > 0 ) &&
			 ( core_grp_adj_tid_thres_cnt >=
			   ( core_grp_adj_tid_cnt - core_grp_adj_tid_thres_cnt ) ) )
		{
			lpgemm_thread_attrs.tid_core_grp_load_high = TRUE;
		}
	}

err_handle:
	free( tid_cnt_for_core_grps );
	free( adj_tid_cnt_for_core_grps );

	if (thread_core_bind_list != NULL )
	{
		for ( int ii = 0; ii < nt_max; ++ii )
		{
			free( thread_core_bind_list[ii] );
		}
	}
	free( thread_core_bind_list );
}

#else

static void lpgemm_detect_thread_topo()
{}

#endif // BLIS_ENABLE_OPENMP

void lpgemm_load_thread_attrs()
{
	lpgemm_thread_attrs.tid_core_grp_id_list = NULL;
	lpgemm_thread_attrs.tid_cnt = 0;
	lpgemm_thread_attrs.openmp_enabled = FALSE;
	lpgemm_thread_attrs.tid_distr_nearly_seq = FALSE;
	lpgemm_thread_attrs.tid_core_grp_load_high = FALSE;

	lpgemm_detect_thread_topo();
}

void lpgemm_init_thread_attrs()
{
	bli_pthread_once
	(
	  &once_check_lpgemm_thread_topo_init,
	  lpgemm_load_thread_attrs
	);
}

// Should be called only after aocl_lpgemm_init_global_cntx.
lpgemm_thread_attrs_t* lpgemm_get_thread_attrs()
{
	return &lpgemm_thread_attrs;
}
