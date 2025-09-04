/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_UTILS_H
#define LPGEMM_UTILS_H

#include "lpgemm_types.h"

// Users of this API needs to free the allocated memory on their own.
dim_t get_64byte_aligned_memory
     (
       void**  original_memory,
       void**  aligned_memory,
       int64_t allocate_size
     );

lpgemm_obj_t* alloc_unpack_tag_lpgemm_obj_t_u8s8s32
     (
       dim_t         length,
       dim_t         width,
       dim_t         stride,
       dim_t         elem_size,
       AOCL_STOR_TAG stor_scheme
     );

lpgemm_obj_t* alloc_pack_tag_lpgemm_obj_t_u8s8s32
     (
       dim_t         length,
       dim_t         width,
       dim_t         stride,
       dim_t         elem_size,
       AOCL_STOR_TAG stor_scheme
     );

lpgemm_obj_t* alloc_reorder_tag_lpgemm_obj_t_u8s8s32
     (
       dim_t         length,
       dim_t         width,
       dim_t         stride,
       dim_t         elem_size,
       AOCL_STOR_TAG stor_scheme
     );

void dealloc_lpgemm_obj_t_u8s8s32( lpgemm_obj_t* obj );

BLIS_INLINE void bli_param_map_char_to_lpmtag
     (
       char mtag,
       AOCL_MEMORY_TAG* lp_mtag
     )
{
        if      ( mtag == 'n' || mtag == 'N' ) *lp_mtag = UNPACKED;
        else if ( mtag == 'p' || mtag == 'P' ) *lp_mtag = PACK;
        else if ( mtag == 'r' || mtag == 'R' ) *lp_mtag = REORDERED;
        else
        {
                *lp_mtag = UNPACKED;
        }
}

BLIS_INLINE void bli_param_map_char_to_lpmat_type
     (
       const char mtag,
       AOCL_MATRIX_TYPE* lp_mat_type
     )
{
        if      ( mtag == 'a' || mtag == 'A' ) *lp_mat_type = A_MATRIX;
        else if ( mtag == 'b' || mtag == 'B' ) *lp_mat_type = B_MATRIX;
        else if ( mtag == 'w' || mtag == 'W' ) *lp_mat_type = AWQ_B_MATRIX;
        else
        {
                *lp_mat_type = B_MATRIX;
        }
}

BLIS_INLINE dim_t make_multiple_of_n( dim_t k, dim_t n )
{
	if ( n <= 0 )
	{
		return 0;
	}

	return ( ( ( k + n - 1 ) / n ) * n );
}

BLIS_INLINE void lpgemm_alloc_mem_panel
     (
       dim_t     size_req,
       packbuf_t buf_type,
       mem_t*    mem,
       rntm_t*   rntm_l
     )
{
	if ( bli_mem_is_unalloc( mem ) )
	{
		bli_pba_acquire_m
		(
		  rntm_l,
		  size_req,
		  buf_type,
		  mem
		);
	}
	else
	{
		siz_t mem_size = bli_mem_size( mem );
		if ( mem_size < size_req )
		{
			bli_pba_release( rntm_l, mem );
			bli_pba_acquire_m
			(
			  rntm_l,
			  size_req,
			  buf_type,
			  mem
			);
		}
	}
}

BLIS_INLINE dim_t get_Bpanel_width_for_kdim_traversal
     (
       dim_t jc,
       dim_t n,
       dim_t NC,
       dim_t NR
     )
{
	dim_t n_mod_NR = n % NR;
	dim_t n_sub_updated = NC;

	if ( ( n % NC ) != 0 )
	{
		// Only applicable to final NC part of jc loop where jc + remaining 
		// elements is less than NC; or when n < NC in which case panel width
		// is atmost n.
		dim_t n_last_loop = ( n / NC ) * NC;
		if ( jc >= n_last_loop )
		{
			n_sub_updated = n - n_last_loop;
			if ( n_mod_NR != 0 )
			{
				n_sub_updated += ( NR - n_mod_NR );
			}
		}
	}

	return n_sub_updated;
}

BLIS_INLINE void get_B_panel_reordered_start_offset_width
     (
       dim_t  jc,
       dim_t  n,
       dim_t  NC,
       dim_t  NR,
       dim_t* panel_start,
       dim_t* panel_offset,
       dim_t* panel_width,
       dim_t* panel_width_kdim_trav
     )
{
	// Since n dimension is split across threads in units of NR blocks,
	// it could happen that B matrix chunk for a thread may be part of
	// two separate NCxKC panels. In this case nc0 is updated such that
	// the jr loop only accesses the remaining portion of current NCxKC
	// panel, with the next jc iteration taking care of the other panel.
	// This ensures that jr loop does not cross panel boundaries.
	( *panel_start ) = ( jc / NC ) * NC;
	( *panel_offset ) = jc - ( *panel_start );

	// Check if jc + current_panel_width (nc0) crosses panel boundaries.
	if ( ( jc + ( *panel_width ) ) > ( ( *panel_start ) + NC ) )
	{
		( *panel_width ) = NC - ( *panel_offset );
	}

	( *panel_width_kdim_trav ) = get_Bpanel_width_for_kdim_traversal
								 (
								   jc, n, NC, NR
								 );
}

BLIS_INLINE void adjust_B_panel_reordered_jc( dim_t* jc, dim_t panel_start )
{
	// Since n dimension is split across threads in units of NR blocks,
	// it could happen that B matrix chunk for a thread may be part of
	// two separate NCxKC panels. In this case jc is reset to immediate
	// previous panel offset so that in the next iteration, the
	// following panel belonging to the B chunk is accessed. This
	// ensures that jr loop does not cross panel boundaries.
	( *jc ) = panel_start;
}

static inline bool is_single_thread( rntm_t* rntm_g )
{
	bool is_st = FALSE;

	dim_t n_threads = bli_rntm_num_threads( rntm_g );
	dim_t jc_ways = bli_rntm_jc_ways( rntm_g );
	dim_t ic_ways = bli_rntm_ic_ways( rntm_g );

	ic_ways = ( ic_ways > 0 ) ? ic_ways : 1;
	jc_ways = ( jc_ways > 0 ) ? jc_ways : 1;

	if ( ( n_threads == 1 ) ||
		 ( ( ic_ways * jc_ways ) == 1 ) )
	{
		is_st = TRUE;
	}

	return is_st;
}

#endif //LPGEMM_UTILS_H
