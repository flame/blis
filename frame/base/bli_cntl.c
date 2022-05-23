/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

void bli_cntl_initialize_node
     (
             cntl_t* cntl,
             opid_t  family,
             bszid_t bszid,
             void_fp var_func,
       const void*   params,
             cntl_t* sub_prenode,
             cntl_t* sub_node
     )
{
	bli_cntl_set_family( family, cntl );
    bli_cntl_set_part( bszid, cntl );
	bli_cntl_set_var_func( var_func, cntl );
	bli_cntl_set_params( params, cntl );
	bli_cntl_set_sub_prenode( sub_prenode, cntl );
	bli_cntl_set_sub_node( sub_node, cntl );
}

// -----------------------------------------------------------------------------

void bli_cntl_mark_family
     (
       opid_t  family,
       cntl_t* cntl
     )
{
	// This function sets the family field of all cntl tree nodes that are
	// children of cntl. It's used by bli_l3_cntl_create_if() after making
	// a copy of a user-given cntl tree, if the user provided one, to mark
	// the operation family, which is used to determine appropriate behavior
	// by various functions when executing the blocked variants.

	// Set the family of the root node.
	bli_cntl_set_family( family, cntl );

	// Recursively set the family field of the sub-tree rooted at the sub-node,
	// if it exists.
	if ( bli_cntl_sub_prenode( cntl ) != NULL )
	{
		bli_cntl_mark_family( family, bli_cntl_sub_prenode( cntl ) );
	}

	// Recursively set the family field of the sub-tree rooted at the prenode,
	// if it exists.
	if ( bli_cntl_sub_node( cntl ) != NULL )
	{
		bli_cntl_mark_family( family, bli_cntl_sub_node( cntl ) );
	}
}

// -----------------------------------------------------------------------------

dim_t bli_cntl_calc_num_threads_in
     (
       const rntm_t* rntm,
       const cntl_t* cntl
     )
{
	dim_t n_threads_in = 1;

	for ( ; cntl != NULL; cntl = bli_cntl_sub_node( cntl ) )
	{
		bszid_t bszid = bli_cntl_part( cntl );
		dim_t   cur_way;

		// We assume bszid is in {NC,KC,MC,NR,MR,KR} if it is not
		// BLIS_NO_PART.
		if ( bszid != BLIS_NO_PART )
			cur_way = bli_rntm_ways_for( bszid, rntm );
		else
			cur_way = 1;

		n_threads_in *= cur_way;
	}

	return n_threads_in;
}

