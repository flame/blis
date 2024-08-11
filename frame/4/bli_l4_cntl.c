/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin
   Copyright (C) 2022, Oracle Labs, Oracle Corporation

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

#ifdef BLIS_ENABLE_LEVEL4

// -----------------------------------------------------------------------------

l4_cntl_t* bli_l4_cntl_create_node
     (
       pool_t*    pool,
       bszid_t    bszid,
       dim_t      scale_num,
       dim_t      scale_den,
       dim_t      depth,
       void_fp    var_func,
       l4_cntl_t* sub_node
     )
{
	l4_params_t* params = bli_sba_acquire( pool, sizeof( l4_params_t ) );

	params->size      = sizeof( l4_params_t );
	params->scale_num = scale_num;
	params->scale_den = scale_den;
	params->depth     = depth;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l4_cntl_create_node(): " );
	#endif

	// Allocate the cntl_t struct.
	l4_cntl_t* cntl = bli_sba_acquire( pool, sizeof( l4_cntl_t ) );

	bli_l4_cntl_set_bszid( bszid, cntl );
	bli_l4_cntl_set_var_func( var_func, cntl );
	bli_l4_cntl_set_params( params, cntl );
	bli_l4_cntl_set_sub_node( sub_node, cntl );

	return cntl;
}

// -----------------------------------------------------------------------------

void bli_l4_cntl_free
     (
       pool_t*    pool,
       l4_cntl_t* cntl
     )
{
	// Base case: simply return when asked to free NULL nodes.
	if ( cntl == NULL ) return;

	//cntl_t* cntl_sub_prenode = bli_cntl_sub_prenode( cntl );
	l4_cntl_t* cntl_sub_node = bli_l4_cntl_sub_node( cntl );
	void*      cntl_params   = bli_l4_cntl_params( cntl );

	// Only recurse into the child node if it exists.
	if ( cntl_sub_node != NULL )
	{
		// Recursively free all memory associated with the sub-node and its
		// children.
		bli_l4_cntl_free( pool, cntl_sub_node );
	}

	// Free the current node's params field, if it is non-NULL.
	if ( cntl_params != NULL )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_cntl_free_w_thrinfo(): " );
		#endif

		bli_sba_release( pool, cntl_params );
	}

	// Free the current node.
	bli_l4_cntl_free_node( pool, cntl );
}

// -----------------------------------------------------------------------------

void bli_l4_cntl_free_node
     ( 
       pool_t*    pool,
       l4_cntl_t* cntl
     )
{       
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_l4_cntl_free_node(): " );
	#endif
    
	bli_sba_release( pool, cntl );
}

#endif
