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

dim_t bli_trsm_determine_kc_f( dim_t    i,
                               dim_t    dim,
                               obj_t*   obj,
                               blksz_t* bsize )
{
	num_t dt;
	dim_t mr;
	dim_t b_alg, b_max;
	dim_t b_use;

	// We assume that this function is being called from an algorithm that
	// is moving "forward" (ie: top to bottom, left to right, top-left
	// to bottom-right).

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize maximum values from the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_for_type( dt, bsize );
	b_max = bli_blksz_max_for_type( dt, bsize );

	// Nudge the default and maximum kc blocksizes up to the nearest
	// multiple of MR. We always use MR (rather than sometimes use NR
	// because even when the triangle is on the right, packing of that
	// matrix uses MR, since only left-side trsm micro-kernels are
	// supported.
	mr    = bli_info_get_default_mr( dt );
	b_alg = bli_align_dim_to_mult( b_alg, mr );
	b_max = bli_align_dim_to_mult( b_max, mr );

	b_use = bli_determine_blocksize_f_sub( i, dim, b_alg, b_max );

	return b_use;
}


dim_t bli_trsm_determine_kc_b( dim_t    i,
                               dim_t    dim,
                               obj_t*   obj,
                               blksz_t* bsize )
{
	num_t dt;
	dim_t mr;
	dim_t b_alg, b_max;
	dim_t b_use;

	// We assume that this function is being called from an algorithm that
	// is moving "backward" (ie: bottom to top, right to left, bottom-right
	// to top-left).

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize maximum values from the blksz_t object.
	dt    = bli_obj_execution_datatype( *obj );
	b_alg = bli_blksz_for_type( dt, bsize );
	b_max = bli_blksz_max_for_type( dt, bsize );

	// Nudge the default and maximum kc blocksizes up to the nearest
	// multiple of MR. We always use MR (rather than sometimes use NR
	// because even when the triangle is on the right, packing of that
	// matrix uses MR, since only left-side trsm micro-kernels are
	// supported.
	mr    = bli_info_get_default_mr( dt );
	b_alg = bli_align_dim_to_mult( b_alg, mr );
	b_max = bli_align_dim_to_mult( b_max, mr );

	b_use = bli_determine_blocksize_b_sub( i, dim, b_alg, b_max );

	return b_use;
}

