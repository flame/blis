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

dim_t bli_trinv_determine_blocksize
     (
             dim_t   i,
             dim_t   dim,
       const obj_t*  obj,
       const cntx_t* cntx,
             cntl_t* cntl
     )
{
	// Query the numerator and denomenator to use to scale the blocksizes
	// that we query from the context.
	const trinv_params_t* params    = bli_cntl_params( cntl );
	const dim_t           scale_num = bli_trinv_params_scale_num( params );
	const dim_t           scale_den = bli_trinv_params_scale_den( params );
	const bszid_t         bszid     = bli_cntl_bszid( cntl );

	// Extract the execution datatype and use it to query the corresponding
	// blocksize and blocksize maximum values from the blksz_t object.
	const num_t    dt     = bli_obj_exec_dt( obj );
	const blksz_t* bsize  = bli_cntx_get_blksz( bszid, cntx );
	const dim_t    b_def0 = bli_blksz_get_def( dt, bsize );
	const dim_t    b_max0 = bli_blksz_get_max( dt, bsize );

	// Scale the queried blocksizes by the scalars.
	const dim_t    b_def  = ( b_def0 * scale_num ) / scale_den;
	const dim_t    b_max  = ( b_max0 * scale_num ) / scale_den;

	// Compute how much of the matrix dimension is left, including the
	// chunk that will correspond to the blocksize we are computing now.
	const dim_t dim_left_now = dim - i;

	// If the dimension currently remaining is less than the maximum
	// blocksize, use it instead of the default blocksize b_def.
	// Otherwise, use b_def.
	dim_t b_now;
	if ( dim_left_now <= b_max ) b_now = dim_left_now;
	else                         b_now = b_def;

	return b_now;
}

#endif
