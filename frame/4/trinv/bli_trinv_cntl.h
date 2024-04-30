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

#ifndef BLIS_TRINV_CNTL_H
#define BLIS_TRINV_CNTL_H

typedef struct
{
	uint64_t size;
	dim_t    scale_num;
	dim_t    scale_den;
	dim_t    depth;
} trinv_params_t;

BLIS_INLINE dim_t bli_trinv_params_scale_num( const trinv_params_t* params )
{
	return params->scale_num;
}

BLIS_INLINE dim_t bli_trinv_params_scale_den( const trinv_params_t* params )
{
	return params->scale_den;
}

BLIS_INLINE dim_t bli_trinv_params_depth( const trinv_params_t* params )
{
	return params->depth;
}

// -----------------------------------------------------------------------------

cntl_t* bli_trinv_cntl_create( uplo_t uplo, diag_t diag, pool_t* pool );

// -----------------------------------------------------------------------------

void bli_trinv_cntl_free( pool_t* pool, cntl_t* cntl );

// -----------------------------------------------------------------------------

cntl_t* bli_trinv_cntl_create_node
     (
       pool_t* pool,
       bszid_t bszid,
       dim_t   scale_num,
       dim_t   scale_den,
       dim_t   depth,
       void_fp var_func,
       cntl_t* sub_node
     );

#endif
