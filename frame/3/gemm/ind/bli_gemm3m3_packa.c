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

void bli_gemm3m3_packa
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	obj_t a_pack;

	// Make a copy of the context for each stage.
	cntx_t cntx_ro  = *cntx;
	cntx_t cntx_io  = *cntx;
	cntx_t cntx_rpi = *cntx;

	// -----------------------------------------------------

	// Initialize the context for the real-only stage.
	bli_gemm3m3_cntx_stage( 0, &cntx_ro );

	// Pack matrix the real-only part of A.
	bli_l3_packm
	(
	  a,
	  &a_pack,
	  &cntx_ro,
	  cntl,
	  thread
	);

	// Proceed with execution using packed matrix A.
	bli_gemm_int
	(
	  &BLIS_ONE,
	  &a_pack,
	  b,
	  &BLIS_ONE,
	  c,
	  cntx,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);

	// Only apply beta within the first of three subproblems.
	bli_obj_scalar_reset( c );

	// -----------------------------------------------------

	// Initialize the context for the imag-only stage.
	bli_gemm3m3_cntx_stage( 1, &cntx_io );

	// Pack matrix the imag-only part of A.
	bli_l3_packm
	(
	  a,
	  &a_pack,
	  &cntx_io,
	  cntl,
	  thread
	);

	// Proceed with execution using packed matrix A.
	bli_gemm_int
	(
	  &BLIS_ONE,
	  &a_pack,
	  b,
	  &BLIS_ONE,
	  c,
	  cntx,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);

	// -----------------------------------------------------

	// Initialize the context for the real+imag stage.
	bli_gemm3m3_cntx_stage( 2, &cntx_rpi );

	// Pack matrix the real+imag part of A.
	bli_l3_packm
	(
	  a,
	  &a_pack,
	  &cntx_rpi,
	  cntl,
	  thread
	);

	// Proceed with execution using packed matrix A.
	bli_gemm_int
	(
	  &BLIS_ONE,
	  &a_pack,
	  b,
	  &BLIS_ONE,
	  c,
	  cntx,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);

}

