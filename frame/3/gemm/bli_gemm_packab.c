/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2022-23, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_gemm_packa
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);
	obj_t a_pack;

	// BY setting family id to BLIS_GEMM_MD, we indicate packing kernels
	// to scale alpha while packing.
	if(bli_obj_dt(c) != bli_obj_dt(b))
		bli_cntl_set_family(BLIS_GEMM_MD, cntl);

	// Pack matrix A according to the control tree node.
	bli_l3_packm
	(
	  a,
	  &a_pack,
	  cntx,
	  rntm,
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
	  rntm,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);
	
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}

// -----------------------------------------------------------------------------

void bli_gemm_packb
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5);

	obj_t b_pack;

	// BY setting family id to BLIS_GEMM_MD, we indicate packing kernels
	// to scale alpha while packing.
	if(bli_obj_dt(c) != bli_obj_dt(a))
		bli_cntl_set_family(BLIS_GEMM_MD, cntl);

	// Pack matrix B according to the control tree node.
	bli_l3_packm
	(
	  b,
	  &b_pack,
	  cntx,
	  rntm,
	  cntl,
	  thread
	);
	// Once packing of B matrix is done, fall back to GEMM execution.
	if(bli_obj_dt(c) != bli_obj_dt(a))
		bli_cntl_set_family(BLIS_GEMM, cntl);


	// Proceed with execution using packed matrix B.
	bli_gemm_int
	(
	  &BLIS_ONE,
	  a,
	  &b_pack,
	  &BLIS_ONE,
	  c,
	  cntx,
	  rntm,
	  bli_cntl_sub_node( cntl ),
	  bli_thrinfo_sub_node( thread )
	);
	
	AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5);
}

