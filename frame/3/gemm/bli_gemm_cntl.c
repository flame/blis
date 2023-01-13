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

cntl_t* bli_gemm_cntl_create
     (
       pool_t* pool,
       opid_t  family,
       pack_t  schema_a,
       pack_t  schema_b,
       void_fp ker
     )
{
	return bli_gemmbp_cntl_create( pool, family, schema_a, schema_b, ker );
}

// -----------------------------------------------------------------------------

cntl_t* bli_gemmbp_cntl_create
     (
       pool_t* pool,
       opid_t  family,
       pack_t  schema_a,
       pack_t  schema_b,
       void_fp ker
     )
{
	void_fp macro_kernel_fp;

	// Choose the default macrokernel based on the operation family...
	if      ( family == BLIS_GEMM )  macro_kernel_fp = bli_gemm_ker_var2;
	else if ( family == BLIS_GEMMT ) macro_kernel_fp =
	                                   #ifdef BLIS_ENABLE_JRIR_TLB
	                                   bli_gemmt_x_ker_var2b;
	                                   #else // ifdef ( _SLAB || _RR )
	                                   bli_gemmt_x_ker_var2;
	                                   #endif
	else if ( family == BLIS_TRMM )  macro_kernel_fp =
	                                   #ifdef BLIS_ENABLE_JRIR_TLB
	                                   bli_trmm_xx_ker_var2b;
	                                   #else // ifdef ( _SLAB || _RR )
	                                   bli_trmm_xx_ker_var2;
	                                   #endif
	else /* should never execute */  macro_kernel_fp = NULL;

	// ...unless a non-NULL kernel function pointer is passed in, in which
	// case we use that instead.
	if ( ker ) macro_kernel_fp = ker;

	// Create two nodes for the macro-kernel.
	cntl_t* gemm_cntl_bu_ke = bli_gemm_cntl_create_node
	(
	  pool,         // the thread's sba pool
	  family,       // the operation family
	  BLIS_MR,
	  NULL,         // variant function pointer not used
	  NULL          // no sub-node; this is the leaf of the tree.
	);

	cntl_t* gemm_cntl_bp_bu = bli_gemm_cntl_create_node
	(
	  pool,         // the thread's sba pool
	  family,
	  BLIS_NR,
	  macro_kernel_fp,
	  gemm_cntl_bu_ke
	);

	// Create a node for packing matrix A.
	cntl_t* gemm_cntl_packa = bli_packm_cntl_create_node
	(
	  pool,
	  bli_l3_packa, // pack the left-hand operand
	  BLIS_MR,
	  BLIS_KR,
	  FALSE,        // do NOT invert diagonal
	  FALSE,        // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_a,     // normally BLIS_PACKED_ROW_PANELS
	  BLIS_BUFFER_FOR_A_BLOCK,
	  gemm_cntl_bp_bu
	);

	// Create a node for partitioning the m dimension by MC.
	cntl_t* gemm_cntl_op_bp = bli_gemm_cntl_create_node
	(
	  pool,
	  family,
	  BLIS_MC,
	  bli_gemm_blk_var1,
	  gemm_cntl_packa
	);

	// Create a node for packing matrix B.
	cntl_t* gemm_cntl_packb = bli_packm_cntl_create_node
	(
	  pool,
	  bli_l3_packb, // pack the right-hand operand
	  BLIS_NR,
	  BLIS_KR,
	  FALSE,        // do NOT invert diagonal
	  FALSE,        // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_b,     // normally BLIS_PACKED_COL_PANELS
	  BLIS_BUFFER_FOR_B_PANEL,
	  gemm_cntl_op_bp
	);

	// Create a node for partitioning the k dimension by KC.
	cntl_t* gemm_cntl_mm_op = bli_gemm_cntl_create_node
	(
	  pool,
	  family,
	  BLIS_KC,
	  bli_gemm_blk_var3,
	  gemm_cntl_packb
	);

	// Create a node for partitioning the n dimension by NC.
	cntl_t* gemm_cntl_vl_mm = bli_gemm_cntl_create_node
	(
	  pool,
	  family,
	  BLIS_NC,
	  bli_gemm_blk_var2,
	  gemm_cntl_mm_op
	);

	return gemm_cntl_vl_mm;
}

// -----------------------------------------------------------------------------

void bli_gemm_cntl_free
     (
       pool_t* pool,
       cntl_t* cntl
     )
{
	bli_cntl_free( pool, cntl );
}

// -----------------------------------------------------------------------------

cntl_t* bli_gemm_cntl_create_node
     (
       pool_t* pool,
       opid_t  family,
       bszid_t bszid,
       void_fp var_func,
       cntl_t* sub_node
     )
{
	return bli_cntl_create_node( pool, family, bszid, var_func, NULL, sub_node );
}

