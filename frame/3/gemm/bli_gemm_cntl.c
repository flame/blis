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

void bli_gemm_cntl_init
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             pack_t       schema_a,
             pack_t       schema_b,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl
     )
{
	bli_gemmbp_cntl_init( family, a, b, c, schema_a, schema_b, cntx, cntl );
}

// -----------------------------------------------------------------------------

void bli_gemmbp_cntl_init
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             pack_t       schema_a,
             pack_t       schema_b,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl
     )
{
	void_fp macro_kernel_fp;

	// Choose the default macrokernel based on the operation family.
	if      ( family == BLIS_GEMM ||
              family == BLIS_HEMM ||
              family == BLIS_SYMM ) macro_kernel_fp = bli_gemm_ker_var2;
	else if ( family == BLIS_GEMMT ) macro_kernel_fp = bli_gemmt_x_ker_var2;
	else if ( family == BLIS_TRMM ||
	          family == BLIS_TRMM3 ) macro_kernel_fp = bli_trmm_xx_ker_var2;
	else /* should never execute */ macro_kernel_fp = NULL;

    const num_t dt      = bli_obj_comp_dt( c );
    const dim_t ir_mult = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
    const dim_t jr_mult = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
    const dim_t ic_mult = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
    const dim_t jc_mult = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

	// Create two nodes for the macro-kernel.
	bli_part_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  BLIS_MR,      // block side id
      ir_mult,      // block side mult
      FALSE,        // use weighted partitioning
	  NULL,         // no sub-node; this is the leaf of the tree.
      &cntl->part_ir
	);

	bli_part_cntl_init_node
	(
	  macro_kernel_fp,
	  BLIS_NR,
      jr_mult,
      FALSE,
      &cntl->part_ir.cntl,
      &cntl->part_jr
	);

	// Create a node for packing matrix A.
	bli_packm_cntl_init_node
	(
	  bli_l3_packa, // pack the left-hand operand
	  BLIS_MR,
	  BLIS_KR,
	  FALSE,        // do NOT invert diagonal
	  FALSE,        // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_a,     // normally BLIS_PACKED_ROW_PANELS
	  BLIS_BUFFER_FOR_A_BLOCK,
      &cntl->part_jr.cntl,
      &cntl->pack_a
	);

	// Create a node for partitioning the m dimension by MC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var1,
	  BLIS_MC,
      ic_mult,
      bli_obj_is_triangular( a ) || bli_obj_is_triangular( c ),
      &cntl->pack_a.cntl,
      &cntl->part_ic
	);

	// Create a node for packing matrix B.
	bli_packm_cntl_init_node
	(
	  bli_l3_packb, // pack the right-hand operand
	  BLIS_NR,
	  BLIS_KR,
	  FALSE,        // do NOT invert diagonal
	  FALSE,        // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_b,     // normally BLIS_PACKED_COL_PANELS
	  BLIS_BUFFER_FOR_B_PANEL,
      &cntl->part_ic.cntl,
      &cntl->pack_b
	);

	// Create a node for partitioning the k dimension by KC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var3,
	  BLIS_KC,
      1,
      FALSE,
      &cntl->pack_b.cntl,
      &cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var2,
	  BLIS_NC,
      jc_mult,
      bli_obj_is_triangular( b ) || bli_obj_is_triangular( c ),
      &cntl->part_pc.cntl,
      &cntl->part_jc
	);

    bli_cntl_mark_family( family, bli_gemm_cntl_root( cntl ) );
}

