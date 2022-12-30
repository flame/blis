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

void bli_trsm_cntl_init
     (
       side_t       side,
       pack_t       schema_a,
       pack_t       schema_b,
       trsm_cntl_t* cntl
     )
{
	if ( bli_is_left( side ) )
		return bli_trsm_l_cntl_init( schema_a, schema_b, cntl );
	else
		return bli_trsm_r_cntl_init( schema_a, schema_b, cntl );
}

void bli_trsm_l_cntl_init
     (
       pack_t       schema_a,
       pack_t       schema_b,
       trsm_cntl_t* cntl
     )
{
    // Set the default macrokernel.
	void_fp macro_kernel_p = bli_trsm_xx_ker_var2;

	const opid_t family = BLIS_TRSM;

	//
	// Create nodes for packing A and the macro-kernel (gemm branch).
	//

	bli_cntl_init_node
	(
	  family,       // the operation family
	  BLIS_MR,
	  NULL,         // variant function pointer not used
	  NULL,         // no sub-node; this is the leaf of the tree.
      &cntl->part_ir_gemm
	);

	bli_cntl_init_node
	(
	  family,
	  BLIS_NR,
	  macro_kernel_p,
	  &cntl->part_ir_gemm,
      &cntl->part_jr_gemm
	);

	// Create a node for packing matrix A.
	bli_packm_cntl_init_node
	(
	  bli_l3_packa, // trsm operation's packm function for A.
	  BLIS_MR,
	  BLIS_MR,
	  FALSE,        // do NOT invert diagonal
	  TRUE,         // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_a,     // normally BLIS_PACKED_ROW_PANELS
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->part_jr_gemm,
      &cntl->pack_a_gemm
	);

	//
	// Create nodes for packing A and the macro-kernel (trsm branch).
	//

	bli_cntl_init_node
	(
	  family,       // the operation family
	  BLIS_MR,
	  NULL,         // variant function pointer not used
	  NULL,         // no sub-node; this is the leaf of the tree.
      &cntl->part_ir_trsm
	);

	bli_cntl_init_node
	(
	  family,
	  BLIS_NR,
	  macro_kernel_p,
	  &cntl->part_ir_trsm,
      &cntl->part_jr_trsm
	);

	// Create a node for packing matrix A.
	bli_packm_cntl_init_node
	(
	  bli_l3_packa, // trsm operation's packm function for A.
	  BLIS_MR,
	  BLIS_MR,
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	  TRUE,         // invert diagonal
#else
	  FALSE,        // do NOT invert diagonal
#endif
	  TRUE,         // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_a,     // normally BLIS_PACKED_ROW_PANELS
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->part_jr_trsm,
      &cntl->pack_a_trsm
	);

	// -------------------------------------------------------------------------

	// Create a node for partitioning the m dimension by MC.
	// NOTE: We attach the gemm sub-tree as the main branch.
	bli_cntl_init_node
	(
	  family,
	  BLIS_MC,
	  bli_trsm_blk_var1,
	  &cntl->pack_a_gemm.cntl,
      &cntl->part_ic
	);

	// Attach the trsm sub-tree as the auxiliary "prenode" branch.
	bli_cntl_set_sub_prenode( &cntl->pack_a_trsm.cntl, &cntl->part_ic );

	// -------------------------------------------------------------------------

	// Create a node for packing matrix B.
	bli_packm_cntl_init_node
	(
	  bli_l3_packb,
	  BLIS_NR,
	  BLIS_MR,
	  FALSE,        // do NOT invert diagonal
	  FALSE,        // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_b,     // normally BLIS_PACKED_COL_PANELS
	  BLIS_BUFFER_FOR_B_PANEL,
	  &cntl->part_ic,
      &cntl->pack_b
	);

	// Create a node for partitioning the k dimension by KC.
	bli_cntl_init_node
	(
	  family,
	  BLIS_KC,
	  bli_trsm_blk_var3,
	  &cntl->pack_b.cntl,
      &cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_cntl_init_node
	(
	  family,
	  BLIS_NC,
	  bli_trsm_blk_var2,
	  &cntl->part_pc,
      &cntl->part_jc
	);
}

void bli_trsm_r_cntl_init
     (
       pack_t       schema_a,
       pack_t       schema_b,
       trsm_cntl_t* cntl
     )
{
	// NOTE: trsm macrokernels are presently disabled for right-side execution.
    // Set the default macrokernel.
	void_fp macro_kernel_p = bli_trsm_xx_ker_var2;

	const opid_t family = BLIS_TRSM;

	// Create two nodes for the macro-kernel.
	bli_cntl_init_node
	(
	  family,
	  BLIS_MR, // needed for bli_thrinfo_rgrow()
	  NULL,    // variant function pointer not used
	  NULL,    // no sub-node; this is the leaf of the tree.
      &cntl->part_ir_trsm
	);

	bli_cntl_init_node
	(
	  family,
	  BLIS_NR, // not used by macro-kernel, but needed for bli_thrinfo_rgrow()
	  macro_kernel_p,
	  &cntl->part_ir_trsm,
      &cntl->part_jr_trsm
	);

	// Create a node for packing matrix A.
	bli_packm_cntl_init_node
	(
	  bli_l3_packa,
	  BLIS_NR,
	  BLIS_MR,
	  FALSE,   // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  schema_a, // normally BLIS_PACKED_ROW_PANELS
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->part_jr_trsm,
      &cntl->pack_a_trsm
	);

	// Create a node for partitioning the m dimension by MC.
	bli_cntl_init_node
	(
	  family,
	  BLIS_MC,
	  bli_trsm_blk_var1,
	  &cntl->pack_a_trsm.cntl,
      &cntl->part_ic
	);

	// Create a node for packing matrix B.
	bli_packm_cntl_init_node
	(
	  bli_l3_packb,
	  BLIS_MR,
	  BLIS_MR,
	  TRUE,    // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  TRUE,    // reverse iteration if lower?
	  schema_b, // normally BLIS_PACKED_COL_PANELS
	  BLIS_BUFFER_FOR_B_PANEL,
	  &cntl->part_ic,
      &cntl->pack_b
	);

	// Create a node for partitioning the k dimension by KC.
	bli_cntl_init_node
	(
	  family,
	  BLIS_KC,
	  bli_trsm_blk_var3,
	  &cntl->pack_b.cntl,
      &cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_cntl_init_node
	(
	  family,
	  BLIS_NC,
	  bli_trsm_blk_var2,
	  &cntl->part_pc,
      &cntl->part_jc
	);
}

