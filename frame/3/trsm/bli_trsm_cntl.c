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
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             pack_t       schema_a,
             pack_t       schema_b,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     )
{
	if ( bli_obj_is_triangular( a ) )
		return bli_trsm_l_cntl_init( a, b, c, schema_a, schema_b, cntx, cntl );
	else
		return bli_trsm_r_cntl_init( a, b, c, schema_a, schema_b, cntx, cntl );
}

void bli_trsm_l_cntl_init
     (
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             pack_t       schema_a,
             pack_t       schema_b,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     )
{
    // Set the default macrokernel.
	void_fp macro_kernel_p = bli_trsm_xx_ker_var2;

    const dir_t  direct   = bli_obj_is_lower( a ) ? BLIS_FWD : BLIS_BWD;
    const num_t  dt       = bli_obj_comp_dt( c );
    const dim_t  ir_bsize = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
    const dim_t  jr_bsize = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
    const dim_t  ic_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
    const dim_t  ic_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_MC, cntx );
    const dim_t  ic_mult  = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
          dim_t  pc_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
          dim_t  pc_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_KC, cntx );
    const dim_t  pc_mult  = 1;
    const dim_t  jc_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );
    const dim_t  jc_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_NC, cntx );
    const dim_t  jc_mult  = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );

    bli_l3_adjust_kc
    (
      BLIS_TRSM,
      a,
      b,
      &pc_alg,
      &pc_max,
      cntx
    );

	//
	// Create nodes for packing A and the macro-kernel (gemm branch).
	//

	bli_part_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  BLIS_MR,      // block size id
      ir_bsize,     // algorithmic block size
      ir_bsize,     // max block size
      ir_bsize,     // block size mult
      BLIS_FWD,     // partitioning direction
      FALSE,        // use weighted partitioning
	  NULL,         // no sub-node; this is the leaf of the tree.
      &cntl->part_ir_gemm
	);

	bli_part_cntl_init_node
	(
	  macro_kernel_p,
	  BLIS_NR,
      jr_bsize,
      jr_bsize,
      jr_bsize,
      BLIS_FWD,
      FALSE,
	  &cntl->part_ir_gemm.cntl,
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
	  &cntl->part_jr_gemm.cntl,
      &cntl->pack_a_gemm
	);

	//
	// Create nodes for packing A and the macro-kernel (trsm branch).
	//

	bli_part_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  BLIS_MR,
      ir_bsize,
      ir_bsize,
      ir_bsize,
      BLIS_FWD,
      FALSE,
	  NULL,         // no sub-node; this is the leaf of the tree.
      &cntl->part_ir_trsm
	);

	bli_part_cntl_init_node
	(
	  macro_kernel_p,
	  BLIS_NR,
      jr_bsize,
      jr_bsize,
      jr_bsize,
      BLIS_FWD,
      FALSE,
	  &cntl->part_ir_trsm.cntl,
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
	  &cntl->part_jr_trsm.cntl,
      &cntl->pack_a_trsm
	);

	// -------------------------------------------------------------------------

	// Create a node for partitioning the m dimension by MC.
	// NOTE: We attach the gemm sub-tree as the main branch.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var1,
	  BLIS_MC,
      ic_alg,
      ic_max,
      ic_mult,
      direct,
      FALSE,
	  &cntl->pack_a_gemm.cntl,
      &cntl->part_ic
	);

	// Attach the trsm sub-tree as the auxiliary "prenode" branch.
	bli_cntl_set_sub_prenode( &cntl->pack_a_trsm.cntl, &cntl->part_ic.cntl );

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
	  &cntl->part_ic.cntl,
      &cntl->pack_b
	);

	// Create a node for partitioning the k dimension by KC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var3,
	  BLIS_KC,
      pc_alg,
      pc_max,
      pc_mult,
      direct,
      FALSE,
	  &cntl->pack_b.cntl,
      &cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var2,
	  BLIS_NC,
      jc_alg,
      jc_max,
      jc_mult,
      BLIS_FWD,
      FALSE,
	  &cntl->part_pc.cntl,
      &cntl->part_jc
	);
}

void bli_trsm_r_cntl_init
     (
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             pack_t       schema_a,
             pack_t       schema_b,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     )
{
	// NOTE: trsm macrokernels are presently disabled for right-side execution.
    // Set the default macrokernel.
	void_fp macro_kernel_p = bli_trsm_xx_ker_var2;

	const dir_t  direct   = bli_obj_is_lower( b ) ? BLIS_BWD : BLIS_FWD;
    const num_t  dt       = bli_obj_comp_dt( c );
    const dim_t  ir_bsize = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
    const dim_t  jr_bsize = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
    const dim_t  ic_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
    const dim_t  ic_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_MC, cntx );
    const dim_t  ic_mult  = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); //note: different!
          dim_t  pc_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
          dim_t  pc_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_KC, cntx );
    const dim_t  pc_mult  = 1;
    const dim_t  jc_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );
    const dim_t  jc_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_NC, cntx );
    const dim_t  jc_mult  = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); //note: different!

    bli_l3_adjust_kc
    (
      BLIS_TRSM,
      a,
      b,
      &pc_alg,
      &pc_max,
      cntx
    );

	// Create two nodes for the macro-kernel.
	bli_part_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  BLIS_MR,      // block size id
      ir_bsize,     // algorithmic block size
      ir_bsize,     // max block size
      ir_bsize,     // block size mult
      BLIS_FWD,     // partitioning direction
      FALSE,        // use weighted partitioning
	  NULL,         // no sub-node; this is the leaf of the tree.
      &cntl->part_ir_trsm
	);

	bli_part_cntl_init_node
	(
	  macro_kernel_p,
	  BLIS_NR, // not used by macro-kernel, but needed for bli_thrinfo_rgrow()
      jr_bsize,
      jr_bsize,
      jr_bsize,
      BLIS_FWD,
      FALSE,
	  &cntl->part_ir_trsm.cntl,
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
	  &cntl->part_jr_trsm.cntl,
      &cntl->pack_a_trsm
	);

	// Create a node for partitioning the m dimension by MC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var1,
	  BLIS_MC,
      ic_alg,
      ic_max,
      ic_mult,
      BLIS_FWD,
      FALSE,
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
	  &cntl->part_ic.cntl,
      &cntl->pack_b
	);

	// Create a node for partitioning the k dimension by KC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var3,
	  BLIS_KC,
      pc_alg,
      pc_max,
      pc_mult,
      direct,
      FALSE,
	  &cntl->pack_b.cntl,
      &cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var2,
	  BLIS_NC,
      jc_alg,
      jc_max,
      jc_mult,
      direct,
      FALSE,
	  &cntl->part_pc.cntl,
      &cntl->part_jc
	);
}

