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

    const bool  a_lo_tri = bli_obj_is_triangular( a ) && bli_obj_is_lower( a );
    const bool  b_up_tri = bli_obj_is_triangular( b ) && bli_obj_is_upper( b );
    const bool  trmm_r   = family == BLIS_TRMM && bli_obj_is_triangular( b );
    const num_t dt       = bli_obj_comp_dt( c );
    const dim_t ir_bsize = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
    const dim_t jr_bsize = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
    const dim_t ic_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
    const dim_t ic_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_MC, cntx );
    const dim_t ic_mult  = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );
    const dir_t ic_dir   = a_lo_tri ? BLIS_BWD : BLIS_FWD;
          dim_t pc_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
          dim_t pc_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_KC, cntx );
    const dim_t pc_mult  = 1;
    const dir_t pc_dir   = a_lo_tri || b_up_tri ? BLIS_BWD : BLIS_FWD;
    const dim_t jc_alg   = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );
    const dim_t jc_max   = bli_cntx_get_blksz_max_dt( dt, BLIS_NC, cntx );
    const dim_t jc_mult  = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
    const dir_t jc_dir   = b_up_tri ? BLIS_BWD : BLIS_FWD;

    bli_l3_adjust_kc
    (
      family,
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
      ir_bsize,     // algorithmic block size
      ir_bsize,     // max block size
      ir_bsize,     // block size mult
      BLIS_FWD,     // partioning direction
      FALSE,        // use weighted partitioning
      &cntl->part_ir
	);

	bli_part_cntl_init_node
	(
	  macro_kernel_fp,
      jr_bsize,
      jr_bsize,
      jr_bsize,
      BLIS_FWD,
      FALSE,
      &cntl->part_jr
	);
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NR,
      &cntl->part_ir.cntl,
      &cntl->part_jr.cntl
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
      &cntl->pack_a
	);
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NONE,
      &cntl->part_jr.cntl,
      &cntl->pack_a.cntl
    );

	// Create a node for partitioning the m dimension by MC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var1,
      ic_alg,
      ic_max,
      ic_mult,
      ic_dir,
      bli_obj_is_triangular( a ) || bli_obj_is_upper_or_lower( c ),
      &cntl->part_ic
	);
    bli_cntl_attach_sub_node
    (
      trmm_r ? BLIS_THREAD_MC | BLIS_THREAD_NC : BLIS_THREAD_MC,
      &cntl->pack_a.cntl,
      &cntl->part_ic.cntl
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
      &cntl->pack_b
	);
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NONE,
      &cntl->part_ic.cntl,
      &cntl->pack_b.cntl
    );

	// Create a node for partitioning the k dimension by KC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var3,
      pc_alg,
      pc_max,
      pc_mult,
      pc_dir,
      FALSE,
      &cntl->part_pc
	);
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_KC,
      &cntl->pack_b.cntl,
      &cntl->part_pc.cntl
    );

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var2,
      jc_alg,
      jc_max,
      jc_mult,
      jc_dir,
      bli_obj_is_triangular( b ) || bli_obj_is_upper_or_lower( c ),
      &cntl->part_jc
	);
    bli_cntl_attach_sub_node
    (
      trmm_r ? BLIS_THREAD_NONE : BLIS_THREAD_NC,
      &cntl->part_pc.cntl,
      &cntl->part_jc.cntl
    );
}

