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

#include "bli_cntx.h"
#include "bli_type_defs.h"
#include "blis.h"

cntl_t* bli_gemm_cntl_create
     (
             goto_cntl_t* cntl,
             opid_t       family,
             num_t        dt_comp,
             obj_t*       a,
             obj_t*       b,
             obj_t*       c,
       const cntx_t*      cntx
     )
{
	      void_fp         ker_fp       = bli_obj_ker_fn( c );
	      void_fp         packa_fp     = bli_obj_pack_fn( a );
	      void_fp         packb_fp     = bli_obj_pack_fn( b );
    const gemm_params_t*  ker_params   = bli_obj_ker_params( c );
    const packm_params_t* packa_params = bli_obj_pack_params( a );
    const packm_params_t* packb_params = bli_obj_pack_params( b );

	// Choose the default macrokernels based on the operation family unless a
    // non-NULL kernel function pointer is passed in, in which case we use that instead.
    if ( ker_fp == NULL )
    {
    	if      ( family == BLIS_GEMM  ) ker_fp = bli_gemm_ker_var2;
    	else if ( family == BLIS_GEMMT ) ker_fp = bli_gemmt_x_ker_var2;
    	else if ( family == BLIS_TRMM  ) ker_fp = bli_trmm_xx_ker_var2;
    }

    if ( packa_fp == NULL )
    {
    	packa_fp = bli_l3_packa;
    }

    if ( packb_fp == NULL )
    {
    	packb_fp = bli_l3_packb;
    }

    cntl->ker_params.params = ker_params;
    cntl->ker_params.dt_comp = dt_comp;
    cntl->ker_params.ukr_row_pref = bli_cntx_get_ukr_prefs_dt( dt_comp, BLIS_GEMM_UKR_ROW_PREF, cntx);

    cntl->mc.blksz     = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MC, cntx );
    cntl->nc.blksz     = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NC, cntx );
    cntl->kc.blksz     = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KC, cntx );
    cntl->mc.blksz_max = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MC, cntx );
    cntl->nc.blksz_max = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NC, cntx );
    cntl->kc.blksz_max = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_KC, cntx );
    cntl->mc.bmult     = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MR, cntx );
    cntl->nc.bmult     = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NR, cntx );
    cntl->kc.bmult     = 1;

    if ( ker_params != NULL )
    {
        cntl->ker_params.ukr_row_pref = bli_mbool_get_dt( dt_comp, &ker_params->ukr_row_pref );

        cntl->mc.bmult = bli_blksz_get_def( dt_comp, &ker_params->mr );
        cntl->nc.bmult = bli_blksz_get_def( dt_comp, &ker_params->nr );
    }

    cntl->mc.blksz     = bli_align_dim_to_mult( cntl->mc.blksz, cntl->mc.bmult );
    cntl->nc.blksz     = bli_align_dim_to_mult( cntl->nc.blksz, cntl->nc.bmult );
    cntl->mc.blksz_max = bli_align_dim_to_mult( cntl->mc.blksz_max, cntl->mc.bmult );
    cntl->nc.blksz_max = bli_align_dim_to_mult( cntl->nc.blksz_max, cntl->nc.bmult );

    bli_l3_adjust_kc( a, b, cntl->mc.bmult, cntl->nc.bmult, &cntl->kc.blksz, &cntl->kc.blksz_max, family );

    cntl->packa_params.params = packa_params;
    cntl->packa_params.does_invert_diag = FALSE;
    cntl->packa_params.rev_iter_if_upper = FALSE;
    cntl->packa_params.rev_iter_if_lower = FALSE;
    cntl->packa_params.pack_schema = BLIS_PACKED_ROW_PANELS;
    cntl->packa_params.pack_buf_type = BLIS_BUFFER_FOR_A_BLOCK;

    cntl->packb_params.params = packb_params;
    cntl->packb_params.does_invert_diag = FALSE;
    cntl->packb_params.rev_iter_if_upper = FALSE;
    cntl->packb_params.rev_iter_if_lower = FALSE;
    cntl->packb_params.pack_schema = BLIS_PACKED_COL_PANELS;
    cntl->packb_params.pack_buf_type = BLIS_BUFFER_FOR_B_PANEL;

	// Create two nodes for the macro-kernel.
    bli_cntl_initialize_node
    (
      &cntl->loop1,
      family,    // the operation family
      BLIS_MR,   // used for thread partitioning
      NULL,      // variant function pointer not used
      NULL,      // not used
      NULL,      // no sub-prenode; this is the leaf of the tree.
      NULL       // no sub-node; this is the leaf of the tree.
    );

    bli_cntl_initialize_node
    (
      &cntl->loop2,
	  family,
      BLIS_NR,
	  ker_fp,
	  &cntl->ker_params,
      NULL,
      &cntl->loop1
	);

	// Create a node for packing matrix A.
    bli_cntl_initialize_node
    (
      &cntl->packa,
	  family,
      BLIS_NO_PART,
	  packa_fp,
	  &cntl->packa_params,
      NULL,
      &cntl->loop2
	);

	// Create a node for partitioning the m dimension by MC.
    bli_cntl_initialize_node
    (
      &cntl->loop3,
	  family,
      BLIS_MC,
	  bli_gemm_blk_var1,
      &cntl->mc,
      NULL,
	  &cntl->packa
	);

	// Create a node for packing matrix B.
    bli_cntl_initialize_node
    (
      &cntl->packb,
	  family,
      BLIS_NO_PART,
	  packb_fp,
	  &cntl->packb_params,
      NULL,
      &cntl->loop3
	);

	// Create a node for partitioning the k dimension by KC.
    bli_cntl_initialize_node
    (
      &cntl->loop4,
	  family,
      BLIS_KC,
	  bli_gemm_blk_var3,
      &cntl->kc,
      NULL,
	  &cntl->packb
	);

	// Create a node for partitioning the n dimension by NC.
    bli_cntl_initialize_node
    (
      &cntl->loop5,
	  family,
      BLIS_NC,
	  bli_gemm_blk_var2,
      &cntl->nc,
      NULL,
	  &cntl->loop4
	);

	return &cntl->loop5;
}

// -----------------------------------------------------------------------------

// This control tree creation function is disabled because it is no longer used.
// (It was originally created in the run up to publishing the 1m journal article,
// but was disabled to reduce complexity.)
#if 0
cntl_t* bli_gemmpb_cntl_create
     (
       opid_t family
     )
{
	void_fp macro_kernel_p = bli_gemm_ker_var1;

	// Change the macro-kernel if the operation family is gemmt or trmm.
	//if      ( family == BLIS_GEMMT ) macro_kernel_p = bli_gemmt_x_ker_var2;
	//else if ( family == BLIS_TRMM ) macro_kernel_p = bli_trmm_xx_ker_var2;

	// Create two nodes for the macro-kernel.
	cntl_t* gemm_cntl_ub_ke = bli_gemm_cntl_create_node
	(
	  family,  // the operation family
	  BLIS_MR, // needed for bli_thrinfo_rgrow()
	  NULL,    // variant function pointer not used
	  NULL     // no sub-node; this is the leaf of the tree.
	);

	cntl_t* gemm_cntl_pb_ub = bli_gemm_cntl_create_node
	(
	  family,
	  BLIS_NR, // not used by macro-kernel, but needed for bli_thrinfo_rgrow()
	  macro_kernel_p,
	  gemm_cntl_ub_ke
	);

	// Create a node for packing matrix A (which is really the right-hand
	// operand "B").
	cntl_t* gemm_cntl_packb = bli_packm_cntl_create_node
	(
	  bli_gemm_packb,  // pack the right-hand operand
	  bli_packm_blk_var1,
	  BLIS_MR,
	  BLIS_KR,
	  FALSE,   // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  BLIS_PACKED_COL_PANELS,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  gemm_cntl_pb_ub
	);

	// Create a node for partitioning the n dimension by MC.
	cntl_t* gemm_cntl_op_pb = bli_gemm_cntl_create_node
	(
	  family,
	  BLIS_MC,
	  bli_gemm_blk_var2,
	  gemm_cntl_packb
	);

	// Create a node for packing matrix B (which is really the left-hand
	// operand "A").
	cntl_t* gemm_cntl_packa = bli_packm_cntl_create_node
	(
	  bli_gemm_packa,  // pack the left-hand operand
	  bli_packm_blk_var1,
	  BLIS_NR,
	  BLIS_KR,
	  FALSE,   // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  BLIS_PACKED_ROW_PANELS,
	  BLIS_BUFFER_FOR_B_PANEL,
	  gemm_cntl_op_pb
	);

	// Create a node for partitioning the k dimension by KC.
	cntl_t* gemm_cntl_mm_op = bli_gemm_cntl_create_node
	(
	  family,
	  BLIS_KC,
	  bli_gemm_blk_var3,
	  gemm_cntl_packa
	);

	// Create a node for partitioning the m dimension by NC.
	cntl_t* gemm_cntl_vl_mm = bli_gemm_cntl_create_node
	(
	  family,
	  BLIS_NC,
	  bli_gemm_blk_var1,
	  gemm_cntl_mm_op
	);

	return gemm_cntl_vl_mm;
}
#endif
