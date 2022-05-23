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

#include "bli_type_defs.h"
#include "blis.h"

cntl_t* bli_trsm_cntl_create
     (
             trsm_cntl_t* cntl,
             side_t       side,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
       const cntx_t*      cntx
     )
{
	if ( bli_is_left( side ) )
		return bli_trsm_l_cntl_create( cntl, a, b, c, cntx );
	else
		return bli_trsm_r_cntl_create( cntl, a, b, c, cntx );
}

cntl_t* bli_trsm_l_cntl_create
     (
             trsm_cntl_t* cntl,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
       const cntx_t*      cntx
     )
{
          num_t           dt_comp      = bli_obj_comp_dt( c );
	      void_fp         ker_fp       = bli_obj_ker_fn( c );
	      void_fp         packa_fp     = bli_obj_ker_fn( a );
	      void_fp         packb_fp     = bli_obj_ker_fn( b );
    const gemm_params_t*  ker_params   = bli_obj_ker_params( c );
    const packm_params_t* packa_params = bli_obj_ker_params( a );
    const packm_params_t* packb_params = bli_obj_ker_params( b );

	// TODO: enable customization
    if ( ker_fp != NULL ||
         packa_fp != NULL ||
         packb_fp != NULL ||
         ker_params != NULL ||
         packa_params != NULL ||
         packb_params != NULL )
    {
        bli_abort();
    }

    ker_fp = bli_trsm_xx_ker_var2;
	packa_fp = bli_l3_packa;
	packb_fp = bli_l3_packb;

    cntl->ker_params.mr = *bli_cntx_get_blksz( BLIS_MR, cntx );
    cntl->ker_params.nr = *bli_cntx_get_blksz( BLIS_NR, cntx );
    cntl->ker_params.ukr = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM_UKR, cntx );
    ker_params = &cntl->ker_params;

    cntl->gemm_packa_params.mr = ker_params->mr;
    cntl->gemm_packa_params.kr = ker_params->mr;
    cntl->gemm_packa_params.does_invert_diag = FALSE;
    cntl->gemm_packa_params.rev_iter_if_upper = TRUE;
    cntl->gemm_packa_params.rev_iter_if_lower = FALSE;
    cntl->gemm_packa_params.pack_schema = BLIS_PACKED_ROW_PANELS;
    cntl->gemm_packa_params.pack_buf_type = BLIS_BUFFER_FOR_A_BLOCK;
    void* gemm_packa_params = &cntl->gemm_packa_params;

    cntl->trsm_packa_params.mr = ker_params->mr;
    cntl->trsm_packa_params.kr = ker_params->mr;
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
    cntl->trsm_packa_params.does_invert_diag = TRUE;
#else
    cntl->trsm_packa_params.does_invert_diag = FALSE;
#endif
    cntl->trsm_packa_params.rev_iter_if_upper = TRUE;
    cntl->trsm_packa_params.rev_iter_if_lower = FALSE;
    cntl->trsm_packa_params.pack_schema = BLIS_PACKED_ROW_PANELS;
    cntl->trsm_packa_params.pack_buf_type = BLIS_BUFFER_FOR_A_BLOCK;
    void* trsm_packa_params = &cntl->gemm_packa_params;

    cntl->packb_params.mr = ker_params->nr;
    cntl->packb_params.kr = ker_params->mr;
    cntl->packb_params.does_invert_diag = FALSE;
    cntl->packb_params.rev_iter_if_upper = FALSE;
    cntl->packb_params.rev_iter_if_lower = FALSE;
    cntl->packb_params.pack_schema = BLIS_PACKED_COL_PANELS;
    cntl->packb_params.pack_buf_type = BLIS_BUFFER_FOR_B_PANEL;
    packb_params = &cntl->packb_params;

    cntl->mc.blksz = *bli_cntx_get_blksz( BLIS_MC, cntx );
    cntl->nc.blksz = *bli_cntx_get_blksz( BLIS_NC, cntx );
    cntl->kc.blksz = *bli_cntx_get_blksz( BLIS_KC, cntx );
    cntl->mc.bmult = ker_params->mr;
    cntl->nc.bmult = ker_params->nr;
    cntl->kc.bmult = packa_params->kr;

    bli_align_blksz_to_mult( &cntl->mc.blksz, &cntl->mc.bmult );
    bli_align_blksz_to_mult( &cntl->nc.blksz, &cntl->nc.bmult );
    bli_l3_adjust_kc( a, b, &ker_params->mr, &ker_params->nr, &cntl->kc.blksz, BLIS_TRSM );

	// Create two nodes for the macro-kernel.
    bli_cntl_initialize_node
    (
      &cntl->gemm_loop1,
      BLIS_TRSM, // the operation family
      BLIS_MR,   // used for thread partitioning
      NULL,      // variant function pointer not used
      NULL,      // not used
      NULL,      // no sub-prenode; this is the leaf of the tree.
      NULL       // no sub-node; this is the leaf of the tree.
    );

    bli_cntl_initialize_node
    (
      &cntl->gemm_loop2,
	  BLIS_TRSM,
      BLIS_NR,
	  ker_fp,
	  ker_params,
      NULL,
      &cntl->gemm_loop1
	);

	// Create a node for packing matrix A.
    bli_cntl_initialize_node
    (
      &cntl->gemm_packa,
	  BLIS_TRSM,
      BLIS_NO_PART,
	  packa_fp,
	  gemm_packa_params,
      NULL,
      &cntl->gemm_loop2
	);

	// Create two nodes for the macro-kernel.
    bli_cntl_initialize_node
    (
      &cntl->trsm_loop1,
      BLIS_TRSM, // the operation family
      BLIS_MR,   // used for thread partitioning
      NULL,      // variant function pointer not used
      NULL,      // not used
      NULL,      // no sub-prenode; this is the leaf of the tree.
      NULL       // no sub-node; this is the leaf of the tree.
    );

    bli_cntl_initialize_node
    (
      &cntl->trsm_loop2,
	  BLIS_TRSM,
      BLIS_NR,
	  ker_fp,
	  ker_params,
      NULL,
      &cntl->trsm_loop1
	);

	// Create a node for packing matrix A.
    bli_cntl_initialize_node
    (
      &cntl->trsm_packa,
	  BLIS_TRSM,
      BLIS_NO_PART,
	  packa_fp,
	  trsm_packa_params,
      NULL,
      &cntl->trsm_loop2
	);

	// Create a node for partitioning the m dimension by MC.
    bli_cntl_initialize_node
    (
      &cntl->loop3,
	  BLIS_TRSM,
      BLIS_MC,
	  bli_gemm_blk_var1,
      &cntl->mc,
	  &cntl->trsm_packa,
	  &cntl->gemm_packa
	);

	// Create a node for packing matrix B.
    bli_cntl_initialize_node
    (
      &cntl->packb,
	  BLIS_TRSM,
      BLIS_NO_PART,
	  packb_fp,
	  packb_params,
      NULL,
      &cntl->loop3
	);

	// Create a node for partitioning the k dimension by KC.
    bli_cntl_initialize_node
    (
      &cntl->loop4,
	  BLIS_TRSM,
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
	  BLIS_TRSM,
      BLIS_NC,
	  bli_gemm_blk_var2,
      &cntl->nc,
      NULL,
	  &cntl->loop4
	);

	return &cntl->loop5;
}

cntl_t* bli_trsm_r_cntl_create
     (
             trsm_cntl_t* cntl,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
       const cntx_t*      cntx
     )
{
          num_t           dt_comp      = bli_obj_comp_dt( c );
	      void_fp         ker_fp       = bli_obj_ker_fn( c );
	      void_fp         packa_fp     = bli_obj_ker_fn( a );
	      void_fp         packb_fp     = bli_obj_ker_fn( b );
    const gemm_params_t*  ker_params   = bli_obj_ker_params( c );
    const packm_params_t* packa_params = bli_obj_ker_params( a );
    const packm_params_t* packb_params = bli_obj_ker_params( b );

	// TODO: enable customization
    if ( ker_fp != NULL ||
         packa_fp != NULL ||
         packb_fp != NULL ||
         ker_params != NULL ||
         packa_params != NULL ||
         packb_params != NULL )
    {
        bli_abort();
    }

    ker_fp = bli_trsm_xx_ker_var2;
	packa_fp = bli_l3_packa;
	packb_fp = bli_l3_packb;

    cntl->ker_params.mr = *bli_cntx_get_blksz( BLIS_MR, cntx );
    cntl->ker_params.nr = *bli_cntx_get_blksz( BLIS_NR, cntx );
    cntl->ker_params.ukr = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM_UKR, cntx );
    ker_params = &cntl->ker_params;

    cntl->trsm_packa_params.mr = ker_params->nr;
    cntl->trsm_packa_params.kr = ker_params->mr;
    cntl->trsm_packa_params.does_invert_diag = FALSE;
    cntl->trsm_packa_params.rev_iter_if_upper = FALSE;
    cntl->trsm_packa_params.rev_iter_if_lower = FALSE;
    cntl->trsm_packa_params.pack_schema = BLIS_PACKED_ROW_PANELS;
    cntl->trsm_packa_params.pack_buf_type = BLIS_BUFFER_FOR_A_BLOCK;
    packa_params = &cntl->gemm_packa_params;

    cntl->packb_params.mr = ker_params->mr;
    cntl->packb_params.kr = ker_params->mr;
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
    cntl->packb_params.does_invert_diag = TRUE;
#else
    cntl->trsm_packa_params.does_invert_diag = FALSE;
#endif
    cntl->packb_params.rev_iter_if_upper = TRUE;
    cntl->packb_params.rev_iter_if_lower = FALSE;
    cntl->packb_params.pack_schema = BLIS_PACKED_COL_PANELS;
    cntl->packb_params.pack_buf_type = BLIS_BUFFER_FOR_B_PANEL;
    packb_params = &cntl->packb_params;

    cntl->mc.blksz = *bli_cntx_get_blksz( BLIS_MC, cntx );
    cntl->nc.blksz = *bli_cntx_get_blksz( BLIS_NC, cntx );
    cntl->kc.blksz = *bli_cntx_get_blksz( BLIS_KC, cntx );
    cntl->mc.bmult = ker_params->mr;
    cntl->nc.bmult = ker_params->nr;
    cntl->kc.bmult = packa_params->kr;

    bli_align_blksz_to_mult( &cntl->mc.blksz, &cntl->mc.bmult );
    bli_align_blksz_to_mult( &cntl->nc.blksz, &cntl->nc.bmult );
    bli_l3_adjust_kc( a, b, &ker_params->mr, &ker_params->nr, &cntl->kc.blksz, BLIS_TRSM );

	// Create two nodes for the macro-kernel.
    bli_cntl_initialize_node
    (
      &cntl->trsm_loop1,
      BLIS_TRSM, // the operation family
      BLIS_MR,   // used for thread partitioning
      NULL,      // variant function pointer not used
      NULL,      // not used
      NULL,      // no sub-prenode; this is the leaf of the tree.
      NULL       // no sub-node; this is the leaf of the tree.
    );

    bli_cntl_initialize_node
    (
      &cntl->trsm_loop2,
	  BLIS_TRSM,
      BLIS_NR,
	  ker_fp,
	  ker_params,
      NULL,
      &cntl->trsm_loop1
	);

	// Create a node for packing matrix A.
    bli_cntl_initialize_node
    (
      &cntl->trsm_packa,
	  BLIS_TRSM,
      BLIS_NO_PART,
	  packa_fp,
	  packa_params,
      NULL,
      &cntl->trsm_loop2
	);

	// Create a node for partitioning the m dimension by MC.
    bli_cntl_initialize_node
    (
      &cntl->loop3,
	  BLIS_TRSM,
      BLIS_MC,
	  bli_gemm_blk_var1,
      &cntl->mc,
      NULL,
	  &cntl->trsm_packa
	);

	// Create a node for packing matrix B.
    bli_cntl_initialize_node
    (
      &cntl->packb,
	  BLIS_TRSM,
      BLIS_NO_PART,
	  packb_fp,
	  packb_params,
      NULL,
      &cntl->loop3
	);

	// Create a node for partitioning the k dimension by KC.
    bli_cntl_initialize_node
    (
      &cntl->loop4,
	  BLIS_TRSM,
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
	  BLIS_TRSM,
      BLIS_NC,
	  bli_gemm_blk_var2,
      &cntl->nc,
      NULL,
	  &cntl->loop4
	);

	return &cntl->loop5;
}

