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


void bli_gemm_var_cntl_init_node
     (
       void_fp          var_func,
       gemm_ukr_vft     ukr,
       bool             row_pref,
       gemm_var_cntl_t* cntl
     )
{
	// Initialize the gemm_var_cntl_t struct.
	cntl->ukr      = ukr;
    cntl->row_pref = row_pref;

	bli_cntl_init_node
	(
	  var_func,
      &cntl->cntl
	);
}

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
	void_fp macro_kernel_fp = NULL;

#ifdef BLIS_ENABLE_JRIR_TLB

	if ( family == BLIS_GEMMT )
    {
        macro_kernel_fp = bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2b : bli_gemmt_u_ker_var2b;
    }
	else if ( family == BLIS_TRMM ||
	          family == BLIS_TRMM3 )
    {
        macro_kernel_fp = bli_obj_is_triangular( a )
            ? bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2b : bli_trmm_lu_ker_var2b
            : bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2b : bli_trmm_ru_ker_var2b;
    }
	else
    {
        macro_kernel_fp = bli_gemm_ker_var2b;
    }

#else

	if ( family == BLIS_GEMMT )
    {
        macro_kernel_fp = bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2 : bli_gemmt_u_ker_var2;
    }
	else if ( family == BLIS_TRMM ||
	          family == BLIS_TRMM3 )
    {
        macro_kernel_fp = bli_obj_is_triangular( a )
            ? bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2 : bli_trmm_lu_ker_var2
            : bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2 : bli_trmm_ru_ker_var2;
    }
	else
    {
        macro_kernel_fp = bli_gemm_ker_var2;
    }

#endif

    const num_t        dt_a         = bli_obj_dt( a );
    const num_t        dt_b         = bli_obj_dt( b );
    const num_t        dt_ap        = bli_obj_target_dt( a );
    const num_t        dt_bp        = bli_obj_target_dt( b );
    const num_t        dt_exec      = bli_obj_exec_dt( c );

	const gemm_ukr_vft gemm_ukr     = bli_cntx_get_ukr_dt( dt_exec, BLIS_GEMM_VIR_UKR, cntx );
    const bool         row_pref     = bli_cntx_get_ukr_prefs_dt( dt_exec, BLIS_GEMM_UKR_ROW_PREF, cntx );

    const bool         a_lo_tri     = bli_obj_is_triangular( a ) && bli_obj_is_lower( a );
    const bool         b_up_tri     = bli_obj_is_triangular( b ) && bli_obj_is_upper( b );
    const bool         trmm_r       = family == BLIS_TRMM && bli_obj_is_triangular( b );

    const dim_t        ic_alg       = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_MC, cntx );
    const dim_t        ic_max       = bli_cntx_get_blksz_max_dt( dt_exec, BLIS_MC, cntx );
    const dim_t        ic_mult      = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_MR, cntx );
    const dir_t        ic_dir       = a_lo_tri ? BLIS_BWD : BLIS_FWD;
          dim_t        pc_alg       = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_KC, cntx );
          dim_t        pc_max       = bli_cntx_get_blksz_max_dt( dt_exec, BLIS_KC, cntx );
    const dim_t        pc_mult      = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_KR, cntx );
    const dir_t        pc_dir       = a_lo_tri || b_up_tri ? BLIS_BWD : BLIS_FWD;
    const dim_t        jc_alg       = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_NC, cntx );
    const dim_t        jc_max       = bli_cntx_get_blksz_max_dt( dt_exec, BLIS_NC, cntx );
    const dim_t        jc_mult      = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_NR, cntx );
    const dir_t        jc_dir       = b_up_tri ? BLIS_BWD : BLIS_FWD;

    const dim_t        bmult_m_def  = bli_cntx_get_blksz_def_dt(   dt_ap, BLIS_MR, cntx );
    const dim_t        bmult_m_pack = bli_cntx_get_blksz_max_dt(   dt_ap, BLIS_MR, cntx );
    const dim_t        bmult_n_def  = bli_cntx_get_blksz_def_dt(   dt_bp, BLIS_NR, cntx );
    const dim_t        bmult_n_pack = bli_cntx_get_blksz_max_dt(   dt_bp, BLIS_NR, cntx );
    const dim_t        bmult_k_def  = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_KR, cntx );

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
	bli_cntl_init_node
	(
	  NULL,         // variant function pointer not used
      &cntl->ir_loop
	);

	bli_gemm_var_cntl_init_node
	(
	  macro_kernel_fp,
      gemm_ukr,
      row_pref,
      &cntl->ker
	);
    bli_cntl_attach_sub_node
    (
      BLIS_THREAD_NR,
      ( cntl_t* )&cntl->ir_loop,
      ( cntl_t* )&cntl->ker
    );

	// Create a node for packing matrix A.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packa, // pack the left-hand operand
      dt_a,
      dt_ap,
	  bmult_m_def,
	  bmult_m_pack,
	  bmult_k_def,
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
      ( cntl_t* )&cntl->ker,
      ( cntl_t* )&cntl->pack_a
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
      ( cntl_t* )&cntl->pack_a,
      ( cntl_t* )&cntl->part_ic
    );

	// Create a node for packing matrix B.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packb, // pack the right-hand operand
      dt_b,
      dt_bp,
	  bmult_n_def,
	  bmult_n_pack,
	  bmult_k_def,
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
      ( cntl_t* )&cntl->part_ic,
      ( cntl_t* )&cntl->pack_b
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
      ( cntl_t* )&cntl->pack_b,
      ( cntl_t* )&cntl->part_pc
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
      ( cntl_t* )&cntl->part_pc,
      ( cntl_t* )&cntl->part_jc
    );
}

