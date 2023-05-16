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


static packm_ker_ft GENARRAY(packm_struc_cxk,packm_struc_cxk);
static packm_ker_ft GENARRAY2_ALL(packm_struc_cxk_md,packm_struc_cxk_md);

void bli_gemm_var_cntl_init_node
     (
       void_fp          var_func,
       num_t            dt_comp,
       num_t            dt_out,
       gemm_ukr_ft      ukr,
       gemm_ukr_ft     real_ukr,
       bool             row_pref,
       dim_t            mr,
       dim_t            nr,
       dim_t            mr_scale,
       dim_t            nr_scale,
       gemm_var_cntl_t* cntl
     )
{
	// Initialize the gemm_var_cntl_t struct.
	cntl->dt_comp  = dt_comp;
	cntl->dt_out   = dt_out;
	cntl->ukr      = ukr;
	cntl->real_ukr = real_ukr;
	cntl->row_pref = row_pref;
	cntl->mr       = mr;
	cntl->nr       = nr;
	cntl->mr_scale = mr_scale;
	cntl->nr_scale = nr_scale;

	bli_cntl_init_node
	(
	  var_func,
	  &cntl->cntl
	);
}

void bli_gemm_cntl_init
     (
             ind_t        im,
             opid_t       family,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl
     )
{
	const prec_t      comp_prec     = bli_obj_comp_prec( c );
	const num_t       dt_c          = bli_obj_dt( c );
	const num_t       dt_comp       = ( im == BLIS_1M ? BLIS_REAL : bli_dt_domain( dt_c ) ) | comp_prec;
	      gemm_ukr_ft gemm_ukr      = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM_UKR, cntx );
	      gemm_ukr_ft real_gemm_ukr = NULL;
	const bool        row_pref      = bli_cntx_get_ukr_prefs_dt( dt_comp, BLIS_GEMM_UKR_ROW_PREF, cntx );

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	bool needs_swap = (   row_pref && bli_obj_is_col_tilted( c ) ) ||
	                  ( ! row_pref && bli_obj_is_row_tilted( c ) );

	// NOTE: This case casts right-side symm/hemm/trmm/trmm3 in terms of left side.
	// This may be necessary when the current subconfiguration uses a gemm microkernel
	// that assumes that the packing kernel will have already duplicated
	// (broadcast) element of B in the packed copy of B. Supporting
	// duplication within the logic that packs micropanels from symmetric
	// matrices is ugly, but technically supported. This can
	// lead to the microkernel being executed on an output matrix with the
	// microkernel's general stride IO case (unless the microkernel supports
	// both both row and column IO cases as well). As a
	// consequence, those subconfigurations need a way to force the symmetric
	// matrix to be on the left (and thus the general matrix to the on the
	// right). So our solution is that in those cases, the subconfigurations
	// simply #define BLIS_DISABLE_{SYMM,HEMM,TRMM,TRMM3}_RIGHT.

	// If A is being multiplied from the right, transpose all operands
	// so that we can perform the computation as if A were being multiplied
	// from the left.
#ifdef BLIS_DISABLE_SYMM_RIGHT
	if ( family == BLIS_SYMM )
		needs_swap = bli_obj_is_symmetric( b );
#endif
#ifdef BLIS_DISABLE_HEMM_RIGHT
	if ( family == BLIS_HEMM )
		needs_swap = bli_obj_is_hermitian( b );
#endif
#ifdef BLIS_DISABLE_TRMM_RIGHT
	if ( family == BLIS_TRMM )
		needs_swap = bli_obj_is_triangular( b );
#endif
#ifdef BLIS_DISABLE_TRMM3_RIGHT
	if ( family == BLIS_TRMM3 )
		needs_swap = bli_obj_is_triangular( b );
#endif

	// Swap the A and B operands if required. This transforms the operation
	// C = alpha A B + beta C into C^T = alpha B^T A^T + beta C^T.
	if ( needs_swap )
	{
		bli_obj_swap( a, b );

		bli_obj_induce_trans( a );
		bli_obj_induce_trans( b );
		bli_obj_induce_trans( c );
	}

	// If alpha is non-unit, typecast and apply it to the scalar attached
	// to B, unless it happens to be triangular.
	if ( bli_obj_root_is_triangular( b ) )
	{
		if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( alpha, a );
	}
	else // if ( bli_obj_root_is_triangular( b ) )
	{
		if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( alpha, b );
	}

	// If beta is non-unit, typecast and apply it to the scalar attached
	// to C.
	if ( !bli_obj_equals( beta, &BLIS_ONE ) )
		bli_obj_scalar_apply_scalar( beta, c );

	void_fp macro_kernel_fp = family == BLIS_GEMM ||
	                          family == BLIS_HEMM ||
							  family == BLIS_SYMM ? bli_gemm_ker_var2 :
#ifdef BLIS_ENABLE_JRIR_TLB
	                          family == BLIS_GEMMT ?
	                             bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2b : bli_gemmt_u_ker_var2b :
	                          family == BLIS_TRMM ||
	                          family == BLIS_TRMM3 ?
	                              bli_obj_is_triangular( a ) ?
	                                 bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2b : bli_trmm_lu_ker_var2b :
	                                 bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2b : bli_trmm_ru_ker_var2b :
	                          NULL; // Should never happen
#else
	                          family == BLIS_GEMMT ?
	                             bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2 : bli_gemmt_u_ker_var2 :
	                          family == BLIS_TRMM ||
	                          family == BLIS_TRMM3 ?
	                              bli_obj_is_triangular( a ) ?
	                                 bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2 : bli_trmm_lu_ker_var2 :
	                                 bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2 : bli_trmm_ru_ker_var2 :
	                          NULL; // Should never happen
#endif

	const num_t         dt_a          = bli_obj_dt( a );
	const num_t         dt_b          = bli_obj_dt( b );
	const num_t         dt_ap         = bli_dt_domain( dt_a ) | comp_prec;
	const num_t         dt_bp         = bli_dt_domain( dt_b ) | comp_prec;
	const bool          trmm_r        = family == BLIS_TRMM && bli_obj_is_triangular( b );
	const bool          a_lo_tri      = bli_obj_is_triangular( a ) && bli_obj_is_lower( a );
	const bool          b_up_tri      = bli_obj_is_triangular( b ) && bli_obj_is_upper( b );
	      pack_t        schema_a      = BLIS_PACKED_ROW_PANELS;
	      pack_t        schema_b      = BLIS_PACKED_COL_PANELS;
	const packm_ker_ft  packm_a_ukr   = dt_a == dt_ap ? packm_struc_cxk[ dt_a ]
	                                                  : packm_struc_cxk_md[ dt_a ][ dt_ap ];
	const packm_ker_ft  packm_b_ukr   = dt_b == dt_bp ? packm_struc_cxk[ dt_b ]
	                                                  : packm_struc_cxk_md[ dt_b ][ dt_bp ];
	const dim_t         mr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MR, cntx );
	const dim_t         mr_pack       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MR, cntx );
	const dim_t         mr_bcast      = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBM, cntx );
	      dim_t         mr_scale      = 1;
	const dim_t         nr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NR, cntx );
	const dim_t         nr_pack       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NR, cntx );
	const dim_t         nr_bcast      = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBN, cntx );
	      dim_t         nr_scale      = 1;
	const dim_t         kr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KR, cntx );
	const dim_t         mc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MC, cntx );
	const dim_t         mc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MC, cntx );
	      dim_t         mc_scale      = 1;
	const dim_t         nc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NC, cntx );
	const dim_t         nc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NC, cntx );
	      dim_t         nc_scale      = 1;
	const dim_t         kc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KC, cntx );
	const dim_t         kc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_KC, cntx );
	      dim_t         kc_scale      = 1;

	if ( im == BLIS_1M )
	{
		if ( ! row_pref )
		{
			schema_a = BLIS_PACKED_ROW_PANELS_1E;
			schema_b = BLIS_PACKED_COL_PANELS_1R;
			mr_scale = 2;
			mc_scale = 2;
		}
		else
		{
			schema_a = BLIS_PACKED_ROW_PANELS_1R;
			schema_b = BLIS_PACKED_COL_PANELS_1E;
			nr_scale = 2;
			nc_scale = 2;
		}

		kc_scale = 2;
		real_gemm_ukr = gemm_ukr;
		gemm_ukr = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM1M_UKR, cntx );
	}

#if 0
#ifdef BLIS_ENABLE_GEMM_MD
	cntx_t cntx_local;

	// If any of the storage datatypes differ, or if the computation precision
	// differs from the storage precision of C, utilize the mixed datatype
	// code path.
	// NOTE: If we ever want to support the caller setting the computation
	// domain explicitly, we will need to check the computation dt against the
	// storage dt of C (instead of the computation precision against the
	// storage precision of C).
	if ( bli_obj_dt( &c_local ) != bli_obj_dt( &a_local ) ||
	     bli_obj_dt( &c_local ) != bli_obj_dt( &b_local ) ||
	     bli_obj_comp_prec( &c_local ) != bli_obj_prec( &c_local ) )
	{
		// Handle mixed datatype cases in bli_gemm_md(), which may modify
		// the objects or the context. (If the context is modified, cntx
		// is adjusted to point to cntx_local.)
		bli_gemm_md( &a_local, &b_local, beta, &c_local, &schema_a, &schema_b, &cntx_local, &cntx );
	}
#endif
#endif

	// Create two nodes for the macro-kernel.
	bli_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  &cntl->ir_loop
	);

	bli_gemm_var_cntl_init_node
	(
	  macro_kernel_fp,
	  dt_comp,
	  dt_c,
	  gemm_ukr,
	  real_gemm_ukr,
	  row_pref,
	  mr_def / mr_scale,
	  nr_def / nr_scale,
	  mr_scale,
	  nr_scale,
	  &cntl->ker
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NR,
	  ( cntl_t* )&cntl->ir_loop,
	  ( cntl_t* )&cntl->ker
	);

	// Give the gemm kernel control tree node to the
	// virtual microkernel as the parameters, so that e.g.
	// the 1m virtual microkernel can look up the real-domain
	// micro-kernel and its parameters.
	bli_gemm_var_cntl_set_params( &cntl->ker, ( cntl_t* )&cntl->ker );

	// Create a node for packing matrix A.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packa, // pack the left-hand operand
	  dt_a,
	  dt_ap,
	  dt_comp,
	  packm_a_ukr,
	  mr_def / mr_scale,
	  mr_pack,
	  mr_bcast,
	  mr_scale,
	  kr_def,
	  FALSE,
	  FALSE,
	  FALSE,
	  schema_a,
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
	  dt_comp,
	  mc_def / mc_scale,
	  mc_max / mc_scale,
	  mc_scale,
	  mr_def / mr_scale,
	  mr_scale,
	  a_lo_tri ? BLIS_BWD : BLIS_FWD,
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
	  dt_comp,
	  packm_b_ukr,
	  nr_def / nr_scale,
	  nr_pack,
	  nr_bcast,
	  nr_scale,
	  kr_def,
	  FALSE,
	  FALSE,
	  FALSE,
	  schema_b,
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
	  dt_comp,
	  kc_def / kc_scale,
	  kc_max / kc_scale,
	  kc_scale,
	  kr_def,
	  1,
	  a_lo_tri || b_up_tri ? BLIS_BWD : BLIS_FWD,
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
	  dt_comp,
	  nc_def / nc_scale,
	  nc_max / nc_scale,
	  nc_scale,
	  nr_def / nr_scale,
	  nr_scale,
	  b_up_tri ? BLIS_BWD : BLIS_FWD,
	  bli_obj_is_triangular( b ) || bli_obj_is_upper_or_lower( c ),
	  &cntl->part_jc
	);
	bli_cntl_attach_sub_node
	(
	  trmm_r ? BLIS_THREAD_NONE : BLIS_THREAD_NC,
	  ( cntl_t* )&cntl->part_pc,
	  ( cntl_t* )&cntl->part_jc
	);

	bli_gemm_cntl_finalize
	(
	  family,
	  a,
	  b,
	  c,
	  cntl
	);
}

void bli_gemm_cntl_finalize
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             gemm_cntl_t* cntl
     )
{
	( void )c;

	const dim_t ic_mult = bli_part_cntl_blksz_mult( ( cntl_t* )&cntl->part_ic );
	const dim_t jc_mult = bli_part_cntl_blksz_mult( ( cntl_t* )&cntl->part_jc );

	//
	// Ensure that:
	//
	// 1. KC is a multiple of MR (NR) if A (B) is triangular, hermitian, or symmetric.
	//    KC is always rounded up.
	//
	// 2. MC and NR are multiples of MR and NR, respectively. MC and NC are always
	//    rounded down.
	//

	// Nudge the default and maximum kc blocksizes up to the nearest
	// multiple of MR if A is Hermitian, symmetric, or triangular or
	// NR if B is Hermitian, symmetric, or triangular. If neither case
	// applies, then we leave the blocksizes unchanged. For trsm we
	// always use MR (rather than sometimes using NR) because even
	// when the triangle is on the right, packing of that matrix uses
	// MR, since only left-side trsm micro-kernels are supported.
	if ( !bli_obj_root_is_general( a ) || family == BLIS_TRSM )
	{
		bli_part_cntl_align_blksz_to_mult( ic_mult, true, ( cntl_t* )&cntl->part_pc );
	}
	else if ( !bli_obj_root_is_general( b ) )
	{
		bli_part_cntl_align_blksz_to_mult( jc_mult, true, ( cntl_t* )&cntl->part_pc );
	}

	bli_part_cntl_align_blksz( false, ( cntl_t* )&cntl->part_ic );
	bli_part_cntl_align_blksz( false, ( cntl_t* )&cntl->part_jc );
}

