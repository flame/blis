/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

/*
 * List of default block sizes for zen4.
 * Converted it to macro as this list is used at multiple places in this file.
 */

// blcksizes for double datatype are dynamic and will be modified at runtime by
// function bli_dynamic_blkszs_zen4
#define BLI_CNTX_DEFAULT_BLKSZ_LIST_GENOA(blkszs) \
	/*                                           s      d      c      z */  \
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,     8,    24,    12 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    12,    24,     4,     4 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   512,   120,    96,    60 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   480,   512,   640,   512 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  6144,  4032,  4080,  2004 );  \
	                                                                        \
	bli_blksz_init_easy( &blkszs[ BLIS_AF ],     5,     5,    -1,    -1 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_DF ],     8,     8,    -1,    -1 );

#define BLI_CNTX_DEFAULT_BLKSZ_LIST_BERGAMO(blkszs) \
	/*                                           s      d      c      z */  \
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,     8,    24,    12 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    12,    24,     4,     4 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   512,   120,    96,    60 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   480,   512,   640,   512 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  6144,  4032,  4080,  2004 );  \
	                                                                        \
	bli_blksz_init_easy( &blkszs[ BLIS_AF ],     5,     5,    -1,    -1 );  \
	bli_blksz_init_easy( &blkszs[ BLIS_DF ],     8,     8,    -1,    -1 );

void bli_cntx_init_zen4( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	blksz_t thresh[ BLIS_NUM_THRESH ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_zen4_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  13,
	  // gemm
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_skx_asm_32x12_l2,   FALSE,
	  BLIS_GEMM_UKR,       BLIS_DOUBLE,   bli_dgemm_zen4_asm_8x24,    TRUE,
	  BLIS_GEMM_UKR,       BLIS_SCOMPLEX, bli_cgemm_zen4_asm_24x4,      FALSE,
	  /*bli_zgemm_zen4_asm_12x4 is a column preferred kernel*/
	  BLIS_GEMM_UKR,       BLIS_DCOMPLEX, bli_zgemm_zen4_asm_12x4,      FALSE,

	  // Different  GEMM kernels are used for TRSM for zen4 architecture
	  BLIS_GEMM_FOR_TRSM_UKR,       BLIS_FLOAT,    bli_sgemm_haswell_asm_6x16,  TRUE,
	  BLIS_GEMM_FOR_TRSM_UKR,       BLIS_DOUBLE,   bli_dgemm_zen4_asm_8x24,   TRUE,
	  BLIS_GEMM_FOR_TRSM_UKR,       BLIS_DCOMPLEX, bli_zgemm_zen4_asm_4x12,     TRUE,

	  // gemmtrsm_l
	  BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_haswell_asm_6x16, TRUE,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_l_zen4_asm_8x24,    TRUE,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DCOMPLEX, bli_zgemmtrsm_l_zen4_asm_4x12,    TRUE,
	  // gemmtrsm_u
	  BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_haswell_asm_6x16, TRUE,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_u_zen4_asm_8x24,    TRUE,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DCOMPLEX, bli_zgemmtrsm_u_zen4_asm_4x12,    TRUE,
	  cntx
	);

	// Update the context with architecture specific threshold functions
	bli_cntx_set_l3_thresh_funcs
	(
	  3,
	  // GEMM
	  BLIS_GEMM,  bli_cntx_gemmsup_thresh_is_met_zen4,
	  // GEMMT
	  BLIS_GEMMT, bli_cntx_gemmtsup_thresh_is_met_zen,
	  // SYRK
	  BLIS_SYRK,  bli_cntx_syrksup_thresh_is_met_zen,
	  cntx
	);

	// Update the context with optimized packm kernels.
	bli_cntx_set_packm_kers
	(
	  11,
	  BLIS_PACKM_6XK_KER,  BLIS_FLOAT,    bli_spackm_haswell_asm_6xk,
	  BLIS_PACKM_16XK_KER, BLIS_FLOAT,    bli_spackm_haswell_asm_16xk,
	  BLIS_PACKM_6XK_KER,  BLIS_DOUBLE,   bli_dpackm_haswell_asm_6xk,
	  BLIS_PACKM_8XK_KER,  BLIS_DOUBLE,   bli_dpackm_zen4_asm_8xk,
	  BLIS_PACKM_24XK_KER, BLIS_DOUBLE,   bli_dpackm_zen4_asm_24xk,
	  BLIS_PACKM_32XK_KER, BLIS_DOUBLE,   bli_dpackm_zen4_asm_32xk,
	  BLIS_PACKM_4XK_KER,  BLIS_SCOMPLEX, bli_cpackm_zen4_asm_4xk,
	  BLIS_PACKM_24XK_KER, BLIS_SCOMPLEX, bli_cpackm_zen4_asm_24xk,
	  BLIS_PACKM_3XK_KER,  BLIS_DCOMPLEX, bli_zpackm_haswell_asm_3xk,
	  BLIS_PACKM_12XK_KER, BLIS_DCOMPLEX, bli_zpackm_zen4_asm_12xk,
	  BLIS_PACKM_4XK_KER,  BLIS_DCOMPLEX, bli_zpackm_zen4_asm_4xk,
	  cntx
	);

	// Update the context with optimized level-1f kernels.
	bli_cntx_set_l1f_kers
	(
	  9,
	  // axpyf
	  BLIS_AXPYF_KER,     BLIS_FLOAT,    bli_saxpyf_zen_int_5,
	  BLIS_AXPYF_KER,     BLIS_DOUBLE,   bli_daxpyf_zen_int_5,
	  BLIS_AXPYF_KER,     BLIS_SCOMPLEX, bli_caxpyf_zen_int_5,
	  BLIS_AXPYF_KER,     BLIS_DCOMPLEX, bli_zaxpyf_zen_int_5,
	  // dotxf
	  BLIS_DOTXF_KER,     BLIS_FLOAT,    bli_sdotxf_zen_int_8,
	  BLIS_DOTXF_KER,     BLIS_DOUBLE,   bli_ddotxf_zen_int_8,
	  BLIS_DOTXF_KER,     BLIS_DCOMPLEX, bli_zdotxf_zen_int_6,
	  BLIS_DOTXF_KER,     BLIS_SCOMPLEX, bli_cdotxf_zen_int_6,
	  // axpy2v
	  BLIS_AXPY2V_KER,    BLIS_DOUBLE,   bli_daxpy2v_zen_int,
	  cntx
	);

	// Update the context with optimized level-1v kernels.
	bli_cntx_set_l1v_kers
	(
	  32,
	  // addv
	  BLIS_ADDV_KER,  BLIS_DOUBLE,     bli_daddv_zen4_int,

	  // amaxv
	  BLIS_AMAXV_KER,  BLIS_FLOAT,    bli_samaxv_zen4_int,
	  BLIS_AMAXV_KER,  BLIS_DOUBLE,   bli_damaxv_zen4_int,

	  // axpbyv
	  BLIS_AXPBYV_KER, BLIS_FLOAT,    bli_saxpbyv_zen_int_10,
	  BLIS_AXPBYV_KER, BLIS_DOUBLE,   bli_daxpbyv_zen4_int,
	  BLIS_AXPBYV_KER, BLIS_SCOMPLEX, bli_caxpbyv_zen_int,
	  BLIS_AXPBYV_KER, BLIS_DCOMPLEX, bli_zaxpbyv_zen_int,

	  // axpyv
	  BLIS_AXPYV_KER,  BLIS_FLOAT,    bli_saxpyv_zen4_int,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE,   bli_daxpyv_zen4_int,
	  BLIS_AXPYV_KER,  BLIS_SCOMPLEX, bli_caxpyv_zen_int_5,
	  BLIS_AXPYV_KER,  BLIS_DCOMPLEX, bli_zaxpyv_zen4_int,

	  // dotv
	  BLIS_DOTV_KER,   BLIS_FLOAT,    bli_sdotv_zen4_int,
	  BLIS_DOTV_KER,   BLIS_DOUBLE,   bli_ddotv_zen4_int,
	  BLIS_DOTV_KER,   BLIS_SCOMPLEX, bli_cdotv_zen_int_5,
	  BLIS_DOTV_KER,   BLIS_DCOMPLEX, bli_zdotv_zen4_int,

	  // dotxv
	  BLIS_DOTXV_KER,  BLIS_FLOAT,    bli_sdotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DOUBLE,   bli_ddotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DCOMPLEX, bli_zdotxv_zen_int,

	  // scalv
	  BLIS_SCALV_KER,  BLIS_FLOAT,    bli_sscalv_zen4_int,
	  BLIS_SCALV_KER,  BLIS_DOUBLE,   bli_dscalv_zen4_int,
	  BLIS_SCALV_KER,  BLIS_SCOMPLEX, bli_cscalv_zen4_int,
	  BLIS_SCALV_KER,  BLIS_DCOMPLEX, bli_zscalv_zen4_int,

	  // swapv
	  BLIS_SWAPV_KER,  BLIS_FLOAT,    bli_sswapv_zen_int_8,
	  BLIS_SWAPV_KER,  BLIS_DOUBLE,   bli_dswapv_zen_int_8,

	  // copyv
	  BLIS_COPYV_KER,  BLIS_FLOAT,    bli_scopyv_zen4_asm,
	  BLIS_COPYV_KER,  BLIS_DOUBLE,   bli_dcopyv_zen4_asm,
	  BLIS_COPYV_KER,  BLIS_DCOMPLEX, bli_zcopyv_zen4_asm,

	  // setv
	  BLIS_SETV_KER,   BLIS_FLOAT,    bli_ssetv_zen4_int,
	  BLIS_SETV_KER,   BLIS_DOUBLE,   bli_dsetv_zen4_int,
	  BLIS_SETV_KER,   BLIS_DCOMPLEX, bli_zsetv_zen4_int,

	  // scal2v
	  BLIS_SCAL2V_KER, BLIS_DOUBLE,   bli_dscal2v_zen4_int,
	  BLIS_SCAL2V_KER, BLIS_DCOMPLEX, bli_zscal2v_zen_int,
	  cntx
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//
	// These are reference block sizes and may be overridden based on
	// number of threads used at runtime.

	if ( bli_init_model_query_id() == BLIS_MODEL_BERGAMO )
	{
	    BLI_CNTX_DEFAULT_BLKSZ_LIST_BERGAMO(blkszs);
	}
	else // BLIS_MODEL_DEFAULT choice, also currently used for BLIS_MODEL_GENOA and BLIS_MODEL_GENOA_X
	{
	    BLI_CNTX_DEFAULT_BLKSZ_LIST_GENOA(blkszs);
	}

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  BLIS_NAT, 7,
	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
	  // level-1f
	  BLIS_AF, &blkszs[ BLIS_AF ], BLIS_AF,
	  BLIS_DF, &blkszs[ BLIS_DF ], BLIS_DF,
	  cntx
	);

	// -------------------------------------------------------------------------

	// Initialize TRSM blocksize objects with architecture-specific values.
	// Using different cache block sizes for TRSM instead of common level-3 block sizes.
	// Tuning is done for double-precision only.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],     6,     8,    24,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,    24,     4,    12 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,   120,   144,    40 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   512,   256,   512 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4080,  4008,  4080,  2004 );

	// Update the context with the current architecture's register and cache
	// blocksizes for level-3 TRSM problems.
	bli_cntx_set_trsm_blkszs
	(
	  5,
	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ],
	  BLIS_KC, &blkszs[ BLIS_KC ],
	  BLIS_MC, &blkszs[ BLIS_MC ],
	  BLIS_NR, &blkszs[ BLIS_NR ],
	  BLIS_MR, &blkszs[ BLIS_MR ],
	  cntx
	);

	// -------------------------------------------------------------------------

	// Initialize sup thresholds with architecture-appropriate values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &thresh[ BLIS_MT ],   682,  1000,   380,   110 );
	bli_blksz_init_easy( &thresh[ BLIS_NT ],   512,  1000,   256,   128 );
	bli_blksz_init_easy( &thresh[ BLIS_KT ],   240,   220,   220,   110 );

	// Initialize the context with the sup thresholds.
	bli_cntx_set_l3_sup_thresh
	(
	  3,
	  BLIS_MT, &thresh[ BLIS_MT ],
	  BLIS_NT, &thresh[ BLIS_NT ],
	  BLIS_KT, &thresh[ BLIS_KT ],
	  cntx
	);

	// Initialize the context with the sup handlers.
	bli_cntx_set_l3_sup_handlers
	(
	  2,
	  BLIS_GEMM,  bli_gemmsup_ref,
	  BLIS_GEMMT, bli_gemmtsup_ref,
	  cntx
	);

	// Update the context with optimized small/unpacked gemm kernels.
	bli_cntx_set_l3_sup_kers
	(
	  32,
	  BLIS_RRR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_RRC, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_RCR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_RCC, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_CRR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_CRC, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_CCR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,
	  BLIS_CCC, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m_new, FALSE,

	  BLIS_RRR, BLIS_FLOAT, bli_sgemmsup_rv_zen4_asm_6x64m, TRUE,
	  BLIS_RRC, BLIS_FLOAT, bli_sgemmsup_rd_zen4_asm_6x64m, TRUE,
	  BLIS_RCR, BLIS_FLOAT, bli_sgemmsup_rv_zen4_asm_6x64m, TRUE,
	  BLIS_RCC, BLIS_FLOAT, bli_sgemmsup_rv_zen4_asm_6x64n, TRUE,
	  BLIS_CRR, BLIS_FLOAT, bli_sgemmsup_rv_zen4_asm_6x64m, TRUE,
	  BLIS_CRC, BLIS_FLOAT, bli_sgemmsup_rd_zen4_asm_6x64n, TRUE,
	  BLIS_CCR, BLIS_FLOAT, bli_sgemmsup_rv_zen4_asm_6x64n, TRUE,
	  BLIS_CCC, BLIS_FLOAT, bli_sgemmsup_rv_zen4_asm_6x64n, TRUE,

	  BLIS_RRR, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_RRC, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_RCR, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_CRR, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_RCC, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_CRC, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_CCR, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,
	  BLIS_CCC, BLIS_SCOMPLEX, bli_cgemmsup_cv_zen4_asm_24x4m, FALSE,

	  BLIS_RRR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_RRC, BLIS_DCOMPLEX, bli_zgemmsup_cd_zen4_asm_12x4m, FALSE,
	  BLIS_RCR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_RCC, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CRR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CRC, BLIS_DCOMPLEX, bli_zgemmsup_cd_zen4_asm_12x4m, FALSE,
	  BLIS_CCR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CCC, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  cntx
	);

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR ],     6,    24,    24,    12,
	                                             6,     9,    24,    12 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    64,     8,     4,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   192,   144,   120,    48 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,   480,   512,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8064,  4080,  4080,  1020 );

	// Update the context with the current architecture's register and cache
	// blocksizes for small/unpacked level-3 problems.
	bli_cntx_set_l3_sup_blkszs
	(
	  5,
	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ],
	  BLIS_KC, &blkszs[ BLIS_KC ],
	  BLIS_MC, &blkszs[ BLIS_MC ],
	  BLIS_NR, &blkszs[ BLIS_NR ],
	  BLIS_MR, &blkszs[ BLIS_MR ],
	  cntx
	);

	// Initialize level-3 sup blocksize objects for operations dealing with
	// triangular objects with architecture-specific values.
	//
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],     6,    24,     3,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,     8,     8,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,   144,    72,    48 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,   480,   128,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8160,  4080,  2040,  1020 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_l3_sup_tri_blkszs
	(
	  5,
	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ],
	  BLIS_KC, &blkszs[ BLIS_KC ],
	  BLIS_MC, &blkszs[ BLIS_MC ],
	  BLIS_NR, &blkszs[ BLIS_NR ],
	  BLIS_MR, &blkszs[ BLIS_MR ],
	  cntx
	);

	bli_cntx_set_l3_sup_tri_kers
	(
	  30,
	  BLIS_RRR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m, TRUE,
	  BLIS_RRC, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16m, TRUE,
	  BLIS_RCR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m, TRUE,
	  BLIS_RCC, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n, TRUE,
	  BLIS_CRR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m, TRUE,
	  BLIS_CRC, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16n, TRUE,
	  BLIS_CCR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n, TRUE,
	  BLIS_CCC, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n, TRUE,

	  BLIS_RRR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m,   FALSE,
	  BLIS_RRC, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8m, TRUE,
	  BLIS_RCR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m,   FALSE,
	  BLIS_RCC, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m,   FALSE,
	  BLIS_CRR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m,   FALSE,
	  BLIS_CRC, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8n, TRUE,
	  BLIS_CCR, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m,   FALSE,
	  BLIS_CCC, BLIS_DOUBLE, bli_dgemmsup_cv_zen4_asm_24x8m,   FALSE,

	  BLIS_RRR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_RCR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_CRR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_RCC, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,
	  BLIS_CCR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,
	  BLIS_CCC, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,

	  BLIS_RRR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen4_asm_4x4m, TRUE,
	  BLIS_RRC, BLIS_DCOMPLEX, bli_zgemmsup_rd_zen_asm_3x4m, TRUE,
	  BLIS_RCR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen4_asm_4x4m, TRUE,
	  BLIS_RCC, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen4_asm_4x4m, TRUE,
	  BLIS_CRR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen4_asm_4x4m, TRUE,
	  BLIS_CRC, BLIS_DCOMPLEX, bli_zgemmsup_rd_zen_asm_3x4n, TRUE,
	  BLIS_CCR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen4_asm_4x4m, TRUE,
	  BLIS_CCC, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen4_asm_4x4m, TRUE,
	  cntx
	);
}
