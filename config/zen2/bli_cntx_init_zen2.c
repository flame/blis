/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_cntx_init_zen2( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_zen2_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  8,
	  // gemm
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_haswell_asm_6x16,
	  BLIS_GEMM_UKR,       BLIS_DOUBLE,   bli_dgemm_haswell_asm_6x8,
	  BLIS_GEMM_UKR,       BLIS_SCOMPLEX, bli_cgemm_haswell_asm_3x8,
	  BLIS_GEMM_UKR,       BLIS_DCOMPLEX, bli_zgemm_haswell_asm_3x4,

	  // gemmtrsm_l
	  BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsm_l_haswell_asm_6x16, TRUE,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_l_haswell_asm_6x8,  TRUE,
	  // gemmtrsm_u
	  BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsm_u_haswell_asm_6x16,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE,   bli_dgemmtrsm_u_haswell_asm_6x8,

	  // level-3 sup
	  BLIS_GEMMSUP_RRR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n,

	  BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_haswell_asm_6x16m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_haswell_asm_6x16n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_haswell_asm_6x16n,
#if 0
	  BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16m,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n,
	  BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16n,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n,
#endif

	// Update the context with architecture specific threshold functions
	bli_cntx_set_l3_thresh_funcs
	(
	  2,
	  // GEMMT
	  BLIS_GEMMT, bli_cntx_gemmtsup_thresh_is_met_zen,
	  // SYRK
	  BLIS_SYRK,  bli_cntx_syrksup_thresh_is_met_zen,
	  cntx
	);

	// Update the context with optimized packm kernels.
	bli_cntx_set_packm_kers
	(
	  8,
	  BLIS_PACKM_6XK_KER,  BLIS_FLOAT,    bli_spackm_haswell_asm_6xk,
	  BLIS_PACKM_16XK_KER, BLIS_FLOAT,    bli_spackm_haswell_asm_16xk,
	  BLIS_PACKM_6XK_KER,  BLIS_DOUBLE,   bli_dpackm_haswell_asm_6xk,
	  BLIS_PACKM_8XK_KER,  BLIS_DOUBLE,   bli_dpackm_haswell_asm_8xk,
	  BLIS_PACKM_3XK_KER,  BLIS_SCOMPLEX, bli_cpackm_haswell_asm_3xk,
	  BLIS_PACKM_8XK_KER,  BLIS_SCOMPLEX, bli_cpackm_haswell_asm_8xk,
	  BLIS_PACKM_3XK_KER,  BLIS_DCOMPLEX, bli_zpackm_haswell_asm_3xk,
	  BLIS_PACKM_4XK_KER,  BLIS_DCOMPLEX, bli_zpackm_haswell_asm_4xk,
	  cntx
	);

	// Update the context with optimized level-1f kernels.
	bli_cntx_set_l1f_kers
	(
	  12,
	  // axpyf
	  BLIS_AXPYF_KER,     BLIS_FLOAT,    bli_saxpyf_zen_int_5,
	  BLIS_AXPYF_KER,     BLIS_DOUBLE,   bli_daxpyf_zen_int_5,
	  BLIS_AXPYF_KER,     BLIS_SCOMPLEX, bli_caxpyf_zen_int_5,
	  BLIS_AXPYF_KER,     BLIS_DCOMPLEX, bli_zaxpyf_zen_int_5,
	  // dotxaxpyf
	  BLIS_DOTXAXPYF_KER, BLIS_SCOMPLEX, bli_cdotxaxpyf_zen_int_8,
	  BLIS_DOTXAXPYF_KER, BLIS_DCOMPLEX, bli_zdotxaxpyf_zen_int_8,
	  // dotxf
	  BLIS_DOTXF_KER,     BLIS_FLOAT,    bli_sdotxf_zen_int_8,
	  BLIS_DOTXF_KER,     BLIS_DOUBLE,   bli_ddotxf_zen_int_8,
	  BLIS_DOTXF_KER,     BLIS_DCOMPLEX, bli_zdotxf_zen_int_6,
	  BLIS_DOTXF_KER,     BLIS_SCOMPLEX, bli_cdotxf_zen_int_6,
	  // axpy2v
	  BLIS_AXPY2V_KER,    BLIS_DOUBLE,   bli_daxpy2v_zen_int,
	  BLIS_AXPY2V_KER,    BLIS_DCOMPLEX, bli_zaxpy2v_zen_int,
	  cntx
	);

	// Update the context with optimized level-1v kernels.
	bli_cntx_set_l1v_kers
	(
	  40,
	  // addv
	  BLIS_ADDV_KER,  BLIS_FLOAT,      bli_saddv_zen_int,
	  BLIS_ADDV_KER,  BLIS_DOUBLE,     bli_daddv_zen_int,
	  BLIS_ADDV_KER,  BLIS_SCOMPLEX,   bli_caddv_zen_int,
	  BLIS_ADDV_KER,  BLIS_DCOMPLEX,   bli_zaddv_zen_int,

	  // amaxv
	  BLIS_AMAXV_KER,  BLIS_FLOAT,    bli_samaxv_zen_int,
	  BLIS_AMAXV_KER,  BLIS_DOUBLE,   bli_damaxv_zen_int,

	  // axpbyv
	  BLIS_AXPBYV_KER, BLIS_FLOAT,    bli_saxpbyv_zen_int_10,
	  BLIS_AXPBYV_KER, BLIS_DOUBLE,   bli_daxpbyv_zen_int_10,
	  BLIS_AXPBYV_KER, BLIS_SCOMPLEX, bli_caxpbyv_zen_int,
	  BLIS_AXPBYV_KER, BLIS_DCOMPLEX, bli_zaxpbyv_zen_int,

	  // axpyv
	  BLIS_AXPYV_KER,  BLIS_FLOAT,    bli_saxpyv_zen_int_10,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE,   bli_daxpyv_zen_int_10,
	  BLIS_AXPYV_KER,  BLIS_SCOMPLEX, bli_caxpyv_zen_int_5,
	  BLIS_AXPYV_KER,  BLIS_DCOMPLEX, bli_zaxpyv_zen_int_5,

	  // dotv
	  BLIS_DOTV_KER,   BLIS_FLOAT,    bli_sdotv_zen_int_10,
	  BLIS_DOTV_KER,   BLIS_DOUBLE,   bli_ddotv_zen_int_10,
	  BLIS_DOTV_KER,   BLIS_SCOMPLEX, bli_cdotv_zen_int_5,
	  BLIS_DOTV_KER,   BLIS_DCOMPLEX, bli_zdotv_zen_int_5,

	  // dotxv
	  BLIS_DOTXV_KER,  BLIS_FLOAT,    bli_sdotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DOUBLE,   bli_ddotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DCOMPLEX, bli_zdotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_SCOMPLEX, bli_cdotxv_zen_int,

	  // scalv
	  BLIS_SCALV_KER,  BLIS_FLOAT,    bli_sscalv_zen_int_10,
	  BLIS_SCALV_KER,  BLIS_DOUBLE,   bli_dscalv_zen_int_10,
	  BLIS_SCALV_KER,  BLIS_SCOMPLEX, bli_cscalv_zen_int,
	  BLIS_SCALV_KER,  BLIS_DCOMPLEX, bli_zscalv_zen_int,

	  // swapv
	  BLIS_SWAPV_KER,  BLIS_FLOAT,    bli_sswapv_zen_int_8,
	  BLIS_SWAPV_KER,  BLIS_DOUBLE,   bli_dswapv_zen_int_8,

	  // copyv
	  BLIS_COPYV_KER,  BLIS_FLOAT,    bli_scopyv_zen_int,
	  BLIS_COPYV_KER,  BLIS_DOUBLE,   bli_dcopyv_zen_int,
	  BLIS_COPYV_KER,  BLIS_SCOMPLEX, bli_ccopyv_zen_int,
	  BLIS_COPYV_KER,  BLIS_DCOMPLEX, bli_zcopyv_zen_int,

	  // setv
	  BLIS_SETV_KER,   BLIS_FLOAT,    bli_ssetv_zen_int,
	  BLIS_SETV_KER,   BLIS_DOUBLE,   bli_dsetv_zen_int,
	  BLIS_SETV_KER,   BLIS_SCOMPLEX, bli_csetv_zen_int,
	  BLIS_SETV_KER,   BLIS_DCOMPLEX, bli_zsetv_zen_int,

	  // scal2v
	  BLIS_SCAL2V_KER, BLIS_FLOAT,    bli_sscal2v_zen_int,
	  BLIS_SCAL2V_KER, BLIS_DOUBLE,   bli_dscal2v_zen_int,
	  BLIS_SCAL2V_KER, BLIS_SCOMPLEX, bli_cscal2v_zen_int,
	  BLIS_SCAL2V_KER, BLIS_DCOMPLEX, bli_zscal2v_zen_int,
	  cntx
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],     6,     6,     3,     3 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,     8,     8,     4 );
#if AOCL_BLIS_MULTIINSTANCE
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,   240,   144,    18 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   512,   256,   566 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4080,  2040,  4080,   256 );
#else
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,    72,   144,    18 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   256,   256,   566 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4080,  4080,  4080,   256 );
#endif

	bli_blksz_init_easy( &blkszs[ BLIS_AF ],     5,     5,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_DF ],     8,     8,    -1,    -1 );

	// Initialize sup thresholds with architecture-appropriate values.
	//                                          s     d     c     z
#if 1
	bli_blksz_init_easy( &blkszs[ BLIS_MT ],  500,  249,   -1,   -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ],  500,  249,   -1,   -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ],  500,  249,   -1,   -1 );
#else
	bli_blksz_init_easy( &blkszs[ BLIS_MT ], 100000, 100000,   -1,   -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ], 100000, 100000,   -1,   -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ], 100000, 100000,   -1,   -1 );
#endif

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                               s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR_SUP ],     6,     6,    -1,    -1,
	                                                 9,     9,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR_SUP ],    16,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],   168,    72,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],   256,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ],  4080,  4080,    -1,    -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  cntx,

	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

	  // level-1f
	  BLIS_AF, &blkszs[ BLIS_AF ], BLIS_AF,
	  BLIS_DF, &blkszs[ BLIS_DF ], BLIS_DF,

	  // sup thresholds
	  BLIS_MT, &blkszs[ BLIS_MT ], BLIS_MT,
	  BLIS_NT, &blkszs[ BLIS_NT ], BLIS_NT,
	  BLIS_KT, &blkszs[ BLIS_KT ], BLIS_KT,

	// Initialize TRSM blocksize objects with architecture-specific values.
	// Using different cache block sizes for TRSM instead of common level-3 block sizes.
	// Tuning is done for double-precision only.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,    72,   144,    72 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   256,   492,   256,   256 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4080,  1600,  4080,  4080 );

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
	bli_blksz_init_easy( &thresh[ BLIS_MT ],   512,   256,   380,   110 );
	bli_blksz_init_easy( &thresh[ BLIS_NT ],   200,   256,   256,   128 );
	bli_blksz_init_easy( &thresh[ BLIS_KT ],   240,   220,   220,   110 );

	  BLIS_VA_END
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
	  30,
	  BLIS_RRR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m, TRUE,
	  BLIS_RRC, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8m, TRUE,
	  BLIS_RCR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m, TRUE,
	  BLIS_RCC, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n, TRUE,
	  BLIS_CRR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8m, TRUE,
	  BLIS_CRC, BLIS_DOUBLE, bli_dgemmsup_rd_haswell_asm_6x8n, TRUE,
	  BLIS_CCR, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n, TRUE,
	  BLIS_CCC, BLIS_DOUBLE, bli_dgemmsup_rv_haswell_asm_6x8n, TRUE,

	  BLIS_RRR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m, TRUE,
	  BLIS_RRC, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16m, TRUE,
	  BLIS_RCR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m, TRUE,
	  BLIS_RCC, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n, TRUE,
	  BLIS_CRR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16m, TRUE,
	  BLIS_CRC, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x16n, TRUE,
	  BLIS_CCR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n, TRUE,
	  BLIS_CCC, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x16n, TRUE,

	  BLIS_RRR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_RCR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_CRR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_RCC, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,
	  BLIS_CCR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,
	  BLIS_CCC, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,

	  BLIS_RRR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4m, TRUE,
	  BLIS_RRC, BLIS_DCOMPLEX, bli_zgemmsup_rd_zen_asm_3x4m, TRUE,
	  BLIS_RCR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4m, TRUE,
	  BLIS_RCC, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4n, TRUE,
	  BLIS_CRR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4m, TRUE,
	  BLIS_CRC, BLIS_DCOMPLEX, bli_zgemmsup_rd_zen_asm_3x4n, TRUE,
	  BLIS_CCR, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4n, TRUE,
	  BLIS_CCC, BLIS_DCOMPLEX, bli_zgemmsup_rv_zen_asm_3x4n, TRUE,
	  cntx
	);

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR ],     6,     6,     3,     3,
	                                             9,     9,     3,     3 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,     8,     8,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   144,    72,    72,    36 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,   256,   128,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8160,  4080,  2040,  1020 );

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
#endif
}
