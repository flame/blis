/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_cntx_init_skx( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_skx_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  2,
	  // gemm
	  BLIS_GEMM_UKR,       BLIS_FLOAT ,   bli_sgemm_skx_asm_32x12_l2,   FALSE,
	  BLIS_GEMM_UKR,       BLIS_DOUBLE,   bli_dgemm_skx_asm_16x14,      FALSE,
	  cntx
	);

	// Update the context with optimized level-1f kernels.
	bli_cntx_set_l1f_kers
	(
	  4,
	  // axpyf
	  BLIS_AXPYF_KER,     BLIS_FLOAT,  bli_saxpyf_zen_int_8,
	  BLIS_AXPYF_KER,     BLIS_DOUBLE, bli_daxpyf_zen_int_8,
	  // dotxf
	  BLIS_DOTXF_KER,     BLIS_FLOAT,  bli_sdotxf_zen_int_8,
	  BLIS_DOTXF_KER,     BLIS_DOUBLE, bli_ddotxf_zen_int_8,
	  cntx
	);

	// Update the context with optimized level-1v kernels.
	bli_cntx_set_l1v_kers
	(
	  10,
#if 1
	  // amaxv
	  BLIS_AMAXV_KER,  BLIS_FLOAT,  bli_samaxv_zen_int,
	  BLIS_AMAXV_KER,  BLIS_DOUBLE, bli_damaxv_zen_int,
#endif
	  // axpyv
#if 0
	  BLIS_AXPYV_KER,  BLIS_FLOAT,  bli_saxpyv_zen_int,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE, bli_daxpyv_zen_int,
#else
	  BLIS_AXPYV_KER,  BLIS_FLOAT,  bli_saxpyv_zen_int10,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE, bli_daxpyv_zen_int10,
#endif
	  // dotv
	  BLIS_DOTV_KER,   BLIS_FLOAT,  bli_sdotv_zen_int,
	  BLIS_DOTV_KER,   BLIS_DOUBLE, bli_ddotv_zen_int,
	  // dotxv
	  BLIS_DOTXV_KER,  BLIS_FLOAT,  bli_sdotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DOUBLE, bli_ddotxv_zen_int,
	  // scalv
#if 0
	  BLIS_SCALV_KER,  BLIS_FLOAT,  bli_sscalv_zen_int,
	  BLIS_SCALV_KER,  BLIS_DOUBLE, bli_dscalv_zen_int,
#else
	  BLIS_SCALV_KER,  BLIS_FLOAT,  bli_sscalv_zen_int10,
	  BLIS_SCALV_KER,  BLIS_DOUBLE, bli_dscalv_zen_int10,
#endif
	  cntx
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,    16,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    12,    14,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   480,   240,    -1,    -1 );
	bli_blksz_init     ( &blkszs[ BLIS_KC ],   384,   256,    -1,    -1,
	                                           480,   320,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  3072,  3752,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_AF ],     8,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_DF ],     8,     8,    -1,    -1 );

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

	bli_cntx_set_l3_sup_kers
	(
	  30,
	  BLIS_RRR, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_RRC, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_RCR, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_RCC, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_CRR, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_CRC, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_CCR, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,
	  BLIS_CCC, BLIS_DOUBLE, bli_dgemmsup_rv_zen4_asm_24x8m, FALSE,

	  BLIS_RRR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x64m_avx512, TRUE,
	  BLIS_RRC, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x64m_avx512, TRUE,
	  BLIS_RCR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x64m_avx512, TRUE,
	  BLIS_RCC, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x64n_avx512, TRUE,
	  BLIS_CRR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x64m_avx512, TRUE,
	  BLIS_CRC, BLIS_FLOAT, bli_sgemmsup_rd_zen_asm_6x64n_avx512, TRUE,
	  BLIS_CCR, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x64n_avx512, TRUE,
	  BLIS_CCC, BLIS_FLOAT, bli_sgemmsup_rv_zen_asm_6x64n_avx512, TRUE,

	  BLIS_RRR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_RCR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_CRR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8m, TRUE,
	  BLIS_RCC, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,
	  BLIS_CCR, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,
	  BLIS_CCC, BLIS_SCOMPLEX, bli_cgemmsup_rv_zen_asm_3x8n, TRUE,

	  BLIS_RRR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_RRC, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_RCR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_RCC, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CRR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CRC, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CCR, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  BLIS_CCC, BLIS_DCOMPLEX, bli_zgemmsup_cv_zen4_asm_12x4m, FALSE,
	  cntx
	);

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init     ( &blkszs[ BLIS_MR ],     6,    24,     3,    12,
	                                             6,     9,     3,    12 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    64,     8,     8,     4 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   192,   144,    72,    48 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,   480,   128,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8064,  4080,  2040,  1020 );

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

}

