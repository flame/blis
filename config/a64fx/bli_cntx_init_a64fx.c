/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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
#include "bli_a64fx_sector_cache.h"

void bli_cntx_init_a64fx( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_a64fx_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,    bli_sgemm_armsve_asm_2vx10_unindexed,
	  BLIS_GEMM_UKR, BLIS_DOUBLE,   bli_dgemm_armsve_asm_2vx10_unindexed,
	  BLIS_GEMM_UKR, BLIS_SCOMPLEX, bli_cgemm_armsve_asm_2vx10_unindexed,
	  BLIS_GEMM_UKR, BLIS_DCOMPLEX, bli_zgemm_armsve_asm_2vx10_unindexed,

	  // packm
	  BLIS_PACKM_10XK_KER, BLIS_DOUBLE, bli_dpackm_armsve512_asm_10xk,
	  // 12xk is not used and disabled for GCC 8-9 compatibility.
	  // BLIS_PACKM_12XK_KER, BLIS_DOUBLE, bli_dpackm_armsve512_int_12xk,
	  BLIS_PACKM_16XK_KER, BLIS_DOUBLE, bli_dpackm_armsve512_asm_16xk,

	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,    FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE,   FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_SCOMPLEX, FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DCOMPLEX, FALSE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,    16,    16,     8 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    10,    10,    10,    10 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   256,   128,   192,    96 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  2048,  2048,  1536,  1536 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], 23040, 26880, 11520, 11760 );

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

	  BLIS_VA_END
	);

	// Set A64FX cache sector sizes for each PE/CMG
	// SC Fugaku might disable users' setting cache sizes.
#if !defined(CACHE_SECTOR_SIZE_READONLY)
#pragma omp parallel
	{
	  A64FX_SETUP_SECTOR_CACHE_SIZES(A64FX_SCC(0,1,3,0))
	  A64FX_SETUP_SECTOR_CACHE_SIZES_L2(A64FX_SCC_L2(9,28))
	}
#endif

}

