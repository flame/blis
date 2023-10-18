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
#include "bli_armsve_config_utils.h"

void bli_cntx_init_armsve( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
#if 0
	blksz_t thresh[ BLIS_NUM_THRESH ];
#endif

	// Set default kernel blocksizes and functions.
	bli_cntx_init_armsve_ref( cntx );

	// -------------------------------------------------------------------------

	// Block size.
	dim_t m_r_s, n_r_s, k_c_s, m_c_s, n_c_s;
	dim_t m_r_d, n_r_d, k_c_d, m_c_d, n_c_d;
	bli_s_blksz_armsve(&m_r_s, &n_r_s, &k_c_s, &m_c_s, &n_c_s);
	bli_d_blksz_armsve(&m_r_d, &n_r_d, &k_c_d, &m_c_d, &n_c_d);

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  2,
	  // These are vector-length agnostic kernels. Yet knowing mr is required at runtime.
	  BLIS_GEMM_UKR, BLIS_FLOAT,  bli_sgemm_armsve_asm_2vx10_unindexed, FALSE,
	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_armsve_asm_2vx10_unindexed, FALSE,
	  cntx
	);

	// Set VL-specific packing routines if applicable.
	if (m_r_d==16)
	  bli_cntx_set_packm_kers
	  (
		3,
		BLIS_PACKM_10XK_KER, BLIS_DOUBLE, bli_dpackm_armsve512_asm_10xk,
		BLIS_PACKM_12XK_KER, BLIS_DOUBLE, bli_dpackm_armsve512_asm_12xk,
		BLIS_PACKM_16XK_KER, BLIS_DOUBLE, bli_dpackm_armsve512_asm_16xk,
		cntx
	  );
	else if (m_r_d==8)
	  bli_cntx_set_packm_kers
	  (
		1,
		BLIS_PACKM_8XK_KER, BLIS_DOUBLE, bli_dpackm_armsve256_asm_8xk,
		cntx
	  );

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ], m_r_s, m_r_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ], n_r_s, n_r_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ], m_c_s, m_c_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ], k_c_s, k_c_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], n_c_s, n_c_d,    -1,    -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  BLIS_NAT, 5,
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
	  cntx
	);

#if 0
	// Initialize sup thresholds with architecture-appropriate values.
	//                                          s     d     c     z
	bli_blksz_init_easy( &thresh[ BLIS_MT ],   -1,  101,   -1,   -1 );
	bli_blksz_init_easy( &thresh[ BLIS_NT ],   -1,  101,   -1,   -1 );
	bli_blksz_init_easy( &thresh[ BLIS_KT ],   -1,  101,   -1,   -1 );

	// Initialize the context with the sup thresholds.
	bli_cntx_set_l3_sup_thresh
	(
	  3,
	  BLIS_MT, &thresh[ BLIS_MT ],
	  BLIS_NT, &thresh[ BLIS_NT ],
	  BLIS_KT, &thresh[ BLIS_KT ],
	  cntx
	);

	// Update the context with optimized small/unpacked gemm kernels.
	bli_cntx_set_l3_sup_kers
	(
	  4,
	  BLIS_RRR, BLIS_DOUBLE, bli_dgemmsup_rv_armsve_10x2v_unindexed, TRUE,
	  BLIS_RCR, BLIS_DOUBLE, bli_dgemmsup_rv_armsve_10x2v_unindexed, TRUE,
	  BLIS_CCR, BLIS_DOUBLE, bli_dgemmsup_rv_armsve_10x2v_unindexed, TRUE,
	  BLIS_CCC, BLIS_DOUBLE, bli_dgemmsup_rv_armsve_10x2v_unindexed, TRUE,
	  cntx
	);

	// Initialize level-3 sup blocksize objects with architecture-specific
	// values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    -1, n_r_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    -1, m_r_d,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],    -1,   120,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    -1,   256,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],    -1,  2048,    -1,    -1 );

	// Update the context with the current architecture's register and cache
	// blocksizes for small/unpacked level-3 problems.
	bli_cntx_set_l3_sup_blkszs
	(
	  5,
	  BLIS_NC, &blkszs[ BLIS_NC ],
	  BLIS_KC, &blkszs[ BLIS_KC ],
	  BLIS_MC, &blkszs[ BLIS_MC ],
	  BLIS_NR, &blkszs[ BLIS_NR ],
	  BLIS_MR, &blkszs[ BLIS_MR ],
	  cntx
	);
#endif
}

