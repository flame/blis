/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, The University of Texas at Austin

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

// Instantiate prototypes for packm kernels.
PACKM_KER_PROT(    float,  s, packm_6xk_bb4_power10_ref )
PACKM_KER_PROT(    double, d, packm_6xk_bb2_power10_ref )

// Instantiate prototypes for level-3 kernels.
GEMM_UKR_PROT(     float,  s, gemmbb_power10_ref )
GEMMTRSM_UKR_PROT( float,  s, gemmtrsmbb_l_power10_ref )
GEMMTRSM_UKR_PROT( float,  s, gemmtrsmbb_u_power10_ref )
TRSM_UKR_PROT(     float,  s, trsmbb_l_power10_ref )
TRSM_UKR_PROT(     float,  s, trsmbb_u_power10_ref )

GEMM_UKR_PROT(     double, d, gemmbb_power10_ref )
GEMMTRSM_UKR_PROT( double, d, gemmtrsmbb_l_power10_ref )
GEMMTRSM_UKR_PROT( double, d, gemmtrsmbb_u_power10_ref )
TRSM_UKR_PROT(     double, d, trsmbb_l_power10_ref )
TRSM_UKR_PROT(     double, d, trsmbb_u_power10_ref )

GEMM_UKR_PROT(     scomplex, c, gemmbb_power10_ref )
GEMMTRSM_UKR_PROT( scomplex, c, gemmtrsmbb_l_power10_ref )
GEMMTRSM_UKR_PROT( scomplex, c, gemmtrsmbb_u_power10_ref )
TRSM_UKR_PROT(     scomplex, c, trsmbb_l_power10_ref )
TRSM_UKR_PROT(     scomplex, c, trsmbb_u_power10_ref )

GEMM_UKR_PROT(     dcomplex, z, gemmbb_power10_ref )
GEMMTRSM_UKR_PROT( dcomplex, z, gemmtrsmbb_l_power10_ref )
GEMMTRSM_UKR_PROT( dcomplex, z, gemmtrsmbb_u_power10_ref )
TRSM_UKR_PROT(     dcomplex, z, trsmbb_l_power10_ref )
TRSM_UKR_PROT(     dcomplex, z, trsmbb_u_power10_ref )

void bli_cntx_init_power10( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_power10_ref( cntx );

	// -------------------------------------------------------------------------

	// Update the context with optimized native gemm micro-kernels and
	// their storage preferences.
	bli_cntx_set_l3_nat_ukrs
	(
	  12,
	  BLIS_GEMM_UKR,       BLIS_FLOAT,    bli_sgemm_power10_mma_8x16,     TRUE,

	  BLIS_TRSM_L_UKR,     BLIS_FLOAT,    bli_strsmbb_l_power10_ref,      FALSE,
	  BLIS_TRSM_U_UKR,     BLIS_FLOAT,    bli_strsmbb_u_power10_ref,      FALSE,
	  
	  BLIS_GEMM_UKR,       BLIS_DOUBLE,   bli_dgemm_power10_mma_8x8,      TRUE,  
	  
	  BLIS_TRSM_L_UKR,     BLIS_DOUBLE,   bli_dtrsmbb_l_power10_ref,      FALSE,
	  BLIS_TRSM_U_UKR,     BLIS_DOUBLE,   bli_dtrsmbb_u_power10_ref,      FALSE,
	  BLIS_GEMM_UKR,       BLIS_SCOMPLEX, bli_cgemmbb_power10_ref,        FALSE,
	  BLIS_TRSM_L_UKR,     BLIS_SCOMPLEX, bli_ctrsmbb_l_power10_ref,      FALSE,
	  BLIS_TRSM_U_UKR,     BLIS_SCOMPLEX, bli_ctrsmbb_u_power10_ref,      FALSE,
	  BLIS_GEMM_UKR,       BLIS_DCOMPLEX, bli_zgemmbb_power10_ref,        FALSE,
	  BLIS_TRSM_L_UKR,     BLIS_DCOMPLEX, bli_ztrsmbb_l_power10_ref,      FALSE,
	  BLIS_TRSM_U_UKR,     BLIS_DCOMPLEX, bli_ztrsmbb_u_power10_ref,      FALSE,
	  cntx
	);

	// Update the context with customized virtual [gemm]trsm micro-kernels.
	bli_cntx_set_l3_vir_ukrs
	(
	  8,
	  BLIS_GEMMTRSM_L_UKR, BLIS_FLOAT,    bli_sgemmtrsmbb_l_power10_ref,
	  BLIS_GEMMTRSM_U_UKR, BLIS_FLOAT,    bli_sgemmtrsmbb_u_power10_ref,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DOUBLE,   bli_dgemmtrsmbb_l_power10_ref,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DOUBLE,   bli_dgemmtrsmbb_u_power10_ref,
	  BLIS_GEMMTRSM_L_UKR, BLIS_SCOMPLEX, bli_cgemmtrsmbb_l_power10_ref,
	  BLIS_GEMMTRSM_U_UKR, BLIS_SCOMPLEX, bli_cgemmtrsmbb_u_power10_ref,
	  BLIS_GEMMTRSM_L_UKR, BLIS_DCOMPLEX, bli_zgemmtrsmbb_l_power10_ref,
	  BLIS_GEMMTRSM_U_UKR, BLIS_DCOMPLEX, bli_zgemmtrsmbb_u_power10_ref,
	  cntx
	);

	// Update the context with optimized packm kernels.
	bli_cntx_set_packm_kers
	(
	  2,
	  BLIS_PACKM_6XK_KER,  BLIS_FLOAT,    bli_spackm_6xk_bb4_power10_ref,
	  BLIS_PACKM_6XK_KER,  BLIS_DOUBLE,   bli_dpackm_6xk_bb2_power10_ref,
	  cntx
	);

	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],     8,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,     8,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   832,   320,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],  1026,   960,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  4096,  4096,    -1,    -1 );


	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  BLIS_NAT, 5,
	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,
	  cntx
	);

}
