/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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

/* This function determines the ideal blocksizes for given datatype
   and num_threads.
*/
void bli_dynamic_blkszs_zen6( dim_t n_threads, cntx_t* cntx, num_t dt )
{
	// dynamic blocksizes enabled only for double datatype.
	if (dt != BLIS_DOUBLE) return;

	blksz_t blkszs[ BLIS_NUM_BLKSZS ];
	dim_t mc, kc, nc;

	// determine ideal blocksize
    if (n_threads == 1 )
    {
        mc = 88, kc = 384, nc = 4032;
    }
    else if (n_threads <= 32)
    {
        // these blocksizes are tuned for M >> K, N >> K and K < 500
        mc = 160, kc = 512, nc = 2016;
    }
    else
    {
        mc = 144, kc = 512, nc = 4032;
    }


	// set blocksizes
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],   192,  mc,    72,    48 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],   512,  kc,   128,    64 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],  8064,  nc,  2040,  1020 );

	bli_cntx_set_blkszs
	(
		BLIS_NAT, 3,
		BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
		BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
		BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
		cntx
	);
}