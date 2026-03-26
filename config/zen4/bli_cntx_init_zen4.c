/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022 - 2026, Advanced Micro Devices, Inc. All rights reserved.

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

void bli_cntx_init_zen4( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_zen4_ref( cntx );

	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,
	  // axpyf
	  BLIS_AXPYF_KER,  BLIS_FLOAT,  bli_saxpyf_zen_int_5,
	  BLIS_AXPYF_KER,  BLIS_DOUBLE, bli_daxpyf_zen_int_5,
	  BLIS_AXPYF_KER,  BLIS_SCOMPLEX, bli_caxpyf_zen_int_5,
	  BLIS_AXPYF_KER,  BLIS_DCOMPLEX, bli_zaxpyf_zen_int_5,

	  // dotxf
	  BLIS_DOTXF_KER,  BLIS_FLOAT,  bli_sdotxf_zen_int_8,
	  BLIS_DOTXF_KER,  BLIS_DOUBLE, bli_ddotxf_zen_int_8,

	  // amaxv
	  BLIS_AMAXV_KER,  BLIS_FLOAT,  bli_samaxv_zen_int,
	  BLIS_AMAXV_KER,  BLIS_DOUBLE, bli_damaxv_zen_int,

	  // axpyv
	  BLIS_AXPYV_KER,  BLIS_FLOAT,  bli_saxpyv_zen_int_10,
	  BLIS_AXPYV_KER,  BLIS_DOUBLE, bli_daxpyv_zen_int_10,
	  BLIS_AXPYV_KER,  BLIS_SCOMPLEX,  bli_caxpyv_zen_int_5,
	  BLIS_AXPYV_KER,  BLIS_DCOMPLEX, bli_zaxpyv_zen_int_5,

	  // dotv
	  BLIS_DOTV_KER,   BLIS_FLOAT,  bli_sdotv_zen_int10,
	  BLIS_DOTV_KER,   BLIS_DOUBLE, bli_ddotv_zen_int10,

	  // dotxv
	  BLIS_DOTXV_KER,  BLIS_FLOAT,  bli_sdotxv_zen_int,
	  BLIS_DOTXV_KER,  BLIS_DOUBLE, bli_ddotxv_zen_int,

	  // scalv
	  BLIS_SCALV_KER,  BLIS_FLOAT,  bli_sscalv_zen_int10,
	  BLIS_SCALV_KER,  BLIS_DOUBLE, bli_dscalv_zen_int10,

	  // swapv
	  BLIS_SWAPV_KER,  BLIS_FLOAT,  bli_sswapv_zen_int8,
	  BLIS_SWAPV_KER,  BLIS_DOUBLE, bli_dswapv_zen_int8,

	  // copyv
	  BLIS_COPYV_KER,  BLIS_FLOAT,  bli_scopyv_zen_int,
	  BLIS_COPYV_KER,  BLIS_DOUBLE, bli_dcopyv_zen_int,

	  // setv
	  BLIS_SETV_KER,  BLIS_FLOAT,  bli_ssetv_zen_int,
	  BLIS_SETV_KER,  BLIS_DOUBLE, bli_dsetv_zen_int,

    // addv
	  BLIS_ADDV_KER,  BLIS_DOUBLE,     bli_daddv_zen4_int,

    BLIS_VA_END
	);
}
