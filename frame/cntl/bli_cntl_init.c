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
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

void bli_cntl_init( void )
{
	// Level-1
	bli_scalv_cntl_init();
	bli_packv_cntl_init();
	bli_unpackv_cntl_init();

	// Level-1m
	bli_scalm_cntl_init();
	bli_packm_cntl_init();
	bli_unpackm_cntl_init();

	// Level-2
	bli_gemv_cntl_init();
	bli_ger_cntl_init();
	bli_hemv_cntl_init();
	bli_her_cntl_init();
	bli_her2_cntl_init();
	bli_trmv_cntl_init();
	bli_trsv_cntl_init();

	// Level-3
	bli_gemm_cntl_init();
	bli_trsm_cntl_init();

	// Level-3 via 4m
	bli_gemm4m_cntl_init();
	bli_trsm4m_cntl_init();

	// Level-3 via 3m
	bli_gemm3m_cntl_init();
	bli_trsm3m_cntl_init();

	// Level-3 via 4mh
	bli_gemm4mh_cntl_init();

	// Level-3 via 3mh
	bli_gemm3mh_cntl_init();
}

void bli_cntl_finalize( void )
{
	// Level-1
	bli_scalv_cntl_finalize();
	bli_packv_cntl_finalize();
	bli_unpackv_cntl_finalize();

	// Level-1m
	bli_scalm_cntl_finalize();
	bli_packm_cntl_finalize();
	bli_unpackm_cntl_finalize();

	// Level-2
	bli_gemv_cntl_finalize();
	bli_ger_cntl_finalize();
	bli_hemv_cntl_finalize();
	bli_her_cntl_finalize();
	bli_her2_cntl_finalize();
	bli_trmv_cntl_finalize();
	bli_trsv_cntl_finalize();

	// Level-3
	bli_gemm_cntl_finalize();
	bli_trsm_cntl_finalize();

	// Level-3 via 4m
	bli_gemm4m_cntl_finalize();
	bli_trsm4m_cntl_finalize();

	// Level-3 via 3m
	bli_gemm3m_cntl_finalize();
	bli_trsm3m_cntl_finalize();

	// Level-3 via 4mh
	bli_gemm4mh_cntl_finalize();

	// Level-3 via 3mh
	bli_gemm3mh_cntl_finalize();
}

