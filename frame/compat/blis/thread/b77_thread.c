/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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


//
// Define Fortran-compatible BLIS interfaces.
//

void PASTEF770(bli_thread_set_ways)
     (
       const f77_int* jc,
       const f77_int* pc,
       const f77_int* ic,
       const f77_int* jr,
       const f77_int* ir
     )
{
	dim_t jc0 = *jc;
	dim_t pc0 = *pc;
	dim_t ic0 = *ic;
	dim_t jr0 = *jr;
	dim_t ir0 = *ir;

	// Initialize BLIS.
	//bli_init_auto();

	// Convert/typecast negative values to zero.
	//bli_convert_blas_dim1( *jc, jc0 );
	//bli_convert_blas_dim1( *pc, pc0 );
	//bli_convert_blas_dim1( *ic, ic0 );
	//bli_convert_blas_dim1( *jr, jr0 );
	//bli_convert_blas_dim1( *ir, ir0 );

	// Call the BLIS function.
	bli_thread_set_ways( jc0, pc0, ic0, jr0, ir0 );

	// Finalize BLIS.
	//bli_finalize_auto();
}

void PASTEF770(bli_thread_set_num_threads)
     (
       const f77_int* nt
     )
{
	dim_t nt0 = *nt;

	// Initialize BLIS.
	//bli_init_auto();

	// Convert/typecast negative values to zero.
	//bli_convert_blas_dim1( *nt, nt0 );

	// Call the BLIS function.
	bli_thread_set_num_threads( nt0 );

	// Finalize BLIS.
	//bli_finalize_auto();
}

f77_int PASTEF770(bli_thread_get_jc_nt)
     (
     )
{
	// Call the BLIS function.
	dim_t info_value = bli_thread_get_jc_nt();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}
f77_int PASTEF770(bli_thread_get_pc_nt)
     (
     )
{
	// Call the BLIS function.
	dim_t info_value = bli_thread_get_pc_nt();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}
f77_int PASTEF770(bli_thread_get_ic_nt)
     (
     )
{
	// Call the BLIS function.
	dim_t info_value = bli_thread_get_ic_nt();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}
f77_int PASTEF770(bli_thread_get_jr_nt)
     (
     )
{
	// Call the BLIS function.
	dim_t info_value = bli_thread_get_jr_nt();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}
f77_int PASTEF770(bli_thread_get_ir_nt)
     (
     )
{
	// Call the BLIS function.
	dim_t info_value = bli_thread_get_ir_nt();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}

f77_int PASTEF770(bli_thread_get_num_threads)
     (
     )
{
	// Call the BLIS function.
	dim_t info_value = bli_thread_get_num_threads();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}

f77_int PASTEF770(bli_info_get_info_value)
     (
     )
{
	// Call the BLIS function.
	gint_t info_value = bli_info_get_info_value();
	f77_int f77_info_value = (f77_int) info_value;

	return f77_info_value;
}

void PASTEF770(bli_thread_reset)
     (
     )
{
	// Call the BLIS function.
	bli_thread_reset();
}
