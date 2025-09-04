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

#define bla_syr2k_check( dt_str, op_str, uploa, trans, m, k, lda, ldb, ldc ) \
{ \
	f77_int info = 0; \
	f77_int is_r; \
	f77_int nota, ta, cta; \
	f77_int lower, upper; \
	f77_int nrowa; \
\
	static char* dt_cst = dt_str; \
\
	is_r  = ( dt_cst[0] == 's' || dt_cst[0] == 'd' ); \
	nota  = PASTE_LSAME( trans, "N", (ftnlen)1, (ftnlen)1 ); \
	ta    = PASTE_LSAME( trans, "T", (ftnlen)1, (ftnlen)1 ); \
	cta   = PASTE_LSAME( trans, "C", (ftnlen)1, (ftnlen)1 ); \
	lower = PASTE_LSAME( uploa, "L", (ftnlen)1, (ftnlen)1 ); \
	upper = PASTE_LSAME( uploa, "U", (ftnlen)1, (ftnlen)1 ); \
\
	if ( nota ) { nrowa = *m; } \
	else        { nrowa = *k; } \
\
	if      ( !lower && !upper ) \
		info = 1; \
	else if ( !nota && !ta && (is_r ? !cta : 1) ) \
		info = 2; \
	else if ( *m < 0 ) \
		info = 3; \
	else if ( *k < 0 ) \
		info = 4; \
	else if ( *lda < bli_max( 1, nrowa ) ) \
		info = 7; \
	else if ( *ldb < bli_max( 1, nrowa ) ) \
		info = 9; \
	else if ( *ldc < bli_max( 1, *m    ) ) \
		info = 12; \
\
	if ( info != 0 ) \
	{ \
		char func_str[ BLIS_MAX_BLAS_FUNC_STR_LENGTH ]; \
\
		sprintf( func_str, "%s%-5s", dt_str, op_str ); \
\
		bli_string_mkupper( func_str ); \
\
		PASTE_XERBLA( func_str, &info, (ftnlen)6 ); \
\
		AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_1); \
\
		/* Finalize BLIS. */ \
		bli_finalize_auto(); \
\
		return; \
	} \
}
