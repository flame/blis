/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
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

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffb,
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   alpha,
                           void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                           void*   b, inc_t rs_b, inc_t cs_b, inc_t ps_b,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T GENARRAY(ftypes,trsm_ru_ker_var2);


void bli_trsm_ru_ker_var2( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           obj_t*  beta,
                           obj_t*  c,
                           trsm_t* cntl )
{
	num_t     dt_exec   = bli_obj_execution_datatype( *c );

	doff_t    diagoffb  = bli_obj_diag_offset( *b );

	dim_t     m         = bli_obj_length( *c );
	dim_t     n         = bli_obj_width( *c );
	dim_t     k         = bli_obj_width( *a );

	void*     buf_a     = bli_obj_buffer_at_off( *a );
	inc_t     rs_a      = bli_obj_row_stride( *a );
	inc_t     cs_a      = bli_obj_col_stride( *a );
	inc_t     ps_a      = bli_obj_panel_stride( *a );

	void*     buf_b     = bli_obj_buffer_at_off( *b );
	inc_t     rs_b      = bli_obj_row_stride( *b );
	inc_t     cs_b      = bli_obj_col_stride( *b );
	inc_t     ps_b      = bli_obj_panel_stride( *b );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	num_t     dt_alpha;
	void*     buf_alpha;

	FUNCPTR_T f;

	// If alpha is a scalar constant, use dt_exec to extract the address of the
	// corresponding constant value; otherwise, use the datatype encoded
	// within the alpha object and extract the buffer at the alpha offset.
	bli_set_scalar_dt_buffer( alpha, dt_exec, dt_alpha, buf_alpha );

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( diagoffb,
	   m,
	   n,
	   k,
	   buf_alpha,
	   buf_a, rs_a, cs_a, ps_a,
	   buf_b, rs_b, cs_b, ps_b,
	   buf_c, rs_c, cs_c );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, gemmtrsmukr, gemmukr ) \
\
void PASTEMAC(ch,varname)( \
                           doff_t  diagoffb, \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   alpha, \
                           void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a, \
                           void*   b, inc_t rs_b, inc_t cs_b, inc_t ps_b, \
                           void*   c, inc_t rs_c, inc_t cs_c \
                         ) \
{ \
	/* Temporary C buffer for edge cases. */ \
	ctype           ct[ PASTEMAC(ch,nr) * \
	                    PASTEMAC(ch,mr) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t     rs_ct      = 1; \
	const inc_t     cs_ct      = PASTEMAC(ch,nr); \
\
	/* Alias constants to shorter names. */ \
	const dim_t     MR         = PASTEMAC(ch,nr); \
	const dim_t     NR         = PASTEMAC(ch,mr); \
	const dim_t     PACKMR     = PASTEMAC(ch,packnr); \
	const dim_t     PACKNR     = PASTEMAC(ch,packmr); \
\
	ctype* restrict zero       = PASTEMAC(ch,0); \
	ctype* restrict minus_one  = PASTEMAC(ch,m1); \
	ctype* restrict a_cast     = a; \
	ctype* restrict b_cast     = b; \
	ctype* restrict c_cast     = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict a1; \
	ctype* restrict b1; \
	ctype* restrict c1; \
	ctype* restrict c11; \
	ctype* restrict a10; \
	ctype* restrict a11; \
	ctype* restrict b01; \
	ctype* restrict b11; \
	ctype* restrict a2; \
	ctype* restrict b2; \
\
	doff_t          diagoffb_j; \
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	dim_t           k_b0111; \
	dim_t           k_b01; \
	dim_t           off_b01; \
	dim_t           off_b11; \
	dim_t           i, j; \
	dim_t           rstep_a; \
	dim_t           cstep_b; \
	dim_t           rstep_c, cstep_c; \
\
	/*
	   Assumptions/assertions:
         rs_a == 1
	     cs_a == GEMM_NR
	     ps_a == stride to next row panel of A
         rs_b == GEMM_MR
	     cs_b == 1
	     ps_b == stride to next column panel of B
         rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/ \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Safeguard: If the current panel of B is entirely below its diagonal,
	   it is implicitly zero. So we do nothing. */ \
	if ( bli_is_strictly_below_diag_n( diagoffb, k, n ) ) return; \
\
	/* If there is a zero region to the left of where the diagonal of B
	   intersects the top edge of the panel, adjust the pointer to C and
	   treat this case as if the diagonal offset were zero. This skips over
	   the region (in increments of NR) that was not packed. (Note we skip
	   in increments of NR since that is how the region would have been
	   skipped by packm.) */ \
	if ( diagoffb > 0 ) \
	{ \
		j        = ( diagoffb / NR ) * NR; \
		n        = n - j; \
		diagoffb = diagoffb % NR; \
		c_cast   = c_cast + (j  )*cs_c; \
	} \
\
	/* If there is a zero region below where the diagonal of B intersects the
	   right side of the block, shrink it to prevent "no-op" iterations from
	   executing. */ \
	if ( -diagoffb + n < k ) \
	{ \
		k = -diagoffb + n; \
	} \
\
	/* Check the k dimension, which needs to be a multiple of NR. If k
	   isn't a multiple of NR, we adjust it higher to satisfy the micro-
	   kernel, which is expecting to perform an NR x NR triangular solve.
	   This adjustment of k is consistent with what happened when B was
	   packed: all of its bottom/right edges were zero-padded, and
	   furthermore, the panel that stores the bottom-right corner of the
	   matrix has its diagonal extended into the zero-padded region (as
	   identity). This allows the trsm of that bottom-right panel to
	   proceed without producing any infs or NaNs that would infect the
	   "good" values of the corresponding block of A. */ \
	if ( k % NR != 0 ) k += NR - ( k % NR ); \
\
	/* NOTE: We don't need to check that n is a multiple of PACKNR since we
	   know that the underlying buffer was already allocated to have an n
	   dimension that is a multiple of PACKNR, with the region between the
	   last column and the next multiple of NR zero-padded accordingly. */ \
\
	/* Clear the temporary C buffer in case it has any infs or NaNs. */ \
	PASTEMAC(ch,set0s_mxn)( MR, NR, \
	                        ct, rs_ct, cs_ct ); \
\
	/* Compute number of primary and leftover components of the m and n
       dimensions. */ \
	n_iter = n / NR; \
	n_left = n % NR; \
\
	m_iter = m / MR; \
	m_left = m % MR; \
\
	if ( n_left ) ++n_iter; \
	if ( m_left ) ++m_iter; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	rstep_a = ps_a; \
\
	cstep_b = k * PACKNR; \
\
	rstep_c = rs_c * MR; \
	cstep_c = cs_c * NR; \
\
	b1 = b_cast; \
	c1 = c_cast; \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = 0; j < n_iter; ++j ) \
	{ \
		diagoffb_j = diagoffb - ( doff_t )j*NR; \
		a1         = a_cast; \
		c11        = c1; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Determine the offset to and length of the panel that was packed
		   so we can index into the corresponding location in A. */ \
		off_b01   = 0; \
		k_b0111   = bli_min( k, -diagoffb_j + NR ); \
		k_b01     = k_b0111 - NR; \
		off_b11   = k_b01; \
\
		/* Compute the addresses of the panel B10 and the triangular
		   block B11. */ \
		b01       = b1; \
		b11       = b1 + k_b01 * PACKNR; \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b1; \
\
		/* If the current panel of B intersects the diagonal, use a
		   special micro-kernel that performs a fused gemm and trsm.
		   If the current panel of B resides above the diagonal, use a
		   a regular gemm micro-kernel. Otherwise, if it is below the
		   diagonal, it was not packed (because it is implicitly zero)
		   and so we do nothing. */ \
		if ( bli_intersects_diag_n( diagoffb_j, k, NR ) ) \
		{ \
			/* Loop over the m dimension (MR rows at a time). */ \
			for ( i = 0; i < m_iter; ++i ) \
			{ \
				m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
				/* Compute the addresses of the A10 panel and A11 block. */ \
				a10  = a1 + off_b01 * PACKMR; \
				a11  = a1 + off_b11 * PACKMR; \
\
				/* Compute the addresses of the next panels of A and B. */ \
                a2 = a1 + rstep_a; \
				if ( i == m_iter - 1 ) \
				{ \
					a2 = a_cast; \
					b2 = b1 + k_b0111 * PACKNR; \
					if ( j == n_iter - 1 ) \
						b2 = b_cast; \
				} \
\
				/* Handle interior and edge cases separately. */ \
				if ( m_cur == MR && n_cur == NR ) \
				{ \
					/* Invoke the fused gemm/trsm micro-kernel. */ \
					PASTEMAC(ch,gemmtrsmukr)( k_b01, \
					                          alpha_cast, \
					                          b01, \
					                          b11, \
					                          a10, \
					                          a11, \
					                          a11, \
					                          c11, cs_c, rs_c, \
					                          b2, a2 ); \
				} \
				else \
				{ \
					/* Invoke the fused gemm/trsm micro-kernel. */ \
					PASTEMAC(ch,gemmtrsmukr)( k_b01, \
					                          alpha_cast, \
					                          b01, \
					                          b11, \
					                          a10, \
					                          a11, \
					                          a11, \
					                          ct, cs_ct, rs_ct, \
					                          b2, a2 ); \
\
					/* Copy the result to the bottom edge of C. */ \
					PASTEMAC(ch,copys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        c11, rs_c,  cs_c ); \
				} \
\
				a1  += rstep_a; \
				c11 += rstep_c; \
			} \
		} \
		else if ( bli_is_strictly_above_diag_n( diagoffb_j, k, NR ) ) \
		{ \
			/* Loop over the m dimension (MR rows at a time). */ \
			for ( i = 0; i < m_iter; ++i ) \
			{ \
				m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
				/* Compute the addresses of the next panels of A and B. */ \
                a2 = a1 + rstep_a; \
				if ( i == m_iter - 1 ) \
				{ \
					a2 = a_cast; \
					b2 = b1 + cstep_b; \
					if ( j == n_iter - 1 ) \
						b2 = b_cast; \
				} \
\
				/* Handle interior and edge cases separately. */ \
				if ( m_cur == MR && n_cur == NR ) \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					PASTEMAC(ch,gemmukr)( k, \
					                      minus_one, \
					                      b1, \
					                      a1, \
					                      alpha_cast, \
					                      c11, cs_c, rs_c, \
					                      b2, a2 ); \
				} \
				else \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					PASTEMAC(ch,gemmukr)( k, \
					                      minus_one, \
					                      b1, \
					                      a1, \
					                      zero, \
					                      ct, cs_ct, rs_ct, \
					                      b2, a2 ); \
\
					/* Add the result to the edge of C. */ \
					PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        alpha_cast, \
					                        c11, rs_c,  cs_c ); \
				} \
\
				a1  += rstep_a; \
				c11 += rstep_c; \
			} \
		} \
\
		b1 += k_b0111 * PACKNR; \
		c1 += cstep_c; \
	} \
\
/*
PASTEMAC(ch,fprintm)( stdout, "trsm_ru_ker_var2: a1 (diag)", MR, k_a1011, a1, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_ru_ker_var2: a11 (diag)", MR, MR, a11, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_ru_ker_var2: b1 (diag)", k_a1011, NR, bp_i, NR, 1, "%5.2f", "" );  \
PASTEMAC(ch,fprintm)( stdout, "trsm_ru_ker_var2: bp11 (diag)", MR, NR, bp11, NR, 1, "%5.2f", "" );  \
*/ \
\
/*
PASTEMAC(ch,fprintm)( stdout, "trsm_ru_ker_var2: a1 (ndiag)", MR, k, a1, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_ru_ker_var2: b1 (ndiag)", k, NR, bp, NR, 1, "%5.2f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC2( trsm_ru_ker_var2, GEMMTRSM_L_UKERNEL, GEMM_UKERNEL )

