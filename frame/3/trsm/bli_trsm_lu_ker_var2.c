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
                           doff_t  diagoffa,
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   alpha,
                           void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a,
                           void*   b, inc_t rs_b, inc_t cs_b, inc_t ps_b,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T GENARRAY(ftypes,trsm_lu_ker_var2);


void bli_trsm_lu_ker_var2( obj_t*  alpha,
                           obj_t*  a,
                           obj_t*  b,
                           obj_t*  beta,
                           obj_t*  c,
                           trsm_t* cntl )
{
	num_t     dt_exec   = bli_obj_execution_datatype( *c );

	doff_t    diagoffa  = bli_obj_diag_offset( *a );

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
	f( diagoffa,
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
                           doff_t  diagoffa, \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   alpha, \
                           void*   a, inc_t rs_a, inc_t cs_a, inc_t ps_a, \
                           void*   b, inc_t rs_b, inc_t cs_b, inc_t ps_b, \
                           void*   c, inc_t rs_c, inc_t cs_c \
                         ) \
{ \
	/* Temporary buffer for duplicating elements of B. */ \
	ctype           bd[ PASTEMAC(ch,maxkc) * \
	                    PASTEMAC(ch,packnr) * \
	                    PASTEMAC(ch,ndup) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	ctype* restrict bp; \
\
	/* Temporary C buffer for edge cases. */ \
	ctype           ct[ PASTEMAC(ch,mr) * \
	                    PASTEMAC(ch,nr) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const inc_t     rs_ct      = 1; \
	const inc_t     cs_ct      = PASTEMAC(ch,mr); \
\
	/* Alias constants to shorter names. */ \
	const dim_t     MR         = PASTEMAC(ch,mr); \
	const dim_t     NR         = PASTEMAC(ch,nr); \
	const dim_t     PACKMR     = PASTEMAC(ch,packmr); \
	const dim_t     PACKNR     = PASTEMAC(ch,packnr); \
	const dim_t     NDUP       = PASTEMAC(ch,ndup); \
	const bool_t    DUPB       = NDUP != 1; \
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
	ctype* restrict b11; \
	ctype* restrict a12; \
	ctype* restrict a11; \
	ctype* restrict bp21; \
	ctype* restrict bp11; \
	ctype* restrict a2; \
	ctype* restrict b2; \
\
	doff_t          diagoffa_i; \
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	dim_t           k_nr; \
	dim_t           k_a1112; \
	dim_t           k_a11; \
	dim_t           k_a12; \
	dim_t           off_a11; \
	dim_t           off_a12; \
	dim_t           i, j, ib; \
	dim_t           rstep_a; \
	dim_t           cstep_b; \
	dim_t           rstep_c, cstep_c; \
\
	/*
	   Assumptions/assertions:
         rs_a == 1
	     cs_a == GEMM_MR
	     ps_a == stride to next row panel of A
         rs_b == GEMM_NR
	     cs_b == 1
	     ps_b == stride to next column panel of B
         rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/ \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Safeguard: If matrix A is below the diagonal, it is implicitly zero.
	   So we do nothing. */ \
	if ( bli_is_strictly_below_diag_n( diagoffa, m, k ) ) return; \
\
	/* If there is a zero region to the left of where the diagonal of A
	   intersects the top edge of the block, adjust the pointer to B and
	   treat this case as if the diagonal offset were zero. Note that we
	   don't need to adjust the pointer to A since packm would have simply
	   skipped over the region that was not stored. */ \
	if ( diagoffa > 0 ) \
	{ \
		i        = diagoffa; \
		k        = k - i; \
		diagoffa = 0; \
		b_cast   = b_cast + (i  )*rs_b; \
	} \
\
	/* If there is a zero region below where the diagonal of A intersects the
	   right side of the block, shrink it to prevent "no-op" iterations from
	   executing. */ \
	if ( -diagoffa + k < m ) \
	{ \
		m = -diagoffa + k; \
	} \
\
	/* Check the k dimension, which needs to be a multiple of MR. If k
	   isn't a multiple of MR, we adjust it higher to satisfy the micro-
	   kernel, which is expecting to perform an MR x MR triangular solve.
	   This adjustment of k is consistent with what happened when A was
	   packed: all of its bottom/right edges were zero-padded, and
	   furthermore, the panel that stores the bottom-right corner of the
	   matrix has its diagonal extended into the zero-padded region (as
	   identity). This allows the trsm of that bottom-right panel to
	   proceed without producing any infs or NaNs that would infect the
	   "good" values of the corresponding block of B. */ \
	if ( k % MR != 0 ) k += MR - ( k % MR ); \
\
	/* NOTE: We don't need to check that m is a multiple of PACKMR since we
	   know that the underlying buffer was already allocated to have an m
	   dimension that is a multiple of PACKMR, with the region between the
	   last row and the next multiple of MR zero-padded accordingly. */ \
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
	/* Compute the number of elements in B to duplicate per iteration. */ \
	k_a1112 = k; \
	k_nr    = k_a1112 * NR; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	rstep_a = k * PACKMR; \
\
	cstep_b = ps_b; \
\
	rstep_c = rs_c * MR; \
	cstep_c = cs_c * NR; \
\
	b1 = b_cast; \
	c1 = c_cast; \
\
	/* If the micro-kernel needs elements of B duplicated, set bp to
	   point to the duplication buffer. If no duplication is called for,
	   bp will be set to the current column panel of B for each iteration
	   of the outer loop below. */ \
	if ( DUPB ) bp = bd; \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = 0; j < n_iter; ++j ) \
	{ \
		a1  = a_cast; \
		c11 = c1 + (m_iter-1)*rstep_c; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* If duplication is needed, copy the current iteration's NR
		   columns of B to a local buffer with each value duplicated. */ \
		if ( DUPB ) PASTEMAC(ch,dupl)( k_nr, b1, bp ); \
		else        bp = b1; \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b1; \
\
		/* Loop over the m dimension (MR rows at a time). */ \
		for ( ib = 0; ib < m_iter; ++ib ) \
		{ \
			i          = m_iter - 1 - ib; \
			diagoffa_i = diagoffa + ( doff_t )i*MR; \
\
			m_cur = ( bli_is_not_edge_b( ib, m_iter, m_left ) ? MR : m_left ); \
\
			/* If the current panel of A intersects the diagonal, use a
			   special micro-kernel that performs a fused gemm and trsm.
			   If the current panel of A resides above the diagonal, use a
			   a regular gemm micro-kernel. Otherwise, if it is below the
			   diagonal, it was not packed (because it is implicitly zero)
			   and so we do nothing. */ \
			if ( bli_intersects_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				/* Compute various offsets into and lengths of parts of A. */ \
				off_a11 = bli_max( diagoffa_i, 0 ); \
				k_a1112 = k - off_a11;; \
				k_a11   = MR; \
				k_a12   = k_a1112 - MR; \
				off_a12 = off_a11 + k_a11; \
\
				/* Compute the addresses of the triangular block A11 and the
				   panel A12. */ \
				a11  = a1; \
				a12  = a1 + k_a11 * PACKMR; \
\
				/* Now compute the corresponding addresses in Bd. */ \
				bp11 = bp + off_a11 * NR * NDUP; \
				bp21 = bp + off_a12 * NR * NDUP; \
\
				/* Index into b1 to locate the MR x NR block of b1 that will be
				   updated by the trsm subproblem. */ \
				b11  = b1 + off_a11 * PACKNR; \
\
				/* Compute the addresses of the next panels of A and B. */ \
				a2 = a1 + k_a1112 * PACKMR; \
				if ( ib == m_iter - 1 ) \
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
					/* Invoke the fused gemm/trsm micro-kernel. */ \
					PASTEMAC(ch,gemmtrsmukr)( k_a12, \
					                          alpha_cast, \
					                          a12, \
					                          a11, \
					                          bp21, \
					                          bp11, \
					                          b11, \
					                          c11, rs_c, cs_c, \
					                          a2, b2 ); \
				} \
				else \
				{ \
					/* Invoke the fused gemm/trsm micro-kernel. */ \
					PASTEMAC(ch,gemmtrsmukr)( k_a12, \
					                          alpha_cast, \
					                          a12, \
					                          a11, \
					                          bp21, \
					                          bp11, \
					                          b11, \
					                          ct, rs_ct, cs_ct, \
					                          a2, b2 ); \
\
					/* Copy the result to the bottom edge of C. */ \
					PASTEMAC(ch,copys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        c11, rs_c,  cs_c ); \
				} \
\
				a1 += k_a1112 * PACKMR; \
			} \
			else if ( bli_is_strictly_above_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				/* Compute the addresses of the next panels of A and B. */ \
				a2 = a1 + rstep_a; \
				if ( ib == m_iter - 1 ) \
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
					                      a1, \
					                      bp, \
					                      alpha_cast, \
					                      c11, rs_c, cs_c, \
					                      a2, b2 ); \
				} \
				else \
				{ \
					/* Invoke the gemm micro-kernel. */ \
					PASTEMAC(ch,gemmukr)( k, \
					                      minus_one, \
					                      a1, \
					                      bp, \
					                      zero, \
					                      ct, rs_ct, cs_ct, \
					                      a2, b2 ); \
\
					/* Add the result to the edge of C. */ \
					PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
					                        ct,  rs_ct, cs_ct, \
					                        alpha_cast, \
					                        c11, rs_c,  cs_c ); \
				} \
\
				a1 += rstep_a; \
			} \
\
			c11 -= rstep_c; \
		} \
\
		b1 += cstep_b; \
		c1 += cstep_c; \
	} \
\
/*
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: a1 (diag)", MR, k_a1112, a1, 1, MR, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: b11 (diag)", MR, NR, b11, NR, 1, "%6.3f", "" ); \
printf( "m_iter     = %lu\n", m_iter ); \
printf( "m_cur      = %lu\n", m_cur ); \
printf( "k          = %lu\n", k ); \
printf( "diagoffa_i = %lu\n", diagoffa_i ); \
printf( "off_a1112  = %lu\n", off_a1112 ); \
printf( "k_a1112    = %lu\n", k_a1112 ); \
printf( "k_a12      = %lu\n", k_a12 ); \
printf( "k_a11      = %lu\n", k_a11 ); \
printf( "rs_c,cs_c  = %lu %lu\n", rs_c, cs_c ); \
printf( "rs_ct,cs_ct= %lu %lu\n", rs_ct, cs_ct ); \
*/ \
\
/*
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: bp11 after (diag)", MR, NR, bp11, NR, 1, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: b11 after (diag)", MR, NR, b11, NR, 1, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "trsm_lu_ker_var2: ct after (diag)", m_cur, n_cur, ct, rs_ct, cs_ct, "%5.2f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC2( trsm_lu_ker_var2, GEMMTRSM_U_UKERNEL, GEMM_UKERNEL )

