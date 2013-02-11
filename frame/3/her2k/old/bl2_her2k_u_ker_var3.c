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

#include "blis2.h"

#define FUNCPTR_T her2k_fp

typedef void (*FUNCPTR_T)(
                           doff_t  diagoffc,
                           uplo_t  uploc,
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   a,  inc_t ps_a,
                           void*   bh, inc_t ps_bh,
                           void*   b,  inc_t ps_b,
                           void*   ah, inc_t ps_ah,
                           void*   c,  inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T GENARRAY(ftypes,her2k_u_ker_var3);


void bl2_her2k_u_ker_var3( obj_t*   alpha,
                           obj_t*   a,
                           obj_t*   bh,
                           obj_t*   alpha_conj,
                           obj_t*   b,
                           obj_t*   ah,
                           obj_t*   beta,
                           obj_t*   c,
                           her2k_t* cntl )
{
	num_t     dt_exec   = bl2_obj_execution_datatype( *c );

	doff_t    diagoffc  = bl2_obj_diag_offset( *c );
	uplo_t    uploc     = bl2_obj_uplo( *c );

	dim_t     m         = bl2_obj_length( *c );
	dim_t     n         = bl2_obj_width( *c );
	dim_t     k         = bl2_obj_width( *a );

	void*     buf_a     = bl2_obj_buffer_at_off( *a );
	inc_t     ps_a      = bl2_obj_panel_stride( *a );

	void*     buf_bh    = bl2_obj_buffer_at_off( *bh );
	inc_t     ps_bh     = bl2_obj_panel_stride( *bh );

	void*     buf_b     = bl2_obj_buffer_at_off( *b );
	inc_t     ps_b      = bl2_obj_panel_stride( *b );

	void*     buf_ah    = bl2_obj_buffer_at_off( *ah );
	inc_t     ps_ah     = bl2_obj_panel_stride( *ah );

	void*     buf_c     = bl2_obj_buffer_at_off( *c );
	inc_t     rs_c      = bl2_obj_row_stride( *c );
	inc_t     cs_c      = bl2_obj_col_stride( *c );

	FUNCPTR_T f;


	// Handle the special case where c and a are complex and b is real.
	// Note that this is the ONLY case allowed by the inner kernel whereby
	// the datatypes of a and b differ. In this situation, the execution
	// datatype is real, so we need to inflate (by a factor of two):
	//  - the m dimension,
	//  - the column stride of c,
	//  - the column stride (ie: the panel length) of a, and
	//  - the panel stride of a.
	if ( bl2_obj_is_complex( *a ) && bl2_obj_is_real( *b ) )
	{
		m    *= 2;
		cs_c *= 2;
		//cs_a *= 2;
		ps_a *= 2;
	}

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( diagoffc,
	   uploc,
	   m,
	   n,
	   k,
	   buf_a,  ps_a,
	   buf_bh, ps_bh,
	   buf_b,  ps_b,
	   buf_ah, ps_ah,
	   buf_c,  rs_c,  cs_c );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname)( \
                           doff_t  diagoffc, \
                           uplo_t  uploc, \
                           dim_t   m, \
                           dim_t   n, \
                           dim_t   k, \
                           void*   a,  inc_t ps_a, \
                           void*   bh, inc_t ps_bh, \
                           void*   b,  inc_t ps_b, \
                           void*   ah, inc_t ps_ah, \
                           void*   c,  inc_t rs_c,  inc_t cs_c \
                         ) \
{ \
	/* Temporary b buffers for duplicating elements of bh, ah. */ \
	ctype        bd[ PASTEMAC2(ch,varname,_kc) * \
	                 PASTEMAC2(ch,varname,_nr) * \
	                 PASTEMAC2(ch,varname,_ndup) ]; \
	ctype        ad[ PASTEMAC2(ch,varname,_kc) * \
	                 PASTEMAC2(ch,varname,_nr) * \
	                 PASTEMAC2(ch,varname,_ndup) ]; \
\
	/* Temporary c buffer for edge cases. */ \
	ctype        ct[ PASTEMAC2(ch,varname,_mr) * PASTEMAC2(ch,varname,_nr) ]; \
	const inc_t  rs_ct = 1; \
	const inc_t  cs_ct = PASTEMAC2(ch,varname,_mr); \
\
	/* Alias the m and n register blocksizes to shorter names. */ \
	const dim_t  MR = PASTEMAC2(ch,varname,_mr); \
	const dim_t  NR = PASTEMAC2(ch,varname,_nr); \
\
	ctype*       a_cast  = a; \
	ctype*       bh_cast = bh; \
	ctype*       b_cast  = b; \
	ctype*       ah_cast = ah; \
	ctype*       c_cast  = c; \
	ctype*       a1; \
	ctype*       bh1; \
	ctype*       b1; \
	ctype*       ah1; \
	ctype*       c1; \
	ctype*       c11; \
	inc_t        rstep_a; \
	inc_t        cstep_b; \
	inc_t        rstep_c, cstep_c; \
	doff_t       diagoffc_ij; \
	dim_t        m_iter, m_left; \
	dim_t        n_iter, n_left; \
	dim_t        i, j; \
\
	if ( bl2_zero_dim3( m, n, k ) ) return; \
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
	/* Compute number of primary and leftover components of the m and n
	   dimensions. */ \
	n_iter = n / NR; \
	n_left = n % NR; \
\
	m_iter = m / MR; \
	m_left = m % MR; \
\
	rstep_a = ps_a; \
\
	cstep_b = ps_b; \
\
	rstep_c = MR * rs_c; \
	cstep_c = NR * cs_c; \
\
	bh1 = bh_cast; \
	ah1 = ah_cast; \
	c1  = c_cast; \
\
	for ( j = 0; j < n_iter; ++j ) \
	{ \
		a1  = a_cast; \
		b1  = b_cast; \
		c11 = c1; \
\
		/* Copy the current iteration's NR columns of B to a local buffer
		   with each value duplicated. */ \
		PASTEMAC2(ch,varname,_dupl)( k, bh1, bd ); \
		PASTEMAC2(ch,varname,_dupl)( k, ah1, ad ); \
\
		/* Interior loop. */ \
		for ( i = 0; i < m_iter; ++i ) \
		{ \
			/* Compute the diagonal offset for the submatrix at (i,j). */ \
			diagoffc_ij = diagoffc - (doff_t)j*NR + (doff_t)i*MR; \
\
			/* If the diagonal intersects the current MR x NR submatrix, we
			   compute in the temporary buffer and then add in the elements
			   on or below the diagonal.
			   Otherwise, if the submatrix is strictly above the diagonal,
			   we compute and store as we normally would.
			   And if we're strictly below the diagonal, we do nothing and
			   continue. */ \
			if ( bl2_intersects_diag_n( diagoffc_ij, MR, NR ) ) \
			{ \
				/* Zero the temporary C buffer. */ \
				PASTEMAC(ch,set0_mxn)( MR, NR, \
				                       ct, rs_ct, cs_ct ); \
\
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k, \
				                            a1, \
				                            bd, \
				                            ct, rs_ct, cs_ct ); \
				PASTEMAC2(ch,varname,_ukr)( k, \
				                            b1, \
				                            ad, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Add the result to only the stored part of C. */ \
				PASTEMAC2(ch,ch,adds_mxn_u)( diagoffc_ij, \
				                             MR, NR, \
				                             ct,  rs_ct, cs_ct, \
				                             c11, rs_c,  cs_c ); \
			} \
			else if ( bl2_is_strictly_above_diag_n( diagoffc_ij, MR, NR ) ) \
			{ \
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k, \
				                            a1, \
				                            bd, \
				                            c11, rs_c, cs_c ); \
				PASTEMAC2(ch,varname,_ukr)( k, \
				                            b1, \
				                            ad, \
				                            c11, rs_c, cs_c ); \
			} \
\
			a1  += rstep_a; \
			b1  += rstep_a; \
			c11 += rstep_c; \
		} \
\
		/* Bottom edge handling. This case never occurs since the bottom
		   edge is never reached as part of the interior loop. (It is only
		   updated as part of the bottom-right corner handling below.) */ \
		if ( m_left ) \
		{ \
			; \
		} \
\
		bh1 += cstep_b; \
		ah1 += cstep_b; \
		c1  += cstep_c; \
	} \
\
	if ( n_left ) \
	{ \
		a1  = a_cast; \
		b1  = b_cast; \
		c11 = c1; \
\
		/* Copy the n_left (+ padding) columns of B to a local buffer
		   with each value duplicated. */ \
		PASTEMAC2(ch,varname,_dupl)( k, bh1, bd ); \
		PASTEMAC2(ch,varname,_dupl)( k, ah1, ad ); \
\
		/* Right edge loop. (Note that the diagonal is guaranteed not
		   to factor in here.) */ \
		for ( i = 0; i < m_iter; ++i ) \
		{ \
			/* Zero the temporary C buffer. */ \
			PASTEMAC(ch,set0_mxn)( MR, n_left, \
			                       ct, rs_ct, cs_ct ); \
\
			/* Invoke the micro-kernel. */ \
			PASTEMAC2(ch,varname,_ukr)( k, \
			                            a1, \
			                            bd, \
			                            ct, rs_ct, cs_ct ); \
			PASTEMAC2(ch,varname,_ukr)( k, \
			                            b1, \
			                            ad, \
			                            ct, rs_ct, cs_ct ); \
\
			/* Add the result to the right edge of C. */ \
			PASTEMAC2(ch,ch,adds_mxn)( MR, n_left, \
			                           ct,  rs_ct, cs_ct, \
			                           c11, rs_c,  cs_c ); \
\
			a1  += rstep_a; \
			b1  += rstep_a; \
			c11 += rstep_c; \
		} \
\
		/* Compute the diagonal offset one last time. */ \
		diagoffc_ij = diagoffc - (doff_t)j*NR + (doff_t)i*MR; \
\
		/* Bottom-right corner handling. */ \
		if ( m_left ) \
		{ \
			/* Zero the temporary C buffer. */ \
			PASTEMAC(ch,set0_mxn)( m_left, n_left, \
			                       ct, rs_ct, cs_ct ); \
\
			/* Invoke the micro-kernel. */ \
			PASTEMAC2(ch,varname,_ukr)( k, \
			                            a1, \
			                            bd, \
			                            ct, rs_ct, cs_ct ); \
			PASTEMAC2(ch,varname,_ukr)( k, \
			                            b1, \
			                            ad, \
			                            ct, rs_ct, cs_ct ); \
\
			/* Add the result to only the stored part of C. */ \
			PASTEMAC2(ch,ch,adds_mxn_u)( diagoffc_ij, \
			                             m_left, n_left, \
			                             ct,  rs_ct, cs_ct, \
			                             c11, rs_c,  cs_c ); \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( her2k, her2k_u_ker_var3 )

