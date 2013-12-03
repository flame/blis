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
#include <omp.h>

#define FUNCPTR_T packm_fp

typedef void (*FUNCPTR_T)(
                           struc_t strucc,
                           doff_t  diagoffc,
                           diag_t  diagc,
                           uplo_t  uploc,
                           trans_t transc,
                           dim_t   m,
                           dim_t   n,
                           dim_t   m_max,
                           dim_t   n_max,
                           void*   beta,
                           void*   c, inc_t rs_c, inc_t cs_c,
                           void*   p, inc_t rs_p, inc_t cs_p,
                                      dim_t pd_p, inc_t ps_p
                         );

static FUNCPTR_T GENARRAY(ftypes,packm_blk_var2);


void bli_packm_blk_var2( obj_t*   beta,
                         obj_t*   c,
                         obj_t*   p )
{
	num_t     dt_cp     = bli_obj_datatype( *c );

	struc_t   strucc    = bli_obj_struc( *c );
	doff_t    diagoffc  = bli_obj_diag_offset( *c );
	diag_t    diagc     = bli_obj_diag( *c );
	uplo_t    uploc     = bli_obj_uplo( *c );
	trans_t   transc    = bli_obj_conjtrans_status( *c );

	dim_t     m_p       = bli_obj_length( *p );
	dim_t     n_p       = bli_obj_width( *p );
	dim_t     m_max_p   = bli_obj_padded_length( *p );
	dim_t     n_max_p   = bli_obj_padded_width( *p );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

	void*     buf_p     = bli_obj_buffer_at_off( *p );
	inc_t     rs_p      = bli_obj_row_stride( *p );
	inc_t     cs_p      = bli_obj_col_stride( *p );
	dim_t     pd_p      = bli_obj_panel_dim( *p );
	inc_t     ps_p      = bli_obj_panel_stride( *p );

	void*     buf_beta  = bli_obj_buffer_for_1x1( dt_cp, *beta );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_cp];

	// Invoke the function.
	f( strucc,
	   diagoffc,
	   diagc,
	   uploc,
	   transc,
	   m_p,
	   n_p,
	   m_max_p,
	   n_max_p,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   buf_p, rs_p, cs_p,
	          pd_p, ps_p );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, varname ) \
\
void PASTEMAC(ch,varname )( \
                            struc_t strucc, \
                            doff_t  diagoffc, \
                            diag_t  diagc, \
                            uplo_t  uploc, \
                            trans_t transc, \
                            dim_t   m, \
                            dim_t   n, \
                            dim_t   m_max, \
                            dim_t   n_max, \
                            void*   beta, \
                            void*   c, inc_t rs_c, inc_t cs_c, \
                            void*   p, inc_t rs_p, inc_t cs_p, \
                                       dim_t pd_p, inc_t ps_p  \
                          ) \
{ \
	/* If C needs a transposition, induce it so that we can more simply
	   express the remaining parameters and code. */ \
	if ( bli_does_trans( transc ) ) \
	{ \
		bli_swap_incs( rs_c, cs_c ); \
		bli_negate_diag_offset( diagoffc ); \
		bli_toggle_uplo( uploc ); \
		bli_toggle_trans( transc ); \
	} \
\
	_Pragma( "omp parallel" ) \
	{ \
	guint_t         n_threads = omp_get_num_threads(); \
	guint_t         t_id      = omp_get_thread_num(); \
\
	ctype* restrict beta_cast = beta; \
	ctype* restrict c_cast    = c; \
	ctype* restrict p_cast    = p; \
	ctype* restrict zero      = PASTEMAC(ch,0); \
	ctype* restrict c_begin; \
	ctype* restrict p_begin; \
\
	dim_t           iter_dim; \
	dim_t           num_iter; \
	dim_t           it, ic, ip; \
	dim_t           j; \
	dim_t           ic0, ip0; \
	dim_t           ic_inc, ip_inc; \
	dim_t           panel_dim; \
	dim_t           panel_len; \
	doff_t          diagoffc_i; \
	doff_t          diagoffc_inc; \
	doff_t          diagoffc_i_abs; \
	dim_t           panel_dim_i; \
	inc_t           vs_c; \
	inc_t           incc, ldc; \
	inc_t           ldp; \
	dim_t*          m_panel; \
	dim_t*          n_panel; \
	dim_t           m_panel_max; \
	dim_t           n_panel_max; \
	conj_t          conjc; \
\
	ctype* restrict c10; \
	ctype* restrict p10; \
	dim_t           p10_dim, p10_len; \
	inc_t           incc10, ldc10; \
	doff_t          diagoffc10; \
	conj_t          conjc10; \
\
	ctype* restrict c12; \
	ctype* restrict p12; \
	dim_t           p12_dim, p12_len; \
	inc_t           incc12, ldc12; \
	doff_t          diagoffc12; \
	conj_t          conjc12; \
\
	ctype* restrict c11; \
	ctype* restrict p11; \
	dim_t           p11_m; \
	dim_t           p11_n; \
	inc_t           rs_p11, cs_p11; \
\
\
	/* Extract the conjugation bit from the transposition argument. */ \
	conjc = bli_extract_conj( transc ); \
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored( rs_p, cs_p ) ) \
	{ \
		/* Prepare to pack to row-stored column panels. */ \
		iter_dim     = n; \
		panel_len    = m; \
		panel_dim    = pd_p; \
		incc         = cs_c; \
		ldc          = rs_c; \
		vs_c         = cs_c; \
		diagoffc_inc = -( doff_t)panel_dim; \
		ldp          = rs_p; \
		m_panel      = &m; \
		n_panel      = &panel_dim_i; \
		m_panel_max  = m_max; \
		n_panel_max  = panel_dim; \
		rs_p11       = rs_p; \
		cs_p11       = 1; \
	} \
	else /* if ( bli_is_col_stored( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to pack to column-stored row panels. */ \
		iter_dim     = m; \
		panel_len    = n; \
		panel_dim    = pd_p; \
		incc         = rs_c; \
		ldc          = cs_c; \
		vs_c         = rs_c; \
		diagoffc_inc = ( doff_t )panel_dim; \
		ldp          = cs_p; \
		m_panel      = &panel_dim_i; \
		n_panel      = &n; \
		m_panel_max  = panel_dim; \
		n_panel_max  = n_max; \
		rs_p11       = 1; \
		cs_p11       = cs_p; \
	} \
\
	/* Compute the total number of iterations we'll need. */ \
	num_iter = iter_dim / panel_dim + ( iter_dim % panel_dim ? 1 : 0 ); \
\
	/* Set the initial values and increments for indices related to C and P.
	   Currently we only support forwards iteration. */ \
	{ \
		ic0    = 0; \
		ic_inc = panel_dim; \
		ip0    = 0; \
		ip_inc = 1; \
	} \
\
	for ( ic  = ic0 + t_id*ic_inc, ip = ip0 + t_id*ip_inc, it = t_id; it < num_iter; \
	      ic += ic_inc*n_threads, ip += ip_inc*n_threads, it += n_threads ) \
	{ \
		panel_dim_i    = bli_min( panel_dim, iter_dim - ic ); \
\
		diagoffc_i     = diagoffc + (ip  )*diagoffc_inc; \
		c_begin        = c_cast   + (ic  )*vs_c; \
		p_begin        = p_cast   + (ip  )*ps_p; \
\
		/* If the current panel intersects the diagonal and C is either
		   upper- or lower-stored, then we assume C is symmetric or
		   Hermitian and that it must be densified (note we don't even
		   bother passing in a densify parameter), in which case we pack
		   the panel in three stages. 
		   Otherwise, we pack the panel all at once. */ \
		if ( bli_intersects_diag_n( diagoffc_i, *m_panel, *n_panel ) && \
		     bli_is_upper_or_lower( uploc ) ) \
		{ \
			diagoffc_i_abs = bli_abs( diagoffc_i ); \
\
			if ( ( bli_is_col_stored( rs_p, cs_p ) && diagoffc_i < 0 ) || \
			     ( bli_is_row_stored( rs_p, cs_p ) && diagoffc_i > 0 ) ) \
				bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
\
			if      ( ( bli_is_row_stored( rs_p, cs_p ) && bli_is_upper( uploc ) ) || \
			          ( bli_is_col_stored( rs_p, cs_p ) && bli_is_lower( uploc ) ) ) \
			{ \
				p10_dim    = panel_dim_i; \
				p10_len    = diagoffc_i_abs; \
				p10        = p_begin; \
				c10        = c_begin; \
				incc10     = incc; \
				ldc10      = ldc; \
				conjc10    = conjc; \
\
				p12_dim    = panel_dim_i; \
				p12_len    = panel_len - p10_len; \
				j          = p10_len; \
				diagoffc12 = diagoffc_i_abs - j; \
				p12        = p_begin + (j  )*ldp; \
				c12        = c_begin + (j  )*ldc; \
				c12        = c12 + diagoffc12 * ( doff_t )cs_c + \
				                  -diagoffc12 * ( doff_t )rs_c;  \
				incc12     = ldc; \
				ldc12      = incc; \
				conjc12    = conjc; \
\
				p11_m      = panel_dim_i; \
				p11_n      = panel_dim_i; \
				j          = diagoffc_i_abs; \
				p11        = p_begin + (j  )*ldp; \
				c11        = c_begin + (j  )*ldc; \
\
				if ( bli_is_hermitian( strucc ) ) \
					bli_toggle_conj( conjc12 ); \
			} \
			else /* if ( ( bli_is_row_stored( rs_p, cs_p ) && bli_is_lower( uploc ) ) || \
			             ( bli_is_col_stored( rs_p, cs_p ) && bli_is_upper( uploc ) ) ) */ \
			{ \
				p10_dim    = panel_dim_i; \
				p10_len    = diagoffc_i_abs + panel_dim_i; \
				diagoffc10 = diagoffc_i; \
				p10        = p_begin; \
				c10        = c_begin; \
				c10        = c10 + diagoffc10 * ( doff_t )cs_c + \
				                  -diagoffc10 * ( doff_t )rs_c;  \
				incc10     = ldc; \
				ldc10      = incc; \
				conjc10    = conjc; \
\
				p12_dim    = panel_dim_i; \
				p12_len    = panel_len - p10_len; \
				j          = p10_len; \
				p12        = p_begin + (j  )*ldp; \
				c12        = c_begin + (j  )*ldc; \
				incc12     = incc; \
				ldc12      = ldc; \
				conjc12    = conjc; \
\
				p11_m      = panel_dim_i; \
				p11_n      = panel_dim_i; \
				j          = diagoffc_i_abs; \
				p11        = p_begin + (j  )*ldp; \
				c11        = c_begin + (j  )*ldc; \
\
				if ( bli_is_hermitian( strucc ) ) \
					bli_toggle_conj( conjc10 ); \
			} \
\
			/* Pack to P10. For upper storage, this includes the unstored
			   triangle of C11. */ \
			PASTEMAC(ch,packm_cxk)( conjc10, \
			                        p10_dim, \
			                        p10_len, \
			                        beta_cast, \
			                        c10, incc10, ldc10, \
			                        p10,         ldp ); \
\
			/* Pack to P12. For lower storage, this includes the unstored
			   triangle of C11. */ \
			PASTEMAC(ch,packm_cxk)( conjc12, \
			                        p12_dim, \
			                        p12_len, \
			                        beta_cast, \
			                        c12, incc12, ldc12, \
			                        p12,         ldp ); \
\
			/* Pack the stored triangule of C11 to P11. */ \
			PASTEMAC3(ch,ch,ch,scal2m)( 0, \
			                            BLIS_NONUNIT_DIAG, \
			                            uploc, \
			                            conjc, \
			                            p11_m, \
			                            p11_n, \
			                            beta_cast, \
			                            c11, rs_c,   cs_c, \
			                            p11, rs_p11, cs_p11 ); \
		} \
		else \
		{ \
			/* Note that the following code executes if the current panel either:
			   - does not intersect the diagonal, or
			   - does intersect the diagonal, BUT the matrix is general
			   which means the entire current panel can be copied at once. */ \
\
			/* We use some c10-specific variables here because we might need
			   to change them if the current panel is unstored. (The values
			   below are used if the current panel is stored.) */ \
			c10     = c_begin; \
			incc10  = incc; \
			ldc10   = ldc; \
			conjc10 = conjc; \
\
			/* If the current panel is unstored, we need to make a few
			   adjustments so we refer to the data where it is actually
			   stored, and so we take conjugation into account. (Note
			   this implicitly assumes we are operating on a symmetric or
			   Hermitian matrix, since a general matrix would not contain
			   any unstored region.) */ \
			if ( bli_is_unstored_subpart_n( diagoffc_i, uploc, *m_panel, *n_panel ) ) \
			{ \
				c10 = c10 + diagoffc_i * ( doff_t )cs_c + \
				           -diagoffc_i * ( doff_t )rs_c;  \
				bli_swap_incs( incc10, ldc10 ); \
\
				if ( bli_is_hermitian( strucc ) ) \
					bli_toggle_conj( conjc10 ); \
			} \
\
			/* Pack the current panel. */ \
			PASTEMAC(ch,packm_cxk)( conjc10, \
			                        panel_dim_i, \
			                        panel_len, \
			                        beta_cast, \
			                        c10,     incc10, ldc10, \
			                        p_begin,         ldp ); \
\
/*
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: c", panel_len, panel_dim_i, \
		                      c_begin, ldc, incc, "%5.2f", "" ); \
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: p copied", panel_len, panel_dim_i, \
		                      p_begin, ldp, 1, "%5.2f", "" ); \
*/ \
		} \
\
		/* The packed memory region was acquired/allocated with "aligned"
		   dimensions (ie: dimensions that were possibly inflated up to a
		   multiple). When these dimension are inflated, it creates empty
		   regions along the bottom and/or right edges of the matrix. If
		   either region exists, we set them to zero. This simplifies the
		   register level micro-kernel in that it does not need to support
		   different register blockings for the edge cases. */ \
		if ( *m_panel != m_panel_max ) \
		{ \
			dim_t  i      = *m_panel; \
			dim_t  m_edge = m_panel_max - i; \
			dim_t  n_edge = n_panel_max; \
			ctype* p_edge = p_begin + (i  )*rs_p; \
\
			PASTEMAC2(ch,ch,setm)( 0, \
			                       BLIS_NONUNIT_DIAG, \
			                       BLIS_DENSE, \
			                       m_edge, \
			                       n_edge, \
			                       zero, \
			                       p_edge, rs_p, cs_p ); \
		} \
\
		if ( *n_panel != n_panel_max ) \
		{ \
			dim_t  j      = *n_panel; \
			dim_t  m_edge = m_panel_max; \
			dim_t  n_edge = n_panel_max - j; \
			ctype* p_edge = p_begin + (j  )*cs_p; \
\
			PASTEMAC2(ch,ch,setm)( 0, \
			                       BLIS_NONUNIT_DIAG, \
			                       BLIS_DENSE, \
			                       m_edge, \
			                       n_edge, \
			                       zero, \
			                       p_edge, rs_p, cs_p ); \
		} \
	} \
\
	} /* end omp parallel */ \
\
/*
		if ( rs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: a copied", m_panel_max, n_panel_max, \
		                      p_begin, 1, cs_p, "%4.1f", "" ); \
		if ( cs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: b copied", m_panel_max, n_panel_max, \
		                      p_begin, panel_dim, 1, "%8.5f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC( packm, packm_blk_var2 )

