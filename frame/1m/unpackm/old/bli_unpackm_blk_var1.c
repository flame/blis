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

#define FUNCPTR_T unpackm_fp

typedef void (*FUNCPTR_T)(
                           struc_t strucc,
                           doff_t  diagoffc,
                           diag_t  diagc,
                           uplo_t  uploc,
                           trans_t transc,
                           dim_t   m,
                           dim_t   n,
                           void*   p, inc_t rs_p, inc_t cs_p, inc_t ps_p,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T GENARRAY(ftypes,unpackm_blk_var1);


void bli_unpackm_blk_var1( obj_t*     p,
                           obj_t*     c,
                           unpackm_t* cntl )
{
	num_t     dt_cp     = bli_obj_datatype( *c );

	// Normally we take the parameters from the source argument. But here,
	// the packm/unpackm framework is not yet solidified enough for us to
	// assume that at this point struc(P) == struc(C), (ie: since
	// densification may have marked P's structure as dense when the root
	// is upper or lower). So, we take the struc field from C, not P.
	struc_t   strucc    = bli_obj_struc( *c );
	doff_t    diagoffc  = bli_obj_diag_offset( *c );
	diag_t    diagc     = bli_obj_diag( *c );
	uplo_t    uploc     = bli_obj_uplo( *c );

	// Again, normally the trans argument is on the source matrix. But we
	// know that the packed matrix is not transposed. If there is to be a
	// transposition, it is because C was originally transposed when packed.
	// Thus, we query C for the trans status, not P. Also, we only query
	// the trans status (not the conjugation status), since we probably
	// don't want to un-conjugate if the original matrix was conjugated
	// when packed.
	trans_t   transc    = bli_obj_onlytrans_status( *c );

	dim_t     m_c       = bli_obj_length( *c );
	dim_t     n_c       = bli_obj_width( *c );

	void*     buf_p     = bli_obj_buffer_at_off( *p );
	inc_t     rs_p      = bli_obj_row_stride( *p );
	inc_t     cs_p      = bli_obj_col_stride( *p );
	inc_t     ps_p      = bli_obj_panel_stride( *p );

	void*     buf_c     = bli_obj_buffer_at_off( *c );
	inc_t     rs_c      = bli_obj_row_stride( *c );
	inc_t     cs_c      = bli_obj_col_stride( *c );

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
	   m_c,
	   n_c,
	   buf_p, rs_p, cs_p, ps_p,
	   buf_c, rs_c, cs_c );
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
                            void*   p, inc_t rs_p, inc_t cs_p, inc_t ps_p, \
                            void*   c, inc_t rs_c, inc_t cs_c  \
                          ) \
{ \
	ctype* one       = PASTEMAC(ch,1); \
	ctype* c_cast    = c; \
	ctype* p_cast    = p; \
	ctype* c_begin; \
	ctype* p_begin; \
	dim_t  panel_dim; \
	/*dim_t  panel_len;*/ \
	dim_t  iter_dim; \
	doff_t diagoffc_i; \
	dim_t  panel_dim_i; \
	dim_t  ic, ip; \
	inc_t  diagoffc_inc, vs_c; \
	/*inc_t  incc, ldc;*/ \
	dim_t* m_panel; \
	dim_t* n_panel; \
\
	/* If c needs a transposition, induce it so that we can more simply
	   express the remaining parameters and code. */ \
	if ( bli_does_trans( transc ) ) \
	{ \
		bli_swap_incs( rs_c, cs_c ); \
		bli_negate_diag_offset( diagoffc ); \
		bli_toggle_uplo( uploc ); \
		bli_toggle_trans( transc ); \
	} \
\
	/* If the strides of p indicate row storage, then we are packing to
	   column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( bli_is_row_stored_f( rs_p, cs_p ) ) \
	{ \
		/* Prepare to unpack from column panels. */ \
		iter_dim     = n; \
		/*panel_len    = m;*/ \
		panel_dim    = rs_p; \
		/*incc         = cs_c;*/ \
		/*ldc          = rs_c;*/ \
		vs_c         = cs_c; \
		diagoffc_inc = -( doff_t)panel_dim; \
		m_panel      = &m; \
		n_panel      = &panel_dim_i; \
	} \
	else /* if ( bli_is_col_stored_f( rs_p, cs_p ) ) */ \
	{ \
		/* Prepare to unpack from row panels. */ \
		iter_dim     = m; \
		/*panel_len    = n;*/ \
		panel_dim    = cs_p; \
		/*incc         = rs_c;*/ \
		/*ldc          = cs_c;*/ \
		vs_c         = rs_c; \
		diagoffc_inc = ( doff_t )panel_dim; \
		m_panel      = &panel_dim_i; \
		n_panel      = &n; \
	} \
\
\
	for ( ic  = 0,         ip  = 0, diagoffc_i  = diagoffc;  ic < iter_dim; \
	      ic += panel_dim, ip += 1, diagoffc_i += diagoffc_inc ) \
\
	{ \
		panel_dim_i = bli_min( panel_dim, iter_dim - ic ); \
\
		p_begin = p_cast + ip * ps_p; \
		c_begin = c_cast + ic * vs_c; \
\
		/* If the current panel of C intersects the diagonal AND is upper or
		   lower stored, then we must call scal2m. Otherwise, we can use a
		   variant that is oblivious to structure and storage (and thus tends
		   to be faster). */ \
		PASTEMAC3(ch,ch,ch,scal2m)( diagoffc_i, \
		                            diagc, \
		                            uploc, \
		                            transc, \
		                            *m_panel, \
		                            *n_panel, \
		                            one, \
		                            p_begin, rs_p, cs_p, \
		                            c_begin, rs_c, cs_c ); \
\
		/*PASTEMAC(ch,fprintm)( stdout, "p copied", *m_panel, *n_panel, \
		                      p_begin, rs_p, cs_p, "%4.1f", "" );*/ \
	} \
\
}

INSERT_GENTFUNC_BASIC( unpackm, unpackm_blk_var1 )

