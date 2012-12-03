/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2012, The University of Texas

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
\
					if ( bl2_is_lower( uploc ) ) panel_off_i = 0; \
					else                         panel_off_i = bl2_max( 0, diagoffc_i ); \
				} \
			} \
			else \
			{ \
				panel_len_i = panel_len; \
				panel_off_i = 0; \
			} \
\
\
			c_use = c_begin + panel_off_i*ldc; \
			p_use = p_begin + panel_off_i*panel_dim; \
\
			PASTEMAC(ch,packm_cxk)( conjc, \
			                        panel_dim_i, \
			                        panel_len_i, \
			                        beta_cast, \
			                        c_use, incc, ldc, \
			                        p_use,       panel_dim ); \
\
/*
			if ( bl2_is_unit_diag( diagc ) ) \
			{ \
				PASTEMAC2(ch,ch,setd)( diagoffc_i, \
				                       *m_panel, \
				                       *n_panel, \
				                       beta_cast, \
				                       p_begin, rs_p, cs_p ); \
			} \
\
			if ( bl2_intersects_diag_n( diagoffc_i, *m_panel, *n_panel ) && \
			     bl2_is_upper_or_lower( uploc ) && \
			     densify == TRUE ) \
			{ \
				PASTEMAC(ch,packm_densify)( strucc, \
				                            diagoffc_i, \
				                            uploc, \
				                            transc, \
				                            *m_panel, \
				                            *n_panel, \
				                            beta_cast, \
				                            c_begin, rs_c, cs_c, \
				                            p_begin, rs_p, cs_p ); \
			} \
*/ \
\
\
/*
			PASTEMAC(ch,packm_cxk)( conjc, \
			                        panel_dim_i, \
			                        panel_len, \
			                        beta_cast, \
			                        c_begin, incc, ldc, \
			                        p_begin,       panel_dim ); \
*/ \
\
			/* The packed memory region was acquired/allocated with "aligned"
			   dimensions (ie: dimensions that were possibly inflated up to a
			   multiple). When these dimension are inflated, it creates empty
			   regions along the bottom and/or right edges of the matrix. If
			   either region exists, we set them to zero. This simplifies the
			   register level micro kernel in that it does not need to support
			   different register blockings for the edge cases. */ \
			if ( *m_panel != m_panel_max ) \
			{ \
				dim_t  m_edge = m_panel_max - *m_panel; \
				dim_t  n_edge = n_panel_max; \
				ctype* p_edge = p_begin + (*m_panel  )*rs_p; \
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
				dim_t  m_edge = m_panel_max; \
				dim_t  n_edge = n_panel_max - *n_panel; \
				ctype* p_edge = p_begin + (*n_panel  )*cs_p; \
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
	} \
\
}

INSERT_GENTFUNC_BASIC( packm, packm_blk_var3 )

