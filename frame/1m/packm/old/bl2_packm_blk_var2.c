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
\
/*
			if ( rs_p == 1 ) { \
			printf( "packm_blk_var2: ps_p = %u\n", ps_p ); \
			PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: p copied", m_panel_max, n_panel_max, \
			                      p_begin, rs_p, cs_p, "%4.1f", "" ); \
			} \
*/ \
		} \
\
		if ( rs_p == 1 ) \
		PASTEMAC(ch,fprintm)( stdout, "packm_blk_var2: c copied", m_panel_max, n_panel_max, \
		                      p_begin, 1, panel_dim, "%4.1f", "" ); \
	} \
\
}

INSERT_GENTFUNC_BASIC( packm, packm_blk_var2 )

