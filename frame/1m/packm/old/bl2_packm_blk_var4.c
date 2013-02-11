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
			                             BLIS_NO_TRANSPOSE, \
			                             FALSE, \
			                             m_a10, \
			                             n_a10, \
			                             m_max_a10, \
			                             n_max_a10, \
			                             beta, \
			                             c_begin, rs_c, cs_c, \
			                             p_begin, rs_p, cs_p, ps_pt ); \
\
			p_cast += m_max_a10 * n_max_a10; \
		} \
\
		/* Pack triangle subpartition A11. */ \
		{ \
			j       = n_a10; \
			c_begin = c_cast + (0  )*rs_c + (j  )*cs_c; \
			p_begin = p_cast; \
\
			/* This instance of ps_pt is not used by var3. */ \
			ps_pt   = cs_p * n_max_a11; \
\
			PASTEMAC(ch,packm_blk_var3)( BLIS_TRIANGULAR, \
			                             0, \
			                             diagc, \
			                             uploc, \
			                             BLIS_NO_TRANSPOSE, \
			                             densify, \
			                             invdiag, \
			                             m_a11, \
			                             n_a11, \
			                             m_max_a11, \
			                             n_max_a11, \
			                             beta, \
			                             c_begin, rs_c, cs_c, \
			                             p_begin, rs_p, cs_p, ps_p ); \
\
			p_cast += m_max_a11 * n_max_a11; \
		} \
\
		/* If they exist, pack subpartitions A20 and A21. */ \
		if ( m_a2021 ) \
		{ \
			i       = m_a10; \
			c_begin = c_cast + (i  )*rs_c + (0  )*cs_c; \
			p_begin = p_cast; \
\
			ps_pt   = cs_p * n_max_a2021; \
\
			PASTEMAC(ch,packm_blk_var2)( BLIS_GENERAL, \
			                             0, \
			                             BLIS_NONUNIT_DIAG, \
			                             BLIS_DENSE, \
			                             BLIS_NO_TRANSPOSE, \
			                             FALSE, \
			                             m_a2021, \
			                             n_a2021, \
			                             m_max_a2021, \
			                             n_max_a2021, \
			                             beta, \
			                             c_begin, rs_c, cs_c, \
			                             p_begin, rs_p, cs_p, ps_pt ); \
		} \
\
\
	} \
	else \
	{ \
		bl2_abort(); \
	} \
\
}

INSERT_GENTFUNC_BASIC( packm, packm_blk_var4 )

