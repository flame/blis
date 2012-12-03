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
			} \
\
			a1  += rstep_a; \
			c11 += rstep_c; \
		} \
\
		/* Bottom edge handling. */ \
		if ( m_left ) \
		{ \
			/* Use the diagonal offset for the current panel of A to compute
			   k_use <= k so that we minimize the number of flops with zeros
			   (ie: when the current panel intersects the diagonal). */ \
			diagoffa_i = diagoffa + (doff_t)i*MR; \
			k_diag     = diagoffa_i + MR; \
			if      ( k_diag < 0 ) k_use = 0; \
			else if ( k_diag > k ) k_use = k; \
			else                   k_use = k_diag; \
\
			/* If the current panel intersects the diagonal, we need to
			   scale by beta. (When the the current function is invoked as
			   part of classic trmm, beta will be zero, and when invoked as
			   part of trmm3, beta will be non-zero). If the current panel
			   does not intersect the diagonal (but still has non-zero
			   elements), we accumulate into C (for both trmm and trmm3). */ \
			if ( bl2_intersects_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				/* Copy edge elements of C to the temporary buffer. */ \
				PASTEMAC2(ch,ch,copys_mxn)( m_left, NR, \
				                            c11, rs_c,  cs_c, \
				                            ct,  rs_ct, cs_ct ); \
\
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k_use, \
				                            a1, \
				                            bd, \
				                            beta, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Copy the result to the bottom edge of C. */ \
				PASTEMAC2(ch,ch,copys_mxn)( m_left, NR, \
				                            ct,  rs_ct, cs_ct, \
				                            c11, rs_c,  cs_c ); \
			} \
			else if ( k_use != 0 ) \
			{ \
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k_use, \
				                            a1, \
				                            bd, \
				                            zero, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Add the result to the bottom edge of C. */ \
				PASTEMAC2(ch,ch,adds_mxn)( m_left, NR, \
				                           ct,  rs_ct, cs_ct, \
				                           c11, rs_c,  cs_c ); \
			} \
		} \
\
		b1 += cstep_b; \
		c1 += cstep_c; \
	} \
\
	if ( n_left ) \
	{ \
		a1  = a_cast; \
		c11 = c1; \
\
		/* Copy the n_left (+ padding) columns of B to a local buffer
		   with each value duplicated. */ \
		PASTEMAC2(ch,varname,_dupl)( k, b1, bd ); \
\
		/* Right edge loop. */ \
		for ( i = 0; i < m_iter; ++i ) \
		{ \
			/* Use the diagonal offset for the current panel of A to compute
			   k_use <= k so that we minimize the number of flops with zeros
			   (ie: when the current panel intersects the diagonal). */ \
			diagoffa_i = diagoffa + (doff_t)i*MR; \
			k_diag     = diagoffa_i + MR; \
			if      ( k_diag < 0 ) k_use = 0; \
			else if ( k_diag > k ) k_use = k; \
			else                   k_use = k_diag; \
\
			/* If the current panel intersects the diagonal, we need to
			   scale by beta. (When the the current function is invoked as
			   part of classic trmm, beta will be zero, and when invoked as
			   part of trmm3, beta will be non-zero). If the current panel
			   does not intersect the diagonal (but still has non-zero
			   elements), we accumulate into C (for both trmm and trmm3). */ \
			if ( bl2_intersects_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				/* Copy edge elements of C to the temporary buffer. */ \
				PASTEMAC2(ch,ch,copys_mxn)( MR, n_left, \
				                            c11, rs_c,  cs_c, \
				                            ct,  rs_ct, cs_ct ); \
\
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k_use, \
				                            a1, \
				                            bd, \
				                            beta, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Copy the result to the right edge of C. */ \
				PASTEMAC2(ch,ch,copys_mxn)( MR, n_left, \
				                            ct,  rs_ct, cs_ct, \
				                            c11, rs_c,  cs_c ); \
			} \
			else if ( k_use != 0 ) \
			{ \
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k_use, \
				                            a1, \
				                            bd, \
				                            zero, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Add the result to the right edge of C. */ \
				PASTEMAC2(ch,ch,adds_mxn)( MR, n_left, \
				                           ct,  rs_ct, cs_ct, \
				                           c11, rs_c,  cs_c ); \
			} \
\
			a1  += rstep_a; \
			c11 += rstep_c; \
		} \
\
		/* Bottom-right corner handling. */ \
		if ( m_left ) \
		{ \
			/* Use the diagonal offset for the current panel of A to compute
			   k_use <= k so that we minimize the number of flops with zeros
			   (ie: when the current panel intersects the diagonal). */ \
			diagoffa_i = diagoffa + (doff_t)i*MR; \
			k_diag     = diagoffa_i + MR; \
			if      ( k_diag < 0 ) k_use = 0; \
			else if ( k_diag > k ) k_use = k; \
			else                   k_use = k_diag; \
\
			/* If the current panel intersects the diagonal, we need to
			   scale by beta. (When the the current function is invoked as
			   part of classic trmm, beta will be zero, and when invoked as
			   part of trmm3, beta will be non-zero). If the current panel
			   does not intersect the diagonal (but still has non-zero
			   elements), we accumulate into C (for both trmm and trmm3). */ \
			if ( bl2_intersects_diag_n( diagoffa_i, MR, k ) ) \
			{ \
				/* Copy edge elements of C to the temporary buffer. */ \
				PASTEMAC2(ch,ch,copys_mxn)( m_left, n_left, \
				                            c11, rs_c,  cs_c, \
				                            ct,  rs_ct, cs_ct ); \
\
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k_use, \
				                            a1, \
				                            bd, \
				                            beta, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Copy the result to the bottom-right corner of C. */ \
				PASTEMAC2(ch,ch,copys_mxn)( m_left, n_left, \
				                            ct,  rs_ct, cs_ct, \
				                            c11, rs_c,  cs_c ); \
			} \
			else if ( k_use != 0 ) \
			{ \
				/* Invoke the micro-kernel. */ \
				PASTEMAC2(ch,varname,_ukr)( k_use, \
				                            a1, \
				                            bd, \
				                            zero, \
				                            ct, rs_ct, cs_ct ); \
\
				/* Add the result to the bottom-right corner of C. */ \
				PASTEMAC2(ch,ch,adds_mxn)( m_left, n_left, \
				                           ct,  rs_ct, cs_ct, \
				                           c11, rs_c,  cs_c ); \
			} \
		} \
	} \
}

INSERT_GENTFUNC_BASIC( trmm, trmm_l_ker_var2 )

