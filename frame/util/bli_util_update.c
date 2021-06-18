/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2020 - 21, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"






#undef GENTFUNC
#define GENTFUNC( ctype, ch, varname) \
\
void PASTEMAC(ch, varname) \
	( \
	  dim_t m_off, \
	  dim_t n_off, \
	  dim_t m_cur, \
	  dim_t n_cur, \
	  ctype* ct, inc_t rs_ct, inc_t cs_ct, \
	  ctype* beta_cast, \
	  ctype* c, inc_t rs_c, inc_t cs_c \
	) \
{ \
	dim_t start, end; \
	dim_t m, n, diag; \
\
	ctype beta_val = *beta_cast; \
\
	start = ((n_off < m_off) && (m_off < n_off + n_cur)) ? m_off: n_off; \
	end   = ((n_off < m_off + m_cur) && (m_off + m_cur < n_off + n_cur))? (m_off + m_cur):(n_off + n_cur); \
\
	if ( beta_val == 1.0 ) \
	{ \
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = 0; n <= diag-n_off; n++) \
				c[m*rs_c + n] += ct[m*rs_ct + n]; \
\
		for(; m < m_cur; m++) \
			for(n = 0; n < n_cur; n++) \
				c[m*rs_c + n] += ct[m*rs_ct + n]; \
	} \
	else if( beta_val == 0.0 )\
	{ \
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = 0; n <= diag-n_off; n++) \
				c[m*rs_c + n] = ct[m*rs_ct + n]; \
\
		for(; m < m_cur; m++) \
			for(n = 0; n < n_cur; n++) \
				c[m*rs_c + n] = ct[m*rs_ct + n]; \
	} \
	else \
	{ \
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = 0; n <= diag-n_off; n++) \
				c[m*rs_c + n] = c[m * rs_c + n] * beta_val + ct[m*rs_ct + n]; \
\
		for(; m < m_cur; m++) \
			for(n = 0; n < n_cur; n++) \
				c[m*rs_c + n] = c[m * rs_c + n] * beta_val + ct[m*rs_ct + n]; \
	} \
\
	return; \
}

INSERT_GENTFUNC_BASIC0_SD( update_lower_triang )

#undef GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch, varname) \
	( \
	  dim_t m_off, \
	  dim_t n_off, \
	  dim_t m_cur, \
	  dim_t n_cur, \
	  ctype* ct, inc_t rs_ct, inc_t cs_ct, \
	  ctype* beta_cast, \
	  ctype* c, inc_t rs_c, inc_t cs_c \
	) \
{ \
	dim_t start, end; \
	dim_t m, n, diag; \
\
	ctype beta_val = *beta_cast; \
\
	start = ((n_off < m_off) && (m_off < n_off + n_cur)) ? m_off: n_off; \
	end   = ((n_off < m_off + m_cur) && (m_off + m_cur < n_off + n_cur))? (m_off + m_cur):(n_off + n_cur); \
\
	if( beta_val == 1.0 ) \
	{ \
		for(m = 0; m < start-m_off; m++) \
			for(n = 0; n < n_cur; n++) \
				c[m*rs_c + n] += ct[m*rs_ct + n]; \
\
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = diag-n_off; n < n_cur; n++) \
				c[m*rs_c + n] += ct[m*rs_ct + n]; \
	} \
	else if ( beta_val == 0.0 )\
	{ \
		for(m = 0; m < start-m_off; m++) \
			for(n = 0; n < n_cur; n++) \
				c[m*rs_c + n] = ct[m*rs_ct + n]; \
\
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = diag-n_off; n < n_cur; n++) \
				c[m*rs_c + n] = ct[m*rs_ct + n]; \
	} \
	else \
	{ \
		for(m = 0; m < start-m_off; m++) \
			for(n = 0; n < n_cur; n++) \
				c[m*rs_c + n] = c[m * rs_c + n] * beta_val + ct[m*rs_ct + n]; \
\
	for(diag = start, m= start-m_off; diag < end; diag++, m++) \
		for(n = diag-n_off; n < n_cur; n++) \
			c[m*rs_c + n] = c[m * rs_c + n] * beta_val + ct[m*rs_ct + n]; \
	} \
\
	return; \
}
INSERT_GENTFUNC_BASIC0_SD( update_upper_triang )

#undef GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch, varname) \
	( \
	  dim_t m_off, \
	  dim_t n_off, \
	  dim_t m_cur, \
	  dim_t n_cur, \
	  ctype* ct, inc_t rs_ct, inc_t cs_ct, \
	  ctype* beta_cast, \
	  ctype* c, inc_t rs_c, inc_t cs_c \
	) \
{ \
	dim_t start, end; \
	dim_t m, n, diag; \
\
	ctype beta_val = *beta_cast; \
	ctype c_ri; \
\
	start = ((n_off < m_off) && (m_off < n_off + n_cur)) ? m_off: n_off; \
	end   = ((n_off < m_off + m_cur) && (m_off + m_cur < n_off + n_cur))? (m_off + m_cur):(n_off + n_cur); \
\
	if( beta_val.real != 0.0 || beta_val.imag != 0.0 ) \
	{ \
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = 0; n <= diag-n_off; n++) \
			{ \
				c_ri = c[m * rs_c + n]; \
				c[m*rs_c + n].real = c[m * rs_c + n].real * beta_val.real \
									- c[m * rs_c + n].imag * beta_val.imag \
									+ ct[m*rs_ct + n].real; \
				c[m*rs_c + n].imag = c_ri.real * beta_val.imag \
									+ c[m * rs_c + n].imag * beta_val.real \
									+ ct[m*rs_ct + n].imag; \
			} \
\
		for(; m < m_cur; m++) \
				for(n = 0; n < n_cur; n++) \
				{ \
					c_ri = c[m * rs_c + n]; \
					c[m*rs_c + n].real = c[m * rs_c + n].real * beta_val.real \
										- c[m * rs_c + n].imag * beta_val.imag \
										+ ct[m*rs_ct + n].real; \
					c[m*rs_c + n].imag = c_ri.real * beta_val.imag \
										+ c[m * rs_c + n].imag * beta_val.real \
										+ ct[m*rs_ct + n].imag; \
				} \
	} \
	else \
	{ \
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = 0; n <= diag-n_off; n++) \
			{ \
				c[m*rs_c + n].real = ct[m*rs_ct + n].real; \
				c[m*rs_c + n].imag = ct[m*rs_ct + n].imag; \
			} \
\
			for(; m < m_cur; m++) \
				for(n = 0; n < n_cur; n++) \
				{ \
					c[m*rs_c + n].real = ct[m*rs_ct + n].real; \
					c[m*rs_c + n].imag = ct[m*rs_ct + n].imag; \
				} \
	} \
\
	return; \
}

INSERT_GENTFUNC_BASIC0_CZ( update_lower_triang )

#undef GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch, varname) \
	( \
	  dim_t m_off, \
	  dim_t n_off, \
	  dim_t m_cur, \
	  dim_t n_cur, \
	  ctype* ct, inc_t rs_ct, inc_t cs_ct, \
	  ctype* beta_cast, \
	  ctype* c, inc_t rs_c, inc_t cs_c \
	) \
{ \
	dim_t start, end; \
	dim_t m, n, diag; \
\
	ctype beta_val = *beta_cast; \
	ctype c_ri; \
\
	start = ((n_off < m_off) && (m_off < n_off + n_cur)) ? m_off: n_off; \
	end   = ((n_off < m_off + m_cur) && (m_off + m_cur < n_off + n_cur))? (m_off + m_cur):(n_off + n_cur); \
	if( beta_val.real != 0.0 || beta_val.imag != 0.0 ) \
	{ \
		for(m = 0; m < start-m_off; m++) \
			for(n = 0; n < n_cur; n++) \
			{ \
				c_ri = c[m * rs_c + n]; \
				c[m*rs_c + n].real = c[m * rs_c + n].real * beta_val.real \
									- c[m * rs_c + n].imag * beta_val.imag \
									+ ct[m*rs_ct + n].real; \
				c[m*rs_c + n].imag = c_ri.real * beta_val.imag \
									+ c[m * rs_c + n].imag * beta_val.real \
									+ ct[m*rs_ct + n].imag; \
			} \
\
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = diag-n_off; n < n_cur; n++) \
			{ \
				c_ri = c[m * rs_c + n]; \
				c[m*rs_c + n].real = c[m * rs_c + n].real * beta_val.real \
									- c[m * rs_c + n].imag * beta_val.imag \
									+ ct[m*rs_ct + n].real; \
				c[m*rs_c + n].imag = c_ri.real * beta_val.imag \
									+ c[m * rs_c + n].imag * beta_val.real \
									+ ct[m*rs_ct + n].imag; \
			} \
	} \
	else \
	{ \
		for(m = 0; m < start-m_off; m++) \
			for(n = 0; n < n_cur; n++) \
			{ \
				c[m*rs_c + n].real = ct[m*rs_ct + n].real; \
				c[m*rs_c + n].imag = ct[m*rs_ct + n].imag; \
			} \
\
		for(diag = start, m= start-m_off; diag < end; diag++, m++) \
			for(n = diag-n_off; n < n_cur; n++) \
			{ \
				c[m*rs_c + n].real = ct[m*rs_ct + n].real; \
				c[m*rs_c + n].imag = ct[m*rs_ct + n].imag; \
			} \
	} \
\
	return; \
}
INSERT_GENTFUNC_BASIC0_CZ( update_upper_triang )

