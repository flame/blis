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

#ifndef BLIS_SET0S_EDGE_H
#define BLIS_SET0S_EDGE_H

// set0s_mxn

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define GENTFUNC(ctype,ch,op) \
\
BLIS_INLINE void PASTEMAC(ch,op) \
     ( \
       const dim_t     i, \
       const dim_t     m, \
       const dim_t     j, \
       const dim_t     n, \
       ctype* restrict p, \
       const inc_t     ldp \
     ) \
{ \
	if ( i < m ) \
	{ \
		PASTEMAC(ch,set0s_mxn) \
		( \
		  m - i, \
		  j, \
		  p + i*1, 1, ldp \
		); \
	} \
\
	if ( j < n ) \
	{ \
		PASTEMAC(ch,set0s_mxn) \
		( \
		  m, \
		  n - j, \
		  p + j*ldp, 1, ldp \
		); \
	} \
}

INSERT_GENTFUNC_BASIC(set0s_edge)

#endif
