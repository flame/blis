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

#ifndef BLIS_CONSTANTS_H
#define BLIS_CONSTANTS_H

// return pointers to constants

// 2

#define bl2_s2 \
\
	( BLIS_CONST_S_PTR( BLIS_TWO ) )

#define bl2_d2 \
\
	( BLIS_CONST_D_PTR( BLIS_TWO ) )

#define bl2_c2 \
\
	( BLIS_CONST_C_PTR( BLIS_TWO ) )

#define bl2_z2 \
\
	( BLIS_CONST_Z_PTR( BLIS_TWO ) )

// 1

#define bl2_s1 \
\
	( BLIS_CONST_S_PTR( BLIS_ONE ) )

#define bl2_d1 \
\
	( BLIS_CONST_D_PTR( BLIS_ONE ) )

#define bl2_c1 \
\
	( BLIS_CONST_C_PTR( BLIS_ONE ) )

#define bl2_z1 \
\
	( BLIS_CONST_Z_PTR( BLIS_ONE ) )

// 0

#define bl2_s0 \
\
	( BLIS_CONST_S_PTR( BLIS_ZERO ) )

#define bl2_d0 \
\
	( BLIS_CONST_D_PTR( BLIS_ZERO ) )

#define bl2_c0 \
\
	( BLIS_CONST_C_PTR( BLIS_ZERO ) )

#define bl2_z0 \
\
	( BLIS_CONST_Z_PTR( BLIS_ZERO ) )

// -1

#define bl2_sm1 \
\
	( BLIS_CONST_S_PTR( BLIS_MINUS_ONE ) )

#define bl2_dm1 \
\
	( BLIS_CONST_D_PTR( BLIS_MINUS_ONE ) )

#define bl2_cm1 \
\
	( BLIS_CONST_C_PTR( BLIS_MINUS_ONE ) )

#define bl2_zm1 \
\
	( BLIS_CONST_Z_PTR( BLIS_MINUS_ONE ) )

// -2

#define bl2_sm2 \
\
	( BLIS_CONST_S_PTR( BLIS_MINUS_TWO ) )

#define bl2_dm2 \
\
	( BLIS_CONST_D_PTR( BLIS_MINUS_TWO ) )

#define bl2_cm2 \
\
	( BLIS_CONST_C_PTR( BLIS_MINUS_TWO ) )

#define bl2_zm2 \
\
	( BLIS_CONST_Z_PTR( BLIS_MINUS_TWO ) )


// set to constant

// set0s

#define bl2_sset0s( a ) \
{ \
	(a) = 0.0F; \
}
#define bl2_dset0s( a ) \
{ \
	(a) = 0.0; \
}
#define bl2_cset0s( a ) \
{ \
	(a).real = 0.0F; \
	(a).imag = 0.0F; \
}
#define bl2_zset0s( a ) \
{ \
	(a).real = 0.0; \
	(a).imag = 0.0; \
}

// setimag0

#define bl2_ssetimag0( a ) \
{ \
	; \
}
#define bl2_dsetimag0( a ) \
{ \
	; \
}
#define bl2_csetimag0( a ) \
{ \
	(a).imag = 0.0F; \
}
#define bl2_zsetimag0( a ) \
{ \
	(a).imag = 0.0; \
}

#endif
