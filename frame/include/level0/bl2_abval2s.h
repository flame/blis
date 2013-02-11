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

#ifndef BLIS_ABVAL2S_H
#define BLIS_ABVAL2S_H

// abval2s

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of a.


#define bl2_ssabval2s( x, a ) \
{ \
	a = ( float  )fabsf( ( float  )x ); \
}
#define bl2_dsabval2s( x, a ) \
{ \
	a = ( float  )fabs( ( double )x ); \
}
#define bl2_csabval2s( x, a ) \
{ \
	bl2_csabsq2s( x, a ); \
	bl2_sssqrt2s( a, a ); \
}
#define bl2_zsabval2s( x, a ) \
{ \
	bl2_zsabsq2s( x, a ); \
	bl2_sssqrt2s( a, a ); \
}


#define bl2_sdabval2s( x, a ) \
{ \
	a = ( double )fabsf( ( float  )x ); \
}
#define bl2_ddabval2s( x, a ) \
{ \
	a = ( double )fabs( ( double )x ); \
}
#define bl2_cdabval2s( x, a ) \
{ \
	bl2_cdabsq2s( x, a ); \
	bl2_ddsqrt2s( a, a ); \
}
#define bl2_zdabval2s( x, a ) \
{ \
	bl2_zdabsq2s( x, a ); \
	bl2_ddsqrt2s( a, a ); \
}


#define bl2_scabval2s( x, a ) \
{ \
	(a).real = ( float  )fabsf( ( float  )x ); \
	(a).imag = 0.0F; \
}
#define bl2_dcabval2s( x, a ) \
{ \
	(a).real = ( float  )fabs( ( double )x ); \
	(a).imag = 0.0F; \
}
#define bl2_ccabval2s( x, a ) \
{ \
	bl2_ccabsq2s( x, a ); \
	bl2_ccsqrt2s( a, a ); \
}
#define bl2_zcabval2s( x, a ) \
{ \
	bl2_zcabsq2s( x, a ); \
	bl2_ccsqrt2s( a, a ); \
}


#define bl2_szabval2s( x, a ) \
{ \
	(a).real = ( double )fabsf( ( float  )x ); \
	(a).imag = 0.0F; \
}
#define bl2_dzabval2s( x, a ) \
{ \
	(a).real = ( double )fabs( ( double )x ); \
	(a).imag = 0.0F; \
}
#define bl2_czabval2s( x, a ) \
{ \
	bl2_czabsq2s( x, a ); \
	bl2_zzsqrt2s( a, a ); \
}
#define bl2_zzabval2s( x, a ) \
{ \
	bl2_zzabsq2s( x, a ); \
	bl2_zzsqrt2s( a, a ); \
}


#define bl2_sabval2s( x, a ) \
{ \
	bl2_ssabval2s( x, a ); \
}
#define bl2_dabval2s( x, a ) \
{ \
	bl2_ddabval2s( x, a ); \
}
#define bl2_cabval2s( x, a ) \
{ \
	bl2_ccabval2s( x, a ); \
}
#define bl2_zabval2s( x, a ) \
{ \
	bl2_zzabval2s( x, a ); \
}


#endif
