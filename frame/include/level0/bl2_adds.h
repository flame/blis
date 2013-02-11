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

#ifndef BLIS_ADDS_H
#define BLIS_ADDS_H

// adds

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.

#define bl2_ssadds( a, y ) \
{ \
	(y)      += ( float  )(a); \
}
#define bl2_sdadds( a, y ) \
{ \
	(y)      += ( double )(a); \
}
#define bl2_scadds( a, y ) \
{ \
	(y).real += ( float  )(a); \
	/*(y).imag += 0.0F;*/ \
}
#define bl2_szadds( a, y ) \
{ \
	(y).real += ( double )(a); \
	/*(y).imag += 0.0F;*/ \
}

#define bl2_dsadds( a, y ) \
{ \
	(y)      += ( float  )(a); \
}
#define bl2_ddadds( a, y ) \
{ \
	(y)      += ( double )(a); \
}
#define bl2_dcadds( a, y ) \
{ \
	(y).real += ( float  )(a); \
	/*(y).imag += 0.0F;*/ \
}
#define bl2_dzadds( a, y ) \
{ \
	(y).real += ( double )(a); \
	/*(y).imag += 0.0F;*/ \
}

#define bl2_csadds( a, y ) \
{ \
	(y)      += ( float  )(a).real; \
}
#define bl2_cdadds( a, y ) \
{ \
	(y)      += ( double )(a).real; \
}
#define bl2_ccadds( a, y ) \
{ \
	(y).real += ( float  )(a).real; \
	(y).imag += ( float  )(a).imag; \
}
#define bl2_czadds( a, y ) \
{ \
	(y).real += ( double )(a).real; \
	(y).imag += ( double )(a).imag; \
}

#define bl2_zsadds( a, y ) \
{ \
	(y)      += ( float  )(a).real; \
}
#define bl2_zdadds( a, y ) \
{ \
	(y)      += ( double )(a).real; \
}
#define bl2_zcadds( a, y ) \
{ \
	(y).real += ( float  )(a).real; \
	(y).imag += ( float  )(a).imag; \
}
#define bl2_zzadds( a, y ) \
{ \
	(y).real += ( double )(a).real; \
	(y).imag += ( double )(a).imag; \
}


#define bl2_sadds( a, y ) \
{ \
	bl2_ssadds( a, y ); \
}
#define bl2_dadds( a, y ) \
{ \
	bl2_ddadds( a, y ); \
}
#define bl2_cadds( a, y ) \
{ \
	bl2_ccadds( a, y ); \
}
#define bl2_zadds( a, y ) \
{ \
	bl2_zzadds( a, y ); \
}


#endif
