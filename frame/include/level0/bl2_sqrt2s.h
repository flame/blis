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

#ifndef BLIS_SQRT2S_H
#define BLIS_SQRT2S_H

// sqrt2s

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of a.


#define bl2_sssqrt2s( x, a ) \
{ \
	(a) = ( float  )sqrtf( (x) ); \
}
#define bl2_dssqrt2s( x, a ) \
{ \
	(a) = ( float  )sqrt( (x) ); \
}
#define bl2_cssqrt2s( x, a ) \
{ \
	float  mag = sqrtf( (x).real * (x).real + \
	                    (x).imag * (x).imag ); \
\
	(a)      = ( float  )sqrt( ( mag + (x).real ) / 2.0F ); \
}
#define bl2_zssqrt2s( x, a ) \
{ \
	double mag = sqrt( (x).real * (x).real + \
	                   (x).imag * (x).imag ); \
\
	(a)      = ( float  )sqrt( ( mag + (x).real ) / 2.0 ); \
}


#define bl2_sdsqrt2s( x, a ) \
{ \
	(a) = ( double )sqrtf( (x) ); \
}
#define bl2_ddsqrt2s( x, a ) \
{ \
	(a) = ( double )sqrt( (x) ); \
}
#define bl2_cdsqrt2s( x, a ) \
{ \
	float  mag = sqrtf( (x).real * (x).real + \
	                    (x).imag * (x).imag ); \
\
	(a)      = ( double )sqrt( ( mag + (x).real ) / 2.0F ); \
}
#define bl2_zdsqrt2s( x, a ) \
{ \
	double mag = sqrt( (x).real * (x).real + \
	                   (x).imag * (x).imag ); \
\
	(a)      = ( double )sqrt( ( mag + (x).real ) / 2.0 ); \
}


#define bl2_scsqrt2s( x, a ) \
{ \
	(a).real = ( float  )sqrtf( (x) ); \
	(a).imag = 0.0F; \
}
#define bl2_dcsqrt2s( x, a ) \
{ \
	(a).real = ( float  )sqrt( (x) ); \
	(a).imag = 0.0F; \
}
#define bl2_ccsqrt2s( x, a ) \
{ \
	float  mag = sqrtf( (x).real * (x).real + \
	                    (x).imag * (x).imag ); \
\
	(a).real = ( float  )sqrtf( ( mag + (x).real ) / 2.0F ); \
	(a).imag = ( float  )sqrtf( ( mag - (x).imag ) / 2.0F ); \
}
#define bl2_zcsqrt2s( x, a ) \
{ \
	double mag = sqrt( (x).real * (x).real + \
	                   (x).imag * (x).imag ); \
\
	(a).real = ( float  )sqrt( ( mag + (x).real ) / 2.0 ); \
	(a).imag = ( float  )sqrt( ( mag - (x).imag ) / 2.0 ); \
}


#define bl2_szsqrt2s( x, a ) \
{ \
	(a).real = ( double )sqrtf( (x) ); \
	(a).imag = 0.0F; \
}
#define bl2_dzsqrt2s( x, a ) \
{ \
	(a).real = ( double )sqrt( (x) ); \
	(a).imag = 0.0F; \
}
#define bl2_czsqrt2s( x, a ) \
{ \
	float  mag = sqrtf( (x).real * (x).real + \
	                    (x).imag * (x).imag ); \
\
	(a).real = ( double )sqrtf( ( mag + (x).real ) / 2.0F ); \
	(a).imag = ( double )sqrtf( ( mag - (x).imag ) / 2.0F ); \
}
#define bl2_zzsqrt2s( x, a ) \
{ \
	double mag = sqrt( (x).real * (x).real + \
	                   (x).imag * (x).imag ); \
\
	(a).real = ( double )sqrt( ( mag + (x).real ) / 2.0 ); \
	(a).imag = ( double )sqrt( ( mag - (x).imag ) / 2.0 ); \
}


#define bl2_ssqrt2s( x, a ) \
{ \
	bl2_sssqrt2s( x, a ); \
}
#define bl2_dsqrt2s( x, a ) \
{ \
	bl2_ddsqrt2s( x, a ); \
}
#define bl2_csqrt2s( x, a ) \
{ \
	bl2_ccsqrt2s( x, a ); \
}
#define bl2_zsqrt2s( x, a ) \
{ \
	bl2_zzsqrt2s( x, a ); \
}


#endif
