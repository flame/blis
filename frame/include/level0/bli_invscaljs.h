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

#ifndef BLIS_INVSCALJS_H
#define BLIS_INVSCALJS_H

// invscaljs

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - a is used in conjugated form.

#define bli_ssinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_dsinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_csinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}
#define bli_zsinvscaljs( a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}

#define bli_sdinvscaljs( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_ddinvscaljs( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_cdinvscaljs( a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}
#define bli_zdinvscaljs( a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}

#define bli_scinvscaljs( a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bli_dcinvscaljs( a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bli_ccinvscaljs( a, x ) \
{ \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) (a).imag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real - ( float  ) (a).imag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag + ( float  ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bli_zcinvscaljs( a, x ) \
{ \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) (a).imag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real - ( float  ) (a).imag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag + ( float  ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}

#define bli_szinvscaljs( a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bli_dzinvscaljs( a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bli_czinvscaljs( a, x ) \
{ \
	double temp =              ( double ) (a).real * (a).real + ( double ) (a).imag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real - ( double ) (a).imag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag + ( double ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bli_zzinvscaljs( a, x ) \
{ \
	double temp =              ( double ) (a).real * (a).real + ( double ) (a).imag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real - ( double ) (a).imag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag + ( double ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}


#define bli_sinvscaljs( a, x ) \
{ \
	bli_ssinvscaljs( a, x ); \
}
#define bli_dinvscaljs( a, x ) \
{ \
	bli_ddinvscaljs( a, x ); \
}
#define bli_cinvscaljs( a, x ) \
{ \
	bli_ccinvscaljs( a, x ); \
}
#define bli_zinvscaljs( a, x ) \
{ \
	bli_zzinvscaljs( a, x ); \
}


#endif
