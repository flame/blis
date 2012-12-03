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

#ifndef BLIS_INVSCALS_H
#define BLIS_INVSCALS_H

// invscals

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.

#define bl2_ssinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bl2_dsinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bl2_csinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}
#define bl2_zsinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a).real; \
}

#define bl2_sdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bl2_ddinvscals( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bl2_cdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}
#define bl2_zdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a).real; \
}

#define bl2_scinvscals( a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bl2_dcinvscals( a, x ) \
{ \
	(x).real /= ( float  ) (a); \
	(x).imag /= ( float  ) (a); \
}
#define bl2_ccinvscals( a, x ) \
{ \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) (a).imag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real + ( float  ) (a).imag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag - ( float  ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bl2_zcinvscals( a, x ) \
{ \
	float  temp =              ( float  ) (a).real * (a).real + ( float  ) (a).imag * (a).imag; \
	float  xr   = ( float  ) ( ( float  ) (a).real * (x).real + ( float  ) (a).imag * (x).imag ) / temp; \
	float  xi   = ( float  ) ( ( float  ) (a).real * (x).imag - ( float  ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}

#define bl2_szinvscals( a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bl2_dzinvscals( a, x ) \
{ \
	(x).real /= ( double ) (a); \
	(x).imag /= ( double ) (a); \
}
#define bl2_czinvscals( a, x ) \
{ \
	double temp =              ( double ) (a).real * (a).real + ( double ) (a).imag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real + ( double ) (a).imag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag - ( double ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}
#define bl2_zzinvscals( a, x ) \
{ \
	double temp =              ( double ) (a).real * (a).real + ( double ) (a).imag * (a).imag; \
	double xr   = ( double ) ( ( double ) (a).real * (x).real + ( double ) (a).imag * (x).imag ) / temp; \
	double xi   = ( double ) ( ( double ) (a).real * (x).imag - ( double ) (a).imag * (x).real ) / temp; \
	(x).real    = xr; \
	(x).imag    = xi; \
}


#define bl2_sinvscals( a, x ) \
{ \
	bl2_ssinvscals( a, x ); \
}
#define bl2_dinvscals( a, x ) \
{ \
	bl2_ddinvscals( a, x ); \
}
#define bl2_cinvscals( a, x ) \
{ \
	bl2_ccinvscals( a, x ); \
}
#define bl2_zinvscals( a, x ) \
{ \
	bl2_zzinvscals( a, x ); \
}


#endif
