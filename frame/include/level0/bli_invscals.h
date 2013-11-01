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

#ifndef BLIS_INVSCALS_H
#define BLIS_INVSCALS_H

// invscals

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.


#define bli_ssinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_dsinvscals( a, x ) \
{ \
	(x) /= ( float  ) (a); \
}
#define bli_csinvscals( a, x ) \
{ \
	(x) /= ( float  ) bli_creal(a); \
}
#define bli_zsinvscals( a, x ) \
{ \
	(x) /= ( float  ) bli_zreal(a); \
}


#define bli_sdinvscals( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_ddinvscals( a, x ) \
{ \
	(x) /= ( double ) (a); \
}
#define bli_cdinvscals( a, x ) \
{ \
	(x) /= ( double ) bli_creal(a); \
}
#define bli_zdinvscals( a, x ) \
{ \
	(x) /= ( double ) bli_zreal(a); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_scinvscals( a, x ) \
{ \
	bli_creal(x) /= ( float  ) (a); \
	bli_cimag(x) /= ( float  ) (a); \
}
#define bli_dcinvscals( a, x ) \
{ \
	bli_creal(x) /= ( float  ) (a); \
	bli_cimag(x) /= ( float  ) (a); \
}
#define bli_ccinvscals( a, x ) \
{ \
	float  s     = ( float  )bli_fmaxabs( bli_creal(a), bli_cimag(a) ); \
	float  ar_s  = ( float  )bli_creal(a) / s; \
	float  ai_s  = ( float  )bli_cimag(a) / s; \
	float  xr    = ( float  )bli_creal(x); \
	float  temp  = ( ar_s * ( float  )bli_creal(a) + ai_s * ( float  )bli_cimag(a) ); \
	bli_creal(x) = ( bli_creal(x) * ar_s + bli_cimag(x) * ai_s ) / temp; \
	bli_cimag(x) = ( bli_cimag(x) * ar_s - xr           * ai_s ) / temp; \
}
#define bli_zcinvscals( a, x ) \
{ \
	double s     = ( double )bli_fmaxabs( bli_zreal(a), bli_zimag(a) ); \
	double ar_s  = ( double )bli_zreal(a) / s; \
	double ai_s  = ( double )bli_zimag(a) / s; \
	double xr    = ( double )bli_creal(x); \
	double temp  = ( ar_s * ( double )bli_zreal(a) + ai_s * ( double )bli_zimag(a) ); \
	bli_creal(x) = ( bli_creal(x) * ar_s + bli_cimag(x) * ai_s ) / temp; \
	bli_cimag(x) = ( bli_cimag(x) * ar_s - xr           * ai_s ) / temp; \
}


#define bli_szinvscals( a, x ) \
{ \
	bli_zreal(x) /= ( double ) (a); \
	bli_zimag(x) /= ( double ) (a); \
}
#define bli_dzinvscals( a, x ) \
{ \
	bli_zreal(x) /= ( double ) (a); \
	bli_zimag(x) /= ( double ) (a); \
}
#define bli_czinvscals( a, x ) \
{ \
	double s     = ( double )bli_fmaxabs( bli_creal(a), bli_cimag(a) ); \
	double ar_s  = ( double )bli_creal(a) / s; \
	double ai_s  = ( double )bli_cimag(a) / s; \
	double xr    = ( double )bli_zreal(x); \
	double temp  = ( ar_s * ( double )bli_creal(a) + ai_s * ( double )bli_cimag(a) ); \
	bli_zreal(x) = ( bli_zreal(x) * ar_s + bli_zimag(x) * ai_s ) / temp; \
	bli_zimag(x) = ( bli_zimag(x) * ar_s - xr           * ai_s ) / temp; \
}
#define bli_zzinvscals( a, x ) \
{ \
	double s     = ( double )bli_fmaxabs( bli_zreal(a), bli_zimag(a) ); \
	double ar_s  = ( double )bli_zreal(a) / s; \
	double ai_s  = ( double )bli_zimag(a) / s; \
	double xr    = ( double )bli_zreal(x); \
	double temp  = ( ar_s * ( double )bli_zreal(a) + ai_s * ( double )bli_zimag(a) ); \
	bli_zreal(x) = ( bli_zreal(x) * ar_s + bli_zimag(x) * ai_s ) / temp; \
	bli_zimag(x) = ( bli_zimag(x) * ar_s - xr           * ai_s ) / temp; \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_scinvscals( a, x )  { (x) /= (a); }
#define bli_dcinvscals( a, x )  { (x) /= (a); }
#define bli_ccinvscals( a, x )  { (x) /= (a); }
#define bli_zcinvscals( a, x )  { (x) /= (a); }

#define bli_szinvscals( a, x )  { (x) /= (a); }
#define bli_dzinvscals( a, x )  { (x) /= (a); }
#define bli_czinvscals( a, x )  { (x) /= (a); }
#define bli_zzinvscals( a, x )  { (x) /= (a); }


#endif // BLIS_ENABLE_C99_COMPLEX



#define bli_sinvscals( a, x )  bli_ssinvscals( a, x )
#define bli_dinvscals( a, x )  bli_ddinvscals( a, x )
#define bli_cinvscals( a, x )  bli_ccinvscals( a, x )
#define bli_zinvscals( a, x )  bli_zzinvscals( a, x )


#endif
