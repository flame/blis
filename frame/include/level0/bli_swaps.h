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

#ifndef BLIS_SWAPS_H
#define BLIS_SWAPS_H

// swaps

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bli_ssswaps( x, y ) \
{ \
	float    xnew; \
	xnew      = ( float  ) (y); \
	(y)       = ( float  ) (x); \
	(x)       =            xnew; \
}
#define bli_dsswaps( x, y ) \
{ \
	double   xnew; \
	xnew      = ( double ) (y); \
	(y)       = ( float  ) (x); \
	(x)       =            xnew; \
}
#define bli_csswaps( x, y ) \
{ \
	float    xnew; \
	xnew      = ( float  ) (y); \
	(y)       = ( float  ) (x).real; \
	(x).real  =            xnew; \
	(x).imag  = 0.0F; \
}
#define bli_zsswaps( x, y ) \
{ \
	double   xnew; \
	xnew      = ( double ) (y); \
	(y)       = ( float  ) (x).real; \
	(x).real  =            xnew; \
	(x).imag  = 0.0; \
}

#define bli_sdswaps( x, y ) \
{ \
	float    xnew; \
	xnew      = ( float  ) (y); \
	(y)       = ( double ) (x); \
	(x)       =            xnew; \
}
#define bli_ddswaps( x, y ) \
{ \
	double   xnew; \
	xnew      = ( double ) (y); \
	(y)       = ( double ) (x); \
	(x)       =            xnew; \
}
#define bli_cdswaps( x, y ) \
{ \
	float    xnew; \
	xnew      = ( float  ) (y); \
	(y)       = ( double ) (x).real; \
	(x).real  =            xnew; \
	(x).imag  = 0.0F; \
}
#define bli_zdswaps( x, y ) \
{ \
	double   xnew; \
	xnew      = ( double ) (y); \
	(y)       = ( double ) (x).real; \
	(x).real  =            xnew; \
	(x).imag  = 0.0; \
}

#define bli_scswaps( x, y ) \
{ \
	float    xnew; \
	xnew      = ( float  ) (y).real; \
	(y).real  = ( float  ) (x); \
	(y).imag  = 0.0F; \
	(x)       =            xnew; \
}
#define bli_dcswaps( x, y ) \
{ \
	double   xnew; \
	xnew      = ( double ) (y).real; \
	(y).real  = ( float  ) (x); \
	(y).imag  = 0.0F; \
	(x)       =            xnew; \
}
#define bli_ccswaps( x, y ) \
{ \
	scomplex xnew; \
	xnew      =            (y); \
	(y)       =            (x); \
	(x)       =            xnew; \
}
#define bli_zcswaps( x, y ) \
{ \
	dcomplex xnew; \
	xnew.real = ( double ) (y).real; \
	xnew.imag = ( double ) (y).imag; \
	(y).real  = ( float  ) (x).real; \
	(y).imag  = ( float  ) (x).imag; \
	(x)       =            xnew; \
}

#define bli_szswaps( x, y ) \
{ \
	float    xnew; \
	xnew      = ( float  ) (y).real; \
	(y).real  = ( double ) (x); \
	(y).imag  = 0.0; \
	(x)       =            xnew; \
}
#define bli_dzswaps( x, y ) \
{ \
	double   xnew; \
	xnew      = ( double ) (y).real; \
	(y).real  = ( double ) (x); \
	(y).imag  = 0.0; \
	(x)       =            xnew; \
}
#define bli_czswaps( x, y ) \
{ \
	scomplex xnew; \
	xnew.real = ( float  ) (y).real; \
	xnew.imag = ( float  ) (y).imag; \
	(y).real  = ( double ) (x).real; \
	(y).imag  = ( double ) (x).imag; \
	(x)       =            xnew; \
}
#define bli_zzswaps( x, y ) \
{ \
	dcomplex xnew; \
	xnew      =            (y); \
	(y)       =            (x); \
	(x)       =            xnew; \
}


#define bli_sswaps( x, y ) \
{ \
	bli_ssswaps( x, y ); \
}
#define bli_dswaps( x, y ) \
{ \
	bli_ddswaps( x, y ); \
}
#define bli_cswaps( x, y ) \
{ \
	bli_ccswaps( x, y ); \
}
#define bli_zswaps( x, y ) \
{ \
	bli_zzswaps( x, y ); \
}


#endif
