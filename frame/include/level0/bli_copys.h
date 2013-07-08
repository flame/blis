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

#ifndef BLIS_COPYS_H
#define BLIS_COPYS_H

// copys

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bli_sscopys( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_dscopys( x, y ) \
{ \
	(y) = ( float  ) (x); \
}
#define bli_cscopys( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}
#define bli_zscopys( x, y ) \
{ \
	(y) = ( float  ) (x).real; \
}

#define bli_sdcopys( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_ddcopys( x, y ) \
{ \
	(y) = ( double ) (x); \
}
#define bli_cdcopys( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}
#define bli_zdcopys( x, y ) \
{ \
	(y) = ( double ) (x).real; \
}

#define bli_sccopys( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	(y).imag = 0.0F; \
}
#define bli_dccopys( x, y ) \
{ \
	(y).real = ( float  ) (x); \
	(y).imag = 0.0F; \
}
#define bli_cccopys( x, y ) \
{ \
	(y).real = ( float  ) (x).real; \
	(y).imag = ( float  ) (x).imag; \
}
#define bli_zccopys( x, y ) \
{ \
	(y).real = ( float  ) (x).real; \
	(y).imag = ( float  ) (x).imag; \
}

#define bli_szcopys( x, y ) \
{ \
	(y).real = ( double ) (x); \
	(y).imag = 0.0; \
}
#define bli_dzcopys( x, y ) \
{ \
	(y).real = ( double ) (x); \
	(y).imag = 0.0; \
}
#define bli_czcopys( x, y ) \
{ \
	(y).real = ( double ) (x).real; \
	(y).imag = ( double ) (x).imag; \
}
#define bli_zzcopys( x, y ) \
{ \
	(y).real = ( double ) (x).real; \
	(y).imag = ( double ) (x).imag; \
}

#define bli_iicopys( x, y ) \
{ \
	(y) = ( gint_t ) (x); \
}


#define bli_scopys( x, y ) \
{ \
	bli_sscopys( x, y ); \
}
#define bli_dcopys( x, y ) \
{ \
	bli_ddcopys( x, y ); \
}
#define bli_ccopys( x, y ) \
{ \
	bli_cccopys( x, y ); \
}
#define bli_zcopys( x, y ) \
{ \
	bli_zzcopys( x, y ); \
}
#define bli_icopys( x, y ) \
{ \
	bli_iicopys( x, y ); \
}


#endif
