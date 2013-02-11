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

#ifndef BLIS_CAST_H
#define BLIS_CAST_H

// cast

// Notes:
// - The first char encodes the type of *ap.
// - The second char encodes the type of b.

#define bl2_sscast( ap, b ) \
{ \
	(b) = ( float  ) *(( float*    )(ap)); \
}
#define bl2_dscast( ap, b ) \
{ \
	(b) = ( float  ) *(( double*   )(ap)); \
}
#define bl2_cscast( ap, b ) \
{ \
	(b) = ( float  )  (( scomplex* )(ap))->real; \
}
#define bl2_zscast( ap, b ) \
{ \
	(b) = ( float  )  (( dcomplex* )(ap))->real; \
}

#define bl2_sdcast( ap, b ) \
{ \
	(b) = ( double ) *(( float*    )(ap)); \
}
#define bl2_ddcast( ap, b ) \
{ \
	(b) = ( double ) *(( double*   )(ap)); \
}
#define bl2_cdcast( ap, b ) \
{ \
	(b) = ( double )  (( scomplex* )(ap))->real; \
}
#define bl2_zdcast( ap, b ) \
{ \
	(b) = ( double )  (( dcomplex* )(ap))->real; \
}

#define bl2_sccast( ap, b ) \
{ \
	(b).real = ( float  ) *(( float*    )(ap)); \
	(b).imag = 0.0F; \
}
#define bl2_dccast( ap, b ) \
{ \
	(b).real = ( float  ) *(( double*   )(ap)); \
	(b).imag = 0.0F; \
}
#define bl2_cccast( ap, b ) \
{ \
	(b).real = ( float  )  (( scomplex* )(ap))->real; \
	(b).imag = ( float  )  (( scomplex* )(ap))->imag; \
}
#define bl2_zccast( ap, b ) \
{ \
	(b).real = ( float  )  (( dcomplex* )(ap))->real; \
	(b).imag = ( float  )  (( dcomplex* )(ap))->imag; \
}

#define bl2_szcast( ap, b ) \
{ \
	(b).real = ( double ) *(( float*    )(ap)); \
	(b).imag = 0.0; \
}
#define bl2_dzcast( ap, b ) \
{ \
	(b).real = ( double ) *(( double*   )(ap)); \
	(b).imag = 0.0; \
}
#define bl2_czcast( ap, b ) \
{ \
	(b).real = ( double )  (( scomplex* )(ap))->real; \
	(b).imag = ( double )  (( scomplex* )(ap))->imag; \
}
#define bl2_zzcast( ap, b ) \
{ \
	(b).real = ( double )  (( dcomplex* )(ap))->real; \
	(b).imag = ( double )  (( dcomplex* )(ap))->imag; \
}

#define bl2_scast( ap, b ) \
\
	bl2_sscast( ap, b );

#define bl2_dcast( ap, b ) \
\
	bl2_ddcast( ap, b );

#define bl2_ccast( ap, b ) \
\
	bl2_cccast( ap, b );

#define bl2_zcast( ap, b ) \
\
	bl2_zzcast( ap, b );

#endif
