/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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


#define bli_sscast( ap, b ) \
{ \
	(b) = ( float  )             *(( float*    )(ap)); \
}
#define bli_dscast( ap, b ) \
{ \
	(b) = ( float  )             *(( double*   )(ap)); \
}
#define bli_cscast( ap, b ) \
{ \
	(b) = ( float  )  bli_creal( *(( scomplex* )(ap)) ); \
}
#define bli_zscast( ap, b ) \
{ \
	(b) = ( float  )  bli_zreal( *(( dcomplex* )(ap)) ); \
}


#define bli_sdcast( ap, b ) \
{ \
	(b) = ( double )             *(( float*    )(ap)); \
}
#define bli_ddcast( ap, b ) \
{ \
	(b) = ( double )             *(( double*   )(ap)); \
}
#define bli_cdcast( ap, b ) \
{ \
	(b) = ( double )  bli_creal( *(( scomplex* )(ap)) ); \
}
#define bli_zdcast( ap, b ) \
{ \
	(b) = ( double )  bli_zreal( *(( dcomplex* )(ap)) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_sccast( ap, b ) \
{ \
	bli_scsets( bli_sreal( *(( float*    )(ap)) ), \
	                                          0.0, (b) ); \
}
#define bli_dccast( ap, b ) \
{ \
	bli_dcsets( bli_dreal( *(( double*   )(ap)) ), \
	                                          0.0, (b) ); \
}
#define bli_cccast( ap, b ) \
{ \
	bli_ccsets( bli_creal( *(( scomplex* )(ap)) ), \
	            bli_cimag( *(( scomplex* )(ap)) ), (b) ); \
}
#define bli_zccast( ap, b ) \
{ \
	bli_zcsets( bli_zreal( *(( dcomplex* )(ap)) ), \
	            bli_zimag( *(( dcomplex* )(ap)) ), (b) ); \
}


#define bli_szcast( ap, b ) \
{ \
	bli_szsets( bli_sreal( *(( float*    )(ap)) ), \
	                                          0.0, (b) ); \
}
#define bli_dzcast( ap, b ) \
{ \
	bli_dzsets( bli_dreal( *(( double*   )(ap)) ), \
	                                          0.0, (b) ); \
}
#define bli_czcast( ap, b ) \
{ \
	bli_czsets( bli_creal( *(( scomplex* )(ap)) ), \
	            bli_cimag( *(( scomplex* )(ap)) ), (b) ); \
}
#define bli_zzcast( ap, b ) \
{ \
	bli_zzsets( bli_zreal( *(( dcomplex* )(ap)) ), \
	            bli_zimag( *(( dcomplex* )(ap)) ), (b) ); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_sccast( ap, b )  { (b) = ( scomplex ) *(( float*    )(ap)); }
#define bli_dccast( ap, b )  { (b) = ( scomplex ) *(( double*   )(ap)); }
#define bli_cccast( ap, b )  { (b) = ( scomplex ) *(( scomplex* )(ap)); }
#define bli_zccast( ap, b )  { (b) = ( scomplex ) *(( dcomplex* )(ap)); }

#define bli_szcast( ap, b )  { (b) = ( dcomplex ) *(( float*    )(ap)); }
#define bli_dzcast( ap, b )  { (b) = ( dcomplex ) *(( double*   )(ap)); }
#define bli_czcast( ap, b )  { (b) = ( dcomplex ) *(( scomplex* )(ap)); }
#define bli_zzcast( ap, b )  { (b) = ( dcomplex ) *(( dcomplex* )(ap)); }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_scast( ap, b )  bli_sscast( ap, b )
#define bli_dcast( ap, b )  bli_ddcast( ap, b )
#define bli_ccast( ap, b )  bli_cccast( ap, b )
#define bli_zcast( ap, b )  bli_zzcast( ap, b )

#endif
