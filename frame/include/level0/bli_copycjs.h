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

#ifndef BLIS_COPYCJS_H
#define BLIS_COPYCJS_H

// copycjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.


#define bli_sscopycjs( conj, x, y ) \
{ \
	bli_sscopys( x, y ); \
}
#define bli_dscopycjs( conj, x, y ) \
{ \
	bli_dscopys( x, y ); \
}
#define bli_cscopycjs( conj, x, y ) \
{ \
	bli_cscopys( x, y ); \
}
#define bli_zscopycjs( conj, x, y ) \
{ \
	bli_zscopys( x, y ); \
}


#define bli_sdcopycjs( conj, x, y ) \
{ \
	bli_sdcopys( x, y ); \
}
#define bli_ddcopycjs( conj, x, y ) \
{ \
	bli_ddcopys( x, y ); \
}
#define bli_cdcopycjs( conj, x, y ) \
{ \
	bli_cdcopys( x, y ); \
}
#define bli_zdcopycjs( conj, x, y ) \
{ \
	bli_zdcopys( x, y ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_sccopycjs( conj, x, y ) \
{ \
	bli_sccopys( x, y ); \
}
#define bli_dccopycjs( conj, x, y ) \
{ \
	bli_dccopys( x, y ); \
}
#define bli_cccopycjs( conj, x, y ) \
{ \
	bli_creal(y) =                          bli_creal(x); \
	bli_cimag(y) = ( bli_is_conj( conj ) ? -bli_cimag(x) \
	                                     :  bli_cimag(x) ); \
}
#define bli_zccopycjs( conj, x, y ) \
{ \
	bli_creal(y) =                          bli_zreal(x); \
	bli_cimag(y) = ( bli_is_conj( conj ) ? -bli_zimag(x) \
	                                     :  bli_zimag(x) ); \
}


#define bli_szcopycjs( conj, x, y ) \
{ \
	bli_szcopys( x, y ); \
}
#define bli_dzcopycjs( conj, x, y ) \
{ \
	bli_dzcopys( x, y ); \
}
#define bli_czcopycjs( conj, x, y ) \
{ \
	bli_zreal(y) =                          bli_creal(x); \
	bli_zimag(y) = ( bli_is_conj( conj ) ? -bli_cimag(x) \
	                                     :  bli_cimag(x) ); \
}
#define bli_zzcopycjs( conj, x, y ) \
{ \
	bli_zreal(y) =                          bli_zreal(x); \
	bli_zimag(y) = ( bli_is_conj( conj ) ? -bli_zimag(x) \
	                                     :  bli_zimag(x) ); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_sccopycjs( conj, x, y )  { (y) = (x); }
#define bli_dccopycjs( conj, x, y )  { (y) = (x); }
#define bli_cccopycjs( conj, x, y )  { (y) = ( bli_is_conj( conj ) ? conjf(x) : (x) ); }
#define bli_zccopycjs( conj, x, y )  { (y) = ( bli_is_conj( conj ) ? conj(x)  : (x) ); }

#define bli_szcopycjs( conj, x, y )  { (y) = (x); }
#define bli_dzcopycjs( conj, x, y )  { (y) = (x); }
#define bli_czcopycjs( conj, x, y )  { (y) = ( bli_is_conj( conj ) ? conjf(x) : (x) ); }
#define bli_zzcopycjs( conj, x, y )  { (y) = ( bli_is_conj( conj ) ? conj(x)  : (x) ); }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_iicopycjs( conj, x, y ) \
{ \
	(y) = ( gint_t ) (x); \
}


#define bli_scopycjs( conj, x, y )  bli_sscopycjs( conj, x, y )
#define bli_dcopycjs( conj, x, y )  bli_ddcopycjs( conj, x, y )
#define bli_ccopycjs( conj, x, y )  bli_cccopycjs( conj, x, y )
#define bli_zcopycjs( conj, x, y )  bli_zzcopycjs( conj, x, y )
#define bli_icopycjs( conj, x, y )  bli_iicopycjs( conj, x, y )


#endif
