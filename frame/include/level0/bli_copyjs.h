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

#ifndef BLIS_COPYJS_H
#define BLIS_COPYJS_H

// copyjs

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.


#define bli_sscopyjs( x, y ) \
{ \
	(y) = ( float  ) bli_sreal(x); \
}
#define bli_dscopyjs( x, y ) \
{ \
	(y) = ( float  ) bli_dreal(x); \
}
#define bli_cscopyjs( x, y ) \
{ \
	(y) = ( float  ) bli_creal(x); \
}
#define bli_zscopyjs( x, y ) \
{ \
	(y) = ( float  ) bli_zreal(x); \
}


#define bli_sdcopyjs( x, y ) \
{ \
	(y) = ( double ) bli_sreal(x); \
}
#define bli_ddcopyjs( x, y ) \
{ \
	(y) = ( double ) bli_dreal(x); \
}
#define bli_cdcopyjs( x, y ) \
{ \
	(y) = ( double ) bli_creal(x); \
}
#define bli_zdcopyjs( x, y ) \
{ \
	(y) = ( double ) bli_zreal(x); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_sccopyjs( x, y ) \
{ \
	bli_creal(y) = ( float  ) bli_sreal(x); \
	bli_cimag(y) = ( float  ) bli_simag(x); \
}
#define bli_dccopyjs( x, y ) \
{ \
	bli_creal(y) = ( float  ) bli_dreal(x); \
	bli_cimag(y) = ( float  ) bli_dimag(x); \
}
#define bli_cccopyjs( x, y ) \
{ \
	bli_creal(y) = ( float  ) bli_creal(x); \
	bli_cimag(y) = ( float  )-bli_cimag(x); \
}
#define bli_zccopyjs( x, y ) \
{ \
	bli_creal(y) = ( float  ) bli_zreal(x); \
	bli_cimag(y) = ( float  )-bli_zimag(x); \
}


#define bli_szcopyjs( x, y ) \
{ \
	bli_zreal(y) = ( double ) bli_sreal(x); \
	bli_zimag(y) = ( double ) bli_simag(x); \
}
#define bli_dzcopyjs( x, y ) \
{ \
	bli_zreal(y) = ( double ) bli_dreal(x); \
	bli_zimag(y) = ( double ) bli_dimag(x); \
}
#define bli_czcopyjs( x, y ) \
{ \
	bli_zreal(y) = ( double ) bli_creal(x); \
	bli_zimag(y) = ( double )-bli_cimag(x); \
}
#define bli_zzcopyjs( x, y ) \
{ \
	bli_zreal(y) = ( double ) bli_zreal(x); \
	bli_zimag(y) = ( double )-bli_zimag(x); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_sccopyjs( x, y )  { (y) = (x); }
#define bli_dccopyjs( x, y )  { (y) = (x); }
#define bli_cccopyjs( x, y )  { (y) = conjf(x); }
#define bli_zccopyjs( x, y )  { (y) = conj(x);  }

#define bli_szcopyjs( x, y )  { (y) = (x); }
#define bli_dzcopyjs( x, y )  { (y) = (x); }
#define bli_czcopyjs( x, y )  { (y) = conjf(x); }
#define bli_zzcopyjs( x, y )  { (y) = conj(x);  }


#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_iicopyjs( x, y ) \
{ \
	(y) = ( gint_t ) (x); \
}


#define bli_scopyjs( x, y )  bli_sscopyjs( x, y )
#define bli_dcopyjs( x, y )  bli_ddcopyjs( x, y )
#define bli_ccopyjs( x, y )  bli_cccopyjs( x, y )
#define bli_zcopyjs( x, y )  bli_zzcopyjs( x, y )
#define bli_icopyjs( x, y )  bli_iicopyjs( x, y )


#endif
