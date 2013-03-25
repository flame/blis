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

#ifndef BLIS_SCAL2JS_H
#define BLIS_SCAL2JS_H

// scal2js

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.
// - x is used in conjugated form.

// -- (axy) = (ss?) ------------------------------------------------------------

#define bli_sssscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_ssdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sscscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sszscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bli_sdsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sddscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bli_scsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_scdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_sccscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag  = ( float  )( ( float  ) (a)      * ( float  )-(x).imag ); \
}
#define bli_sczscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag  = ( double )( ( float  ) (a)      * ( float  )-(x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bli_szsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bli_szzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bli_dssscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dsdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dscscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dszscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bli_ddsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dddscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bli_dcsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dcdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dccscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bli_dczscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bli_dzsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( float  )( ( double ) (a)      * ( double )-(x).imag ); \
}
#define bli_dzzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( double )( ( double ) (a)      * ( double )-(x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bli_cssscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_csdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_cscscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_cszscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bli_cdsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cddscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cdcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_cdzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bli_ccsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_ccdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_cccscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_cczscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bli_czsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_czzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bli_zssscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zsdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zscscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_zszscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bli_zdsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zddscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zdcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_zdzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bli_zcsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zcdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zccscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_zczscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x).real + ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x).real - ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bli_zzsscal2js( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzdscal2js( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzcscal2js( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_zzzscal2js( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x).real + ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x).real - ( double ) (a).real * ( double ) (x).imag ); \
}






#define bli_sscal2js( a, x, y ) \
{ \
	bli_sssscal2js( a, x, y ); \
}
#define bli_dscal2js( a, x, y ) \
{ \
	bli_dddscal2js( a, x, y ); \
}
#define bli_cscal2js( a, x, y ) \
{ \
	bli_cccscal2js( a, x, y ); \
}
#define bli_zscal2js( a, x, y ) \
{ \
	bli_zzzscal2js( a, x, y ); \
}


#endif
