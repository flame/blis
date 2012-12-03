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

#ifndef BLIS_SCAL2S_H
#define BLIS_SCAL2S_H

// scal2s

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.

// -- (axy) = (ss?) ------------------------------------------------------------

#define bl2_sssscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_ssdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_sscscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bl2_sszscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bl2_sdsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sddscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sdcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_sdzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bl2_scsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bl2_scdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bl2_sccscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag  = ( float  )( ( float  ) (a)      * ( float  ) (x).imag ); \
}
#define bl2_sczscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag  = ( double )( ( float  ) (a)      * ( float  ) (x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bl2_szsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_szdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_szcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bl2_szzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bl2_dssscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dsdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dscscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dszscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bl2_ddsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_dddscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_ddcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bl2_ddzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bl2_dcsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dcdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dccscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bl2_dczscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bl2_dzsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dzdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bl2_dzcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bl2_dzzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag  = ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bl2_cssscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_csdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_cscscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bl2_cszscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bl2_cdsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_cddscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_cdcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bl2_cdzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bl2_ccsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_ccdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_cccscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bl2_cczscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bl2_czsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_czdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_czcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bl2_czzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bl2_zssscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_zsdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bl2_zscscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bl2_zszscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bl2_zdsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_zddscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bl2_zdcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bl2_zdzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bl2_zcsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_zcdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bl2_zccscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bl2_zczscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag  = ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bl2_zzsscal2s( a, x, y ) \
{ \
	(y)       = ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_zzdscal2s( a, x, y ) \
{ \
	(y)       = ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bl2_zzcscal2s( a, x, y ) \
{ \
	(y).real  = ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bl2_zzzscal2s( a, x, y ) \
{ \
	(y).real  = ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag  = ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}






#define bl2_sscal2s( a, x, y ) \
{ \
	bl2_sssscal2s( a, x, y ); \
}
#define bl2_dscal2s( a, x, y ) \
{ \
	bl2_dddscal2s( a, x, y ); \
}
#define bl2_cscal2s( a, x, y ) \
{ \
	bl2_cccscal2s( a, x, y ); \
}
#define bl2_zscal2s( a, x, y ) \
{ \
	bl2_zzzscal2s( a, x, y ); \
}


#endif
