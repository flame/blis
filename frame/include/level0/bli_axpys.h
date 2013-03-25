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

#ifndef BLIS_AXPYS_H
#define BLIS_AXPYS_H

// axpys

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.

// -- (axy) = (ss?) ------------------------------------------------------------

#define bli_sssaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_ssdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sscaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sszaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bli_sdsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sddaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bli_scsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_scdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_sccaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag += ( float  )( ( float  ) (a)      * ( float  ) (x).imag ); \
}
#define bli_sczaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag += ( double )( ( float  ) (a)      * ( float  ) (x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bli_szsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bli_szzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bli_dssaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dsdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dscaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dszaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bli_ddsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dddaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bli_dcsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dcdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dccaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bli_dczaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bli_dzsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bli_dzzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag += ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bli_cssaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_csdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_cscaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_cszaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bli_cdsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cddaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cdcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_cdzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bli_ccsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_ccdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_cccaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_cczaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bli_czsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_czzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bli_zssaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zsdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zscaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_zszaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bli_zdsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zddaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zdcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_zdzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bli_zcsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zcdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zccaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_zczaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag += ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bli_zzsaxpys( a, x, y ) \
{ \
	(y)      += ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzdaxpys( a, x, y ) \
{ \
	(y)      += ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzcaxpys( a, x, y ) \
{ \
	(y).real += ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_zzzaxpys( a, x, y ) \
{ \
	(y).real += ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag += ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}






#define bli_saxpys( a, x, y ) \
{ \
	bli_sssaxpys( a, x, y ); \
}
#define bli_daxpys( a, x, y ) \
{ \
	bli_dddaxpys( a, x, y ); \
}
#define bli_caxpys( a, x, y ) \
{ \
	bli_cccaxpys( a, x, y ); \
}
#define bli_zaxpys( a, x, y ) \
{ \
	bli_zzzaxpys( a, x, y ); \
}


#endif
