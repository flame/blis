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

#ifndef BLIS_AXMYS_H
#define BLIS_AXMYS_H

// axmys

// Notes:
// - The first char encodes the type of a.
// - The second char encodes the type of x.
// - The third char encodes the type of y.

// -- (axy) = (ss?) ------------------------------------------------------------

#define bli_sssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_ssdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a)      * ( float  ) (x)      ); \
}
#define bli_sszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a)      * ( float  ) (x)      ); \
}

// -- (axy) = (sd?) ------------------------------------------------------------

#define bli_sdsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_sdzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (sc?) ------------------------------------------------------------

#define bli_scsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_scdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
}
#define bli_sccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag -= ( float  )( ( float  ) (a)      * ( float  ) (x).imag ); \
}
#define bli_sczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a)      * ( float  ) (x).real ); \
	(y).imag -= ( double )( ( float  ) (a)      * ( float  ) (x).imag ); \
}

// -- (axy) = (sz?) ------------------------------------------------------------

#define bli_szsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_szcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bli_szzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (ds?) ------------------------------------------------------------

#define bli_dssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dsdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dd?) ------------------------------------------------------------

#define bli_ddsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_dddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x)      ); \
}
#define bli_ddzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x)      ); \
}

// -- (axy) = (dc?) ------------------------------------------------------------

#define bli_dcsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dcdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bli_dczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (dz?) ------------------------------------------------------------

#define bli_dzsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
}
#define bli_dzcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( float  )( ( double ) (a)      * ( double ) (x).imag ); \
}
#define bli_dzzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a)      * ( double ) (x).real ); \
	(y).imag -= ( double )( ( double ) (a)      * ( double ) (x).imag ); \
}

// -- (axy) = (cs?) ------------------------------------------------------------

#define bli_cssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_csdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_cscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_cszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (cd?) ------------------------------------------------------------

#define bli_cdsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_cdcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_cdzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (cc?) ------------------------------------------------------------

#define bli_ccsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_ccdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_cccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_cczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (cz?) ------------------------------------------------------------

#define bli_czsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_czcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_czzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}

// -- (axy) = (zs?) ------------------------------------------------------------

#define bli_zssaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zsdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
}
#define bli_zscaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x)      ); \
}
#define bli_zszaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x)      ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x)      ); \
}

// -- (axy) = (zd?) ------------------------------------------------------------

#define bli_zdsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zddaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
}
#define bli_zdcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x)      ); \
}
#define bli_zdzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x)      ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x)      ); \
}

// -- (axy) = (zc?) ------------------------------------------------------------

#define bli_zcsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zcdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
}
#define bli_zccaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( float  )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}
#define bli_zczaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( float  ) (a).real * ( float  ) (x).real - ( float  ) (a).imag * ( float  ) (x).imag ); \
	(y).imag -= ( double )( ( float  ) (a).imag * ( float  ) (x).real + ( float  ) (a).real * ( float  ) (x).imag ); \
}

// -- (axy) = (zz?) ------------------------------------------------------------

#define bli_zzsaxmys( a, x, y ) \
{ \
	(y)      -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzdaxmys( a, x, y ) \
{ \
	(y)      -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
}
#define bli_zzcaxmys( a, x, y ) \
{ \
	(y).real -= ( float  )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( float  )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}
#define bli_zzzaxmys( a, x, y ) \
{ \
	(y).real -= ( double )( ( double ) (a).real * ( double ) (x).real - ( double ) (a).imag * ( double ) (x).imag ); \
	(y).imag -= ( double )( ( double ) (a).imag * ( double ) (x).real + ( double ) (a).real * ( double ) (x).imag ); \
}






#define bli_saxmys( a, x, y ) \
{ \
	bli_sssaxmys( a, x, y ); \
}
#define bli_daxmys( a, x, y ) \
{ \
	bli_dddaxmys( a, x, y ); \
}
#define bli_caxmys( a, x, y ) \
{ \
	bli_cccaxmys( a, x, y ); \
}
#define bli_zaxmys( a, x, y ) \
{ \
	bli_zzzaxmys( a, x, y ); \
}


#endif
