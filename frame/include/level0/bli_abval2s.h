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

#ifndef BLIS_ABVAL2S_H
#define BLIS_ABVAL2S_H

// abval2s

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of a.


#define bli_ssabval2s( x, a ) \
{ \
	bli_sssetris( fabsf(x), 0.0, (a) ); \
}
#define bli_dsabval2s( x, a ) \
{ \
	bli_dssetris( fabs(x),  0.0, (a) ); \
}

#define bli_sdabval2s( x, a ) \
{ \
	bli_sdsetris( fabsf(x), 0.0, (a) ); \
}
#define bli_ddabval2s( x, a ) \
{ \
	bli_ddsetris( fabs(x),  0.0, (a) ); \
}

#define bli_scabval2s( x, a ) \
{ \
	bli_scsetris( fabsf(x), 0.0, (a) ); \
}
#define bli_dcabval2s( x, a ) \
{ \
	bli_dcsetris( fabs(x),  0.0, (a) ); \
}

#define bli_szabval2s( x, a ) \
{ \
	bli_szsetris( fabsf(x), 0.0, (a) ); \
}
#define bli_dzabval2s( x, a ) \
{ \
	bli_dzsetris( fabs(x),  0.0, (a) ); \
}


#ifndef BLIS_ENABLE_C99_COMPLEX


#define bli_csabval2s( x, a ) \
{ \
	float  s   = bli_fmaxabs( bli_creal(x), bli_cimag(x) ); \
	float  mag = sqrtf( s ) * \
	             sqrtf( ( bli_creal(x) / s ) * bli_creal(x) + \
	                    ( bli_cimag(x) / s ) * bli_cimag(x) ); \
	bli_sssetris( mag, 0.0, a ); \
}
#define bli_zsabval2s( x, a ) \
{ \
	double s   = bli_fmaxabs( bli_zreal(x), bli_zimag(x) ); \
	double mag = sqrt( s ) * \
	             sqrt( ( bli_zreal(x) / s ) * bli_zreal(x) + \
	                   ( bli_zimag(x) / s ) * bli_zimag(x) ); \
	bli_dssetris( mag, 0.0, a ); \
}

#define bli_cdabval2s( x, a ) \
{ \
	double s   = bli_fmaxabs( bli_creal(x), bli_cimag(x) ); \
	double mag = sqrt( s ) * \
	             sqrt( ( bli_creal(x) / s ) * bli_creal(x) + \
	                   ( bli_cimag(x) / s ) * bli_cimag(x) ); \
	bli_ddsetris( mag, 0.0, a ); \
}
#define bli_zdabval2s( x, a ) \
{ \
	double s   = bli_fmaxabs( bli_zreal(x), bli_zimag(x) ); \
	double mag = sqrt( s ) * \
	             sqrt( ( bli_zreal(x) / s ) * bli_zreal(x) + \
	                   ( bli_zimag(x) / s ) * bli_zimag(x) ); \
	bli_ddsetris( mag, 0.0, a ); \
}

#define bli_ccabval2s( x, a ) \
{ \
	float  s   = bli_fmaxabs( bli_creal(x), bli_cimag(x) ); \
	float  mag = sqrtf( s ) * \
	             sqrtf( ( bli_creal(x) / s ) * bli_creal(x) + \
	                    ( bli_cimag(x) / s ) * bli_cimag(x) ); \
	bli_scsetris( mag, 0.0, a ); \
}
#define bli_zcabval2s( x, a ) \
{ \
	double s   = bli_fmaxabs( bli_zreal(x), bli_zimag(x) ); \
	double mag = sqrt( s ) * \
	             sqrt( ( bli_zreal(x) / s ) * bli_zreal(x) + \
	                   ( bli_zimag(x) / s ) * bli_zimag(x) ); \
	bli_dcsetris( mag, 0.0, a ); \
}

#define bli_czabval2s( x, a ) \
{ \
	double s   = bli_fmaxabs( bli_creal(x), bli_cimag(x) ); \
	double mag = sqrt( s ) * \
	             sqrt( ( bli_creal(x) / s ) * bli_creal(x) + \
	                   ( bli_cimag(x) / s ) * bli_cimag(x) ); \
	bli_dzsetris( mag, 0.0, a ); \
}
#define bli_zzabval2s( x, a ) \
{ \
	double s   = bli_fmaxabs( bli_zreal(x), bli_zimag(x) ); \
	double mag = sqrt( s ) * \
	             sqrt( ( bli_zreal(x) / s ) * bli_zreal(x) + \
	                   ( bli_zimag(x) / s ) * bli_zimag(x) ); \
	bli_dzsetris( mag, 0.0, a ); \
}


#else // ifdef BLIS_ENABLE_C99_COMPLEX


#define bli_csabval2s( x, a )  { (a) = cabsf(x); }
#define bli_zsabval2s( x, a )  { (a) = cabs(x);  }

#define bli_cdabval2s( x, a )  { (a) = cabsf(x); }
#define bli_zdabval2s( x, a )  { (a) = cabs(x);  }

#define bli_ccabval2s( x, a )  { (a) = cabsf(x); }
#define bli_zcabval2s( x, a )  { (a) = cabs(x);  }

#define bli_czabval2s( x, a )  { (a) = cabsf(x); }
#define bli_zzabval2s( x, a )  { (a) = cabs(x);  }



#endif // BLIS_ENABLE_C99_COMPLEX


#define bli_sabval2s( x, a )  bli_ssabval2s( x, a )
#define bli_dabval2s( x, a )  bli_ddabval2s( x, a )
#define bli_cabval2s( x, a )  bli_ccabval2s( x, a )
#define bli_zabval2s( x, a )  bli_zzabval2s( x, a )


#endif
