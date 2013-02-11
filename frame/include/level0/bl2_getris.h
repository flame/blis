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

#ifndef BLIS_GETRIS_H
#define BLIS_GETRIS_H

// getris

// Notes:
// - The first char encodes the type of x.
// - The second char encodes the type of y.

#define bl2_ssgetris( x, yr, yi ) \
{ \
	(yr)     = ( float  ) (x); \
	(yi)     = 0.0F; \
}
#define bl2_dsgetris( x, yr, yi ) \
{ \
	(yr)     = ( float  ) (x); \
	(yi)     = 0.0F; \
}
#define bl2_csgetris( x, yr, yi ) \
{ \
	(yr)     = ( float  ) (x).real; \
	(yi)     = ( float  ) (x).imag; \
}
#define bl2_zsgetris( x, yr, yi ) \
{ \
	(yr)     = ( float  ) (x).real; \
	(yi)     = ( float  ) (x).imag; \
}


#define bl2_sdgetris( x, yr, yi ) \
{ \
	(yr)     = ( double ) (x); \
	(yi)     = 0.0; \
}
#define bl2_ddgetris( x, yr, yi ) \
{ \
	(yr)     = ( double ) (x); \
	(yi)     = 0.0; \
}
#define bl2_cdgetris( x, yr, yi ) \
{ \
	(yr)     = ( double ) (x).real; \
	(yi)     = ( double ) (x).imag; \
}
#define bl2_zdgetris( x, yr, yi ) \
{ \
	(yr)     = ( double ) (x).real; \
	(yi)     = ( double ) (x).imag; \
}



#define bl2_sgetris( x, yr, yi ) \
{ \
	bl2_ssgetris( x, yr, yi ); \
}
#define bl2_dgetris( x, yr, yi ) \
{ \
	bl2_ddgetris( x, yr, yi ); \
}
#define bl2_cgetris( x, yr, yi ) \
{ \
	bl2_csgetris( x, yr, yi ); \
}
#define bl2_zgetris( x, yr, yi ) \
{ \
	bl2_zdgetris( x, yr, yi ); \
}


#endif
