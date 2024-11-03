/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Southern Methodist University

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

#ifndef BLIS_TSETS_H
#define BLIS_TSETS_H

// -- Implementation macros ----------------------------------------------------

#define bli_tsetims( \
          dx, px, xr, xi, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(px,py,tcast)(xr), \
	  PASTEMAC(px,py,tcast)(xi), \
	  yr, \
	  yi \
	); \
}

#define bli_tsetrims( \
              px, xr, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  PASTEMAC(px,py,tcast)(xr), \
	  yi, \
	  yr, \
	  yi \
	); \
}

#define bli_tsetiims( \
              px,     xi, \
          dy, py, yr, yi \
        ) \
{ \
	PASTEMAC(dy,assigns) \
	( \
	  yr, \
	  PASTEMAC(px,py,tcast)(xi), \
	  yr, \
	  yi \
	); \
}

// -- API macros ---------------------------------------------------------------

// -- Consolidated --

// tsets
#define bli_tsets( chx,chy, xr, xi, y ) \
        bli_tsetims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y) \
        )

// tsetrs
#define bli_tsetrs( chx,chy, xr, y ) \
        bli_tsetrims \
        ( \
          PASTEMAC(chx,prec), \
                    xr, \
          PASTEMAC(chy,dom), \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y) \
        )

// tsetis
#define bli_tsetis( chx,chy, xi, y ) \
        bli_tsetiims \
        ( \
          PASTEMAC(chx,prec), \
                    xi, \
          PASTEMAC(chy,dom), \
          PASTEMAC(chy,prec), \
          PASTEMAC(chy,real)(y), \
          PASTEMAC(chy,imag)(y) \
        )

// -- Exposed real/imaginary --

// tsetris
#define bli_tsetris( chx,chy, xr, xi, yr, yi ) \
        bli_tsetims \
        ( \
          PASTEMAC(chx,dom),  \
          PASTEMAC(chx,prec), \
                     xr, \
                     xi, \
          PASTEMAC(chy,dom),  \
          PASTEMAC(chy,prec), \
                     yr, \
                     yi \
        )

// -- Set to constant --

// tset0s
#define bli_tset0s( chy, y ) \
    bli_tsets \
    ( \
      chy,chy, \
      PASTEMAC \
      ( \
        PASTEMAC(chy,prec), \
        zero \
      ), \
      PASTEMAC \
      ( \
        PASTEMAC(chy,prec), \
        zero \
      ), \
      y \
    )

// tset1s
#define bli_tset1s( chy, y ) \
    bli_tsets \
    ( \
      chy,chy, \
      PASTEMAC \
      ( \
       PASTEMAC(chy,prec), \
       one \
      ), \
      PASTEMAC \
      ( \
        PASTEMAC(chy,prec), \
        zero \
      ), \
      y \
    )

// tsetr0s
#define bli_tsetr0s( chy, y ) \
    bli_tsetrs( chy,chy, PASTEMAC(PASTEMAC(chy,prec),zero), y )

// tseti0s
#define bli_tseti0s( chy, y ) \
    bli_tsetis( chy,chy, PASTEMAC(PASTEMAC(chy,prec),zero), y )

// tset0ris
#define bli_tset0ris( chy, yr, yi ) \
    bli_tsetris \
    ( \
      chy,chy, \
      PASTEMAC \
      ( \
       PASTEMAC(chy,prec), \
       zero \
      ), \
      PASTEMAC \
      ( \
        PASTEMAC(chy,prec), \
        zero \
      ), \
      yr, \
      yi \
    )

// -- Micro-tile --

// set0s_mxn
#define bli_tset0s_mxn( chy, m, n, y, rs_y, cs_y ) \
{ \
	for ( dim_t _j = 0; _j < (n); ++_j ) \
	for ( dim_t _i = 0; _i < (m); ++_i ) \
	bli_tset0s( chy, *((y) + _i*(rs_y) + _j*(cs_y)) ); \
}

// set0bbs_mxn
#define bli_tset0bbs_mxn( chy, m, n, y, incy, ldy ) \
{ \
	/* Assume that the duplication factor is the row stride of y. */ \
	const dim_t _d    = incy; \
	const dim_t _ds_y = 1; \
\
	for ( dim_t _j = 0; _j < (n); ++_j ) \
	{ \
		PASTEMAC(chy,ctype)* restrict yj = (PASTEMAC(chy,ctype)*)(y) + _j*(ldy); \
\
		for ( dim_t _i = 0; _i < (m); ++_i ) \
		{ \
			PASTEMAC(chy,ctype)* restrict yij = yj + _i*(incy); \
\
			for ( dim_t _p = 0; _p < _d; ++_p ) \
			{ \
				PASTEMAC(chy,ctype)* restrict yijd = yij + _p*_ds_y; \
\
				bli_tset0s( chy, *yijd ); \
			} \
		} \
	} \
}

// bcastbbs_mxn
#define bli_tbcastbbs_mxn( chy, m, n, y, incy, ldy ) \
{ \
	/* Assume that the duplication factor is the row stride of y. */ \
	const dim_t _d    = incy; \
	const dim_t _ds_y = 1; \
\
	for ( dim_t _j = 0; _j < (n); ++_j ) \
	{ \
		PASTEMAC(chy,ctype)* restrict yj = (PASTEMAC(chy,ctype)*)(y) + _j*(ldy); \
\
		for ( dim_t _i = 0; _i < (m); ++_i ) \
		{ \
			PASTEMAC(chy,ctyper)* restrict yij_r = (PASTEMAC(chy,ctyper)*)( (PASTEMAC(chy,ctype)*)yj + _i*(incy) ); \
			PASTEMAC(chy,ctyper)* restrict yij_i = yij_r + (incy); \
\
			for ( dim_t _p = 1; _p < _d; ++_p ) \
			{ \
				PASTEMAC(chy,ctyper)* restrict yijd_r = yij_r + _p*_ds_y; \
				PASTEMAC(chy,ctyper)* restrict yijd_i = yij_i + _p*_ds_y; (void)yijd_i; \
\
				bli_tcopyris( chy,chy, *yij_r, *yij_i, *yijd_r, *yijd_i ); \
			} \
		} \
	} \
}

#define bli_tset0s_edge( chp, i, m, j, n, p, ldp ) \
{ \
	if ( (i) < (m) ) \
	{ \
		bli_tset0s_mxn \
		( \
		  chp, \
		  m - i, \
		  j, \
		  p + (i)*1, 1, ldp \
		); \
	} \
\
	if ( (j) < (n) ) \
	{ \
		bli_tset0s_mxn \
		( \
		  chp, \
		  m, \
		  n - j, \
		  p + (j)*(ldp), 1, ldp \
		); \
	} \
}

#endif

// -- Notes --------------------------------------------------------------------
