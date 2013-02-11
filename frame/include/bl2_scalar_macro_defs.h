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

#ifndef BLIS_SCALAR_MACRO_DEFS_H
#define BLIS_SCALAR_MACRO_DEFS_H


// -- Scalar query macros --

#include "bl2_eq.h"


// -- Scalar constant initialization macros --

#include "bl2_constants.h"


// -- Scalar computation macros --

#include "bl2_absq2s.h"
#include "bl2_abval2s.h"

#include "bl2_adds.h"
#include "bl2_addjs.h"

#include "bl2_axpys.h"
#include "bl2_axpyjs.h"

#include "bl2_axmys.h"

#include "bl2_cast.h"

#include "bl2_conjs.h"

#include "bl2_copys.h"
#include "bl2_copyjs.h"
#include "bl2_copycjs.h"

#include "bl2_copynzs.h"
#include "bl2_copynzjs.h"

#include "bl2_dots.h"
#include "bl2_dotjs.h"

#include "bl2_getris.h"

#include "bl2_grabis.h"

#include "bl2_inverts.h"

#include "bl2_invscals.h"
#include "bl2_invscaljs.h"
#include "bl2_invscalcjs.h"

#include "bl2_neg2s.h"

#include "bl2_projrs.h"

#include "bl2_scals.h"
#include "bl2_scaljs.h"
#include "bl2_scalcjs.h"

#include "bl2_scal2s.h"
#include "bl2_scal2js.h"

#include "bl2_setris.h"

#include "bl2_sqrt2s.h"

#include "bl2_subs.h"
#include "bl2_subjs.h"

#include "bl2_xpbys.h"



// -- Scalar initialization macros --

#include "bl2_rands.h"


// -- Misc. scalar macros --

#include "bl2_fprints.h"


// -- Inlined scalar macros in loops --

#include "bl2_adds_mxn.h"
#include "bl2_adds_mxn_uplo.h"
#include "bl2_copys_mxn.h"
#include "bl2_set0_mxn.h"
#include "bl2_xpbys_mxn.h"
#include "bl2_xpbys_mxn_uplo.h"


// -- Miscellaneous macros --

// min, max, abs

#define bl2_min( a, b ) ( (a) < (b) ? (a) : (b) )
#define bl2_max( a, b ) ( (a) > (b) ? (a) : (b) )
#define bl2_abs( a )    ( (a) < 0 ? -(a) : (a) )

// fmin, fmax, fabs

#define bl2_min( a, b ) ( (a) < (b) ? (a) : (b) )
#define bl2_fmin( a, b ) bl2_min( a, b )
#define bl2_fmax( a, b ) bl2_max( a, b )
#define bl2_fabs( a )    ( (a) < 0.0 ? -(a) : (a) )

// fminabs, fmaxabs
#define bl2_fminabs( a, b ) \
\
	bl2_fmin( bl2_fabs( a ), \
	          bl2_fabs( b ) )

#define bl2_fmaxabs( a, b ) \
\
	bl2_fmax( bl2_fabs( a ), \
	          bl2_fabs( b ) )

// swap_types

#define bl2_swap_types( type1, type2 ) \
{ \
	num_t temp = type1; \
	type1 = type2; \
	type2 = temp; \
}

// swap_dims

#define bl2_swap_dims( dim1, dim2 ) \
{ \
	dim_t temp = dim1; \
	dim1 = dim2; \
	dim2 = temp; \
}

// swap_incs

#define bl2_swap_incs( inc1, inc2 ) \
{ \
	inc_t temp = inc1; \
	inc1 = inc2; \
	inc2 = temp; \
}

// toggle_bool

#define bl2_toggle_bool( b ) \
{ \
	if ( b == TRUE ) b = FALSE; \
	else             b = TRUE; \
}


#endif
