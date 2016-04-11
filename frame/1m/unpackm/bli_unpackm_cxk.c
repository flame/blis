/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

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

#include "blis.h"

#define FUNCPTR_T unpackm_cxk_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conjp,
                           dim_t   n,
                           void*   beta,
                           void*   p,
                           void*   a, inc_t inca, inc_t lda
                         );

#undef  FUNCPTR_ARRAY_LENGTH
#define FUNCPTR_ARRAY_LENGTH 18

static FUNCPTR_T ftypes[FUNCPTR_ARRAY_LENGTH][BLIS_NUM_FP_TYPES] =
{
	/* panel width = 0 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 1 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 2 */
	{
		BLIS_SUNPACKM_2XK_KERNEL,
		BLIS_CUNPACKM_2XK_KERNEL,
		BLIS_DUNPACKM_2XK_KERNEL,
		BLIS_ZUNPACKM_2XK_KERNEL,
	},
	/* panel width = 3 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 4 */
	{
		BLIS_SUNPACKM_4XK_KERNEL,
		BLIS_CUNPACKM_4XK_KERNEL,
		BLIS_DUNPACKM_4XK_KERNEL,
		BLIS_ZUNPACKM_4XK_KERNEL,
	},
	/* panel width = 5 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 6 */
	{
		BLIS_SUNPACKM_6XK_KERNEL,
		BLIS_CUNPACKM_6XK_KERNEL,
		BLIS_DUNPACKM_6XK_KERNEL,
		BLIS_ZUNPACKM_6XK_KERNEL,
	},
	/* panel width = 7 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 8 */
	{
		BLIS_SUNPACKM_8XK_KERNEL,
		BLIS_CUNPACKM_8XK_KERNEL,
		BLIS_DUNPACKM_8XK_KERNEL,
		BLIS_ZUNPACKM_8XK_KERNEL,
	},
	/* panel width = 9 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 10 */
	{
		BLIS_SUNPACKM_10XK_KERNEL,
		BLIS_CUNPACKM_10XK_KERNEL,
		BLIS_DUNPACKM_10XK_KERNEL,
		BLIS_ZUNPACKM_10XK_KERNEL,
	},
	/* panel width = 11 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 12 */
	{
		BLIS_SUNPACKM_12XK_KERNEL,
		BLIS_CUNPACKM_12XK_KERNEL,
		BLIS_DUNPACKM_12XK_KERNEL,
		BLIS_ZUNPACKM_12XK_KERNEL,
	},
	/* panel width = 13 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 14 */
	{
		BLIS_SUNPACKM_14XK_KERNEL,
		BLIS_CUNPACKM_14XK_KERNEL,
		BLIS_DUNPACKM_14XK_KERNEL,
		BLIS_ZUNPACKM_14XK_KERNEL,
	},
	/* panel width = 15 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 16 */
	{
		BLIS_SUNPACKM_16XK_KERNEL,
		BLIS_CUNPACKM_16XK_KERNEL,
		BLIS_DUNPACKM_16XK_KERNEL,
		BLIS_ZUNPACKM_16XK_KERNEL,
	},
	/* panel width = 17 */
	{
		NULL, NULL, NULL, NULL,
	},
};



#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname)( \
                          conj_t  conjp, \
                          dim_t   m, \
                          dim_t   n, \
                          void*   beta, \
                          void*   p,             inc_t ldp, \
                          void*   a, inc_t inca, inc_t lda, \
                          cntx_t* cntx  \
                        ) \
{ \
	dim_t     panel_dim; \
	num_t     dt; \
	FUNCPTR_T f; \
\
	/* If the panel dimension is unit, then we recognize that this allows
	   the kernel to reduce to a copyv, so we call that directly. */ \
	if ( m == 1 ) \
	{ \
		PASTEMAC(ch,copyv) \
		( \
		  conjp, \
		  n, \
		  p, 1, \
		  a, lda, \
		  cntx  \
		); \
		return; \
	} \
\
	/* The panel dimension is always equal to the leading dimension of p. */ \
	panel_dim = ldp; \
\
	/* Acquire the datatype for the current function. */ \
	dt = PASTEMAC(ch,type); \
\
	/* Index into the array to extract the correct function pointer.
	   If the panel dimension is too big to be within the array of
	   explicitly handled kernels, then we treat that kernel the same
	   as if it were in range but unimplemented. */ \
	if ( panel_dim < FUNCPTR_ARRAY_LENGTH ) f = ftypes[panel_dim][dt]; \
	else                                    f = NULL; \
\
	/* If there exists a kernel implementation for the panel dimension
	   provided, and the "width" of the panel is equal to the leading
	   dimension, we invoke the implementation. Otherwise, we use scal2m.
	   By using scal2m to handle edge cases (where m < panel_dim), we
	   allow the kernel implementations to remain very simple. */ \
	if ( f != NULL && m == panel_dim ) \
	{ \
		f \
		( \
		  conjp, \
		  n, \
		  beta, \
		  p, \
		  a, inca, lda  \
		); \
	} \
	else \
	{ \
		/* Treat the panel as m x n and column-stored (unit row stride). */ \
		PASTEMAC(ch,scal2m) \
		( \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  conjp, \
		  m, \
		  n, \
		  beta, \
		  p, 1,    ldp, \
		  a, inca, lda, \
		  cntx  \
		); \
	} \
}

INSERT_GENTFUNC_BASIC0( unpackm_cxk )

