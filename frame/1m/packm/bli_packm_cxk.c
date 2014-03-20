/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

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

#include "blis.h"

#define FUNCPTR_T packm_cxk_fp

typedef void (*FUNCPTR_T)(
                           conj_t  conja,
                           dim_t   n,
                           void*   kappa,
                           void*   a, inc_t inca, inc_t lda,
                           void*   p,             inc_t ldp
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
		BLIS_SPACKM_2XK_KERNEL,
		BLIS_CPACKM_2XK_KERNEL,
		BLIS_DPACKM_2XK_KERNEL,
		BLIS_ZPACKM_2XK_KERNEL,
	},
	/* panel width = 3 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 4 */
	{
		BLIS_SPACKM_4XK_KERNEL,
		BLIS_CPACKM_4XK_KERNEL,
		BLIS_DPACKM_4XK_KERNEL,
		BLIS_ZPACKM_4XK_KERNEL,
	},
	/* panel width = 5 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 6 */
	{
		BLIS_SPACKM_6XK_KERNEL,
		BLIS_CPACKM_6XK_KERNEL,
		BLIS_DPACKM_6XK_KERNEL,
		BLIS_ZPACKM_6XK_KERNEL,
	},
	/* panel width = 7 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 8 */
	{
		BLIS_SPACKM_8XK_KERNEL,
		BLIS_CPACKM_8XK_KERNEL,
		BLIS_DPACKM_8XK_KERNEL,
		BLIS_ZPACKM_8XK_KERNEL,
	},
	/* panel width = 9 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 10 */
	{
		BLIS_SPACKM_10XK_KERNEL,
		BLIS_CPACKM_10XK_KERNEL,
		BLIS_DPACKM_10XK_KERNEL,
		BLIS_ZPACKM_10XK_KERNEL,
	},
	/* panel width = 11 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 12 */
	{
		BLIS_SPACKM_12XK_KERNEL,
		BLIS_CPACKM_12XK_KERNEL,
		BLIS_DPACKM_12XK_KERNEL,
		BLIS_ZPACKM_12XK_KERNEL,
	},
	/* panel width = 13 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 14 */
	{
		BLIS_SPACKM_14XK_KERNEL,
		BLIS_CPACKM_14XK_KERNEL,
		BLIS_DPACKM_14XK_KERNEL,
		BLIS_ZPACKM_14XK_KERNEL,
	},
	/* panel width = 15 */
	{
		NULL, NULL, NULL, NULL,
	},
	/* panel width = 16 */
	{
		BLIS_SPACKM_16XK_KERNEL,
		BLIS_CPACKM_16XK_KERNEL,
		BLIS_DPACKM_16XK_KERNEL,
		BLIS_ZPACKM_16XK_KERNEL,
	},
	/* panel width = 17 */
	{
		NULL, NULL, NULL, NULL,
	},
};



#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname, scal2vker ) \
\
void PASTEMAC(ch,varname)( \
                           conj_t  conja, \
                           dim_t   m, \
                           dim_t   n, \
                           void*   kappa, \
                           void*   a, inc_t inca, inc_t lda, \
                           void*   p,             inc_t ldp  \
                         ) \
{ \
	dim_t     panel_dim; \
	num_t     dt; \
	FUNCPTR_T f; \
\
	/* If the panel dimension is unit, then we recognize that this allows
	   the kernel to reduce to a copyv, so we call that kernel directly. */ \
	if ( m == 1 ) \
	{ \
		PASTEMAC3(ch,ch,ch,scal2vker)( conja, \
		                               n, \
		                               kappa, \
		                               a, lda, \
		                               p, ldp ); \
		return; \
	} \
\
	/* The panel dimension is always equal to the m dimension of p. */ \
	panel_dim = m; \
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
	   provided, we invoke the implementation. Otherwise, we use scal2m. */ \
	if ( f != NULL ) \
	{ \
		f( conja, \
		   n, \
		   kappa, \
		   a, inca, lda, \
		   p,       ldp ); \
	} \
	else \
	{ \
		/* Treat the panel as m x n and column-stored (unit row stride). */ \
		PASTEMAC3(ch,ch,ch,scal2m)( 0, \
		                            BLIS_NONUNIT_DIAG, \
		                            BLIS_DENSE, \
		                            conja, \
		                            m, \
		                            n, \
		                            kappa, \
		                            a, inca, lda, \
		                            p, 1,    ldp ); \
	} \
}

INSERT_GENTFUNC_BASIC( packm_cxk, SCAL2V_KERNEL )

