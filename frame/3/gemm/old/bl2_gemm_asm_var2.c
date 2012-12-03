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

#include "blis2.h"

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           dim_t   n,
                           dim_t   k,
                           void*   a, inc_t rs_a, inc_t cs_a,
                           void*   b, inc_t rs_b, inc_t cs_b,
                           void*   c, inc_t rs_c, inc_t cs_c
                         );

static FUNCPTR_T ftypes[BLIS_NUM_FP_TYPES] =
{
	bl2_sgemm_asm_var2,
	bl2_cgemm_asm_var2,
	bl2_dgemm_asm_var2,
	bl2_zgemm_asm_var2
};

void bl2_gemm_asm_var2( obj_t*  alpha,
                        obj_t*  a,
                        obj_t*  b,
                        obj_t*  beta,
                        obj_t*  c,
                        gemm_t* cntl )
{
	num_t     dt_exec   = bl2_obj_execution_datatype( *c );
	num_t     dt_a      = bl2_obj_datatype( *a );
	num_t     dt_b      = bl2_obj_datatype( *b );

	dim_t     m         = bl2_obj_length( *c );
	dim_t     n         = bl2_obj_width( *c );
	dim_t     k         = bl2_obj_width( *a );

	void*     buf_a     = bl2_obj_buffer_at_off( *a );
	inc_t     rs_a      = bl2_obj_row_stride( *a );
	inc_t     cs_a      = bl2_obj_col_stride( *a );

	void*     buf_b     = bl2_obj_buffer_at_off( *b );
	inc_t     rs_b      = bl2_obj_row_stride( *b );
	inc_t     cs_b      = bl2_obj_col_stride( *b );

	void*     buf_c     = bl2_obj_buffer_at_off( *c );
	inc_t     rs_c      = bl2_obj_row_stride( *c );
	inc_t     cs_c      = bl2_obj_col_stride( *c );

	FUNCPTR_T f;

	// Handle the special case where c and a are complex and b is real.
	// Note that this is the ONLY case allowed by the inner kernel whereby
	// the datatypes of a and b differ. In this situation, the execution
	// datatype is real, so we need to inflate the m and leading dimensions
	// by a factor of two.
/*
	if ( bl2_is_complex( dt_a ) && bl2_is_real( dt_b ) )
	{
		m    *= 2;
		cs_c *= 2;
		cs_a *= 2;
	}
*/

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( m,
	   n,
	   k,
	   buf_a, rs_a, cs_a,
	   buf_b, rs_b, cs_b,
	   buf_c, rs_c, cs_c );
}

void PASTEMAC(s,gemm_asm_var2)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{

}

void PASTEMAC(c,gemm_asm_var2)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{

}

void PASTEMAC(z,gemm_asm_var2)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{

}

#include "pmmintrin.h"

typedef union
{
    __m128d v;
    double  d[2];
} v2df_t;

#define NOSSE 0

void PASTEMAC(d,gemm_asm_var2)(
                                dim_t   m,
                                dim_t   n,
                                dim_t   k,
                                void*   a, inc_t rs_a, inc_t cs_a,
                                void*   b, inc_t rs_b, inc_t cs_b,
                                void*   c, inc_t rs_c, inc_t cs_c
                               )
{
	double* restrict a_cast = a;
	double* restrict b_cast = b;
	double* restrict c_cast = c;
	double* restrict a1;
	double* restrict b1;
	double* restrict c1;
	double* restrict alpha11;
	double* restrict beta11;
	double* restrict gamma11;
	dim_t   i, j, h;
	v2df_t  b1v;;
	v2df_t  a1v, a2v;;
	v2df_t  c1v, c2v;

	dim_t   m_iter = m / 2;
	dim_t   m_left = m % 2;

	inc_t   step_a = 2*rs_a;
	inc_t   step_c = 2*rs_c;

	for ( j = 0; j < n; ++j )
	{
		c1 = c_cast + (j  )* cs_c;
		b1 = b_cast + (j  )* cs_b;
		
		for ( h = 0; h < k; ++h )
		{
			a1      = a_cast + (h  )*cs_a;
			beta11  = b1     + (h  )*rs_b;

#if NOSSE
#else
			b1v.v = _mm_loaddup_pd( beta11 );
#endif

			alpha11 = a1;
			gamma11 = c1;

			for ( i = 0; i < m_iter; ++i )
			{
#if NOSSE
				*(gamma11  ) += *beta11 * *(alpha11  );
				*(gamma11+1) += *beta11 * *(alpha11+1);
				*(gamma11+2) += *beta11 * *(alpha11+2);
				*(gamma11+3) += *beta11 * *(alpha11+3);
#else
				a1v.v = _mm_load_pd( alpha11 );
				//a2v.v = _mm_load_pd( alpha11+2 );
				c1v.v = _mm_load_pd( gamma11 );
				//c2v.v = _mm_load_pd( gamma11+2 );

				c1v.v += b1v.v * a1v.v;
				//c2v.v += b1v.v * a2v.v;

				_mm_store_pd( gamma11,   c1v.v );
				//_mm_store_pd( gamma11+2, c2v.v );
#endif

				alpha11 += step_a;
				gamma11 += step_c;
			}

			for ( i = 0; i < m_left; ++i )
			{
				*(gamma11  ) += *beta11 * *(alpha11  );

				alpha11 += rs_a;
				gamma11 += rs_c;
			}
		}
	}
}
