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

#include <unistd.h>
#include "blis2.h"

//           uplo   m     alpha    x        incx  y        incy  a        lda
void dsyr2_( char*, int*, double*, double*, int*, double*, int*, double*, int* );

int main( int argc, char** argv )
{
	obj_t a, x, y;
	obj_t a_save;
	obj_t alpha;
	dim_t m;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input;
	num_t dt_a, dt_x, dt_y;
	num_t dt_alpha;
	int   r, n_repeats;

	double dtime;
	double dtime_save;
	double gflops;

	bl2_init();

	n_repeats = 3;

#ifndef PRINT
	p_begin = 40;
	p_end   = 2000;
	p_inc   = 40;

	m_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 6;
#endif

#if 1
	dt_alpha = BLIS_DOUBLE;
	dt_x = BLIS_DOUBLE;
	dt_y = BLIS_DOUBLE;
	dt_a = BLIS_DOUBLE;
#else
	dt_alpha = BLIS_DCOMPLEX;
	dt_x = BLIS_DCOMPLEX;
	dt_y = BLIS_DCOMPLEX;
	dt_a = BLIS_DCOMPLEX;
#endif


	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;


		bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

		bl2_obj_create( dt_x, m, 1, 0, 0, &x );
		bl2_obj_create( dt_y, m, 1, 0, 0, &y );
		bl2_obj_create( dt_a, m, m, 0, 0, &a );
		bl2_obj_create( dt_a, m, m, 0, 0, &a_save );

		bl2_randm( &x );
		bl2_randm( &y );
		bl2_randm( &a );

		bl2_obj_set_struc( BLIS_HERMITIAN, a );
		//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
		bl2_obj_set_uplo( BLIS_LOWER, a );
		//bl2_obj_set_uplo( BLIS_UPPER, a );


		bl2_setsc(  (2.0/1.0), 0.0, &alpha );


		bl2_copym( &a, &a_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bl2_copym( &a_save, &a );


			dtime = bl2_clock();

#ifdef PRINT
			bl2_printm( "x", &x, "%4.1f", "" );
			bl2_printm( "y", &y, "%4.1f", "" );
			bl2_printm( "a", &a, "%4.1f", "" );
#endif 

#ifdef BLIS

			//bl2_obj_toggle_conj( x );
			//bl2_obj_toggle_conj( y );

#if 1
			bl2_her2( &alpha,
#else
			bl2_syr2( &alpha,
#endif
			          &x,
			          &y,
			          &a );
/*
			bl2_her2_unb_var1( BLIS_NO_CONJUGATE,
			                   &BLIS_TWO,
			                   &BLIS_TWO,
			                   &x,
			                   &y,
			                   &a, NULL );
*/


#else

			char    uplo   = 'L';
			int     mm     = bl2_obj_length( a );
			int     incx   = bl2_obj_vector_inc( x );
			int     incy   = bl2_obj_vector_inc( y );
			int     lda    = bl2_obj_col_stride( a );
			double* alphap = bl2_obj_buffer( alpha );
			double* xp     = bl2_obj_buffer( x );
			double* yp     = bl2_obj_buffer( y );
			double* ap     = bl2_obj_buffer( a );

			dsyr2_( &uplo,
			        &mm,
			        alphap,
			        xp, &incx,
			        yp, &incy,
			        ap, &lda );
#endif

#ifdef PRINT
			bl2_printm( "a after", &a, "%4.1f", "" );
			exit(1);
#endif

			dtime_save = bl2_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * m * m ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_her2_blis" );
#else
		printf( "data_her2_%s", BLAS );
#endif
		printf( "( %2ld, 1:3 ) = [ %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, dtime_save, gflops );

		bl2_obj_free( &alpha );

		bl2_obj_free( &x );
		bl2_obj_free( &y );
		bl2_obj_free( &a );
		bl2_obj_free( &a_save );
	}

	bl2_finalize();

	return 0;
}

