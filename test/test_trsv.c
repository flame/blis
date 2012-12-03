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

//           uploa  trans, diag,  m     a        lda   x        incx
void dtrsv_( char*, char*, char*, int*, double*, int*, double*, int* );

int main( int argc, char** argv )
{
	obj_t a, x;
	obj_t x_save;
	obj_t alpha;
	dim_t m;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input;
	num_t dt_a, dt_x;
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

	m_input = 15;
	n_input = 15;
#endif

	dt_alpha = BLIS_DOUBLE;
	dt_a = BLIS_DOUBLE;
	dt_x = BLIS_DOUBLE;

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;


		bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

		bl2_obj_create( dt_a, m, m, 0, 0, &a );
		bl2_obj_create( dt_x, m, 1, 0, 0, &x );
		bl2_obj_create( dt_x, m, 1, 0, 0, &x_save );

		bl2_randm( &a );
		bl2_randm( &x );

		bl2_obj_set_struc( BLIS_TRIANGULAR, a );
		bl2_obj_set_uplo( BLIS_LOWER, a );
		bl2_obj_set_trans( BLIS_NO_TRANSPOSE, a );
		bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );

		bl2_sets(  (2.0/1.0), &alpha );


		bl2_copym( &x, &x_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bl2_copym( &x_save, &x );


			dtime = bl2_clock();

#ifdef PRINT
			bl2_printm( "a", &a, "%4.1f", "" );
			bl2_printm( "x", &x, "%4.1f", "" );
#endif

#ifdef BLIS

			bl2_trsv( &BLIS_ONE,
			          &a,
			          &x );

#else

			char    uploa  = 'L';
			char    transa = 'N';
			char    diaga  = 'N';
			int     mm     = bl2_obj_length( a );
			int     lda    = bl2_obj_col_stride( a );
			int     incx   = bl2_obj_vector_inc( x );
			double* ap     = bl2_obj_buffer( a );
			double* xp     = bl2_obj_buffer( x );

			dtrsv_( &uploa,
			        &transa,
			        &diaga,
			        &mm,
			        ap, &lda,
			        xp, &incx );
#endif

#ifdef PRINT
			bl2_printm( "x after", &x, "%4.1f", "" );
			exit(1);
#endif


			dtime = bl2_clock() - dtime;

			dtime_save = bl2_min( dtime, dtime_save );

		}

		gflops = ( 1.0 * m * m ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_trsv_blis" );
#else
		printf( "data_trsv_%s", BLAS );
#endif
		printf( "( %2ld, 1:3 ) = [ %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, dtime_save, gflops );

		bl2_obj_free( &alpha );

		bl2_obj_free( &a );
		bl2_obj_free( &x );
		bl2_obj_free( &x_save );
	}

	bl2_finalize();

	return 0;
}

