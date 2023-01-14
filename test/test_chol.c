/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#ifdef WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "blis.h"

void libblis_test_ceil_pow2( obj_t* alpha )
{
    double alpha_r;
    double alpha_i;

    bli_getsc( alpha, &alpha_r, &alpha_i );

    alpha_r = pow( 2.0, ceil( log2( alpha_r ) ) );

    bli_setsc( alpha_r, alpha_i, alpha );
}

void libblis_test_mobj_randomize( bool normalize, obj_t* a )
{
    bli_randm( a );

    if ( normalize )
    {
        num_t dt   = bli_obj_dt( a );
        num_t dt_r = bli_obj_dt_proj_to_real( a );
        obj_t kappa;
        obj_t kappa_r;

        bli_obj_scalar_init_detached( dt,   &kappa );
        bli_obj_scalar_init_detached( dt_r, &kappa_r );

        // Normalize matrix elements.
        bli_norm1m( a, &kappa_r );
        libblis_test_ceil_pow2( &kappa_r );

        bli_copysc( &kappa_r, &kappa );
        bli_invscalm( &kappa, a );
    }
}

int main( int argc, char** argv )
{
	obj_t a;
	obj_t a_save;
	dim_t m;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input;
	num_t dt;
	int   r, n_repeats;
	uplo_t       uploa;

	double dtime;
	double dtime_save;
	double gflops;

	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 3;

	p_begin = 32;
	p_end   = 4800;
	p_inc   = 32;

	m_input = -1;
	//m_input = 15;

	uploa = BLIS_LOWER;

#if 1
	//dt = BLIS_FLOAT;
	dt = BLIS_DOUBLE;
#else
	//dt = BLIS_SCOMPLEX;
	dt = BLIS_DCOMPLEX;
#endif

	//g_my_kc = 160;

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_end; p += p_inc ) ;
#ifdef BLIS
	printf( "data_chol_blis" );
#else
	printf( "data_chol_%s", BLAS );
#endif
	printf( "( %3llu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0, ( unsigned long )0, 0.0 );

	for ( p = p_end; p_begin <= p; p -= p_inc )
	{
		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		//m = ( dim_t ) p;

		bli_obj_create( dt, m, m, 0, 0, &a );
		bli_obj_create( dt, m, m, 0, 0, &a_save );

		// Set the structure and uplo properties of A and A_save.
		bli_obj_set_struc( BLIS_HERMITIAN, &a );
		bli_obj_set_uplo( uploa, &a );

		bli_obj_set_struc( BLIS_HERMITIAN, &a_save );
		bli_obj_set_uplo( uploa, &a_save );

		// Randomize A, load the diagonal.
		libblis_test_mobj_randomize( TRUE, &a );
		//libblis_test_mobj_load_diag( params, &a );
		//bli_randm( &a );

		bli_shiftd( &BLIS_TWO, &a );
		bli_setid( &BLIS_ZERO, &a );

		// Make the matrix explicitly Hermitian.
		bli_mkherm( &a );
		bli_mktrim( &a );

		bli_copym( &a, &a_save );
	
		dtime_save = DBL_MAX;

		fflush( stdout );

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &a_save, &a );

			dtime = 0.0;
			dtime = bli_clock();

			bli_chol( &a );

			dtime_save = bli_clock_min_diff( dtime_save, dtime );

		}
		//printf("%g\n", dtime_save);

		gflops = ( ( 1.0 / 3.0 ) * m * m * m ) / dtime_save / 1.0e9;

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

#ifdef BLIS
		printf( "data_chol_blis" );
#else
		printf( "data_chol_%s", BLAS );
#endif
		printf( "( %3llu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m, ( unsigned long )256, gflops );

		bli_obj_free( &a );
		bli_obj_free( &a_save );
	}

	//bli_finalize();

	return 0;
}

