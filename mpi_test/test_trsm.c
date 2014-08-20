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

#include <unistd.h>
#include "blis.h"
#include <mpi.h>

//           side   uplo   trans  diag   m     n     alpha    a        lda   b        ldb
//void dtrsm_( char*, char*, char*, char*, int*, int*, double*, double*, int*, double*, int* );

//#define PRINT

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input;
	num_t dt_a, dt_b, dt_c;
	num_t dt_alpha, dt_beta;
	int   r, n_repeats;
	side_t side;
	uplo_t uplo;

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

	n_repeats = 3;

    if( argc < 7 ) 
    {   
        printf("Usage:\n");
        printf("test_foo.x m n k p_begin p_inc p_end:\n");
        exit;
    }   

    int world_size, world_rank, provided;
    MPI_Init_thread( NULL, NULL, MPI_THREAD_FUNNELED, &provided );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );

    m_input = strtol( argv[1], NULL, 10 );
    n_input = strtol( argv[2], NULL, 10 );
    p_begin = strtol( argv[4], NULL, 10 );
    p_inc   = strtol( argv[5], NULL, 10 );
    p_end   = strtol( argv[6], NULL, 10 );

#if 1
	dt_a = BLIS_DOUBLE;
	dt_b = BLIS_DOUBLE;
	dt_c = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;
#else
	dt_a = dt_b = dt_c = dt_alpha = dt_beta = BLIS_FLOAT; 
	//dt_a = dt_b = dt_c = dt_alpha = dt_beta = BLIS_SCOMPLEX; 
#endif

	side = BLIS_LEFT;
	//side = BLIS_RIGHT;

	uplo = BLIS_LOWER;
	//uplo = BLIS_UPPER;

    for ( p = p_begin + world_rank * p_inc; p <= p_end; p += p_inc * world_size )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;


		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		if ( bli_is_left( side ) )
			bli_obj_create( dt_a, m, m, 0, 0, &a );
		else
			bli_obj_create( dt_a, n, n, 0, 0, &a );
		bli_obj_create( dt_b, m, n, 0, 0, &b );
		bli_obj_create( dt_c, m, n, 0, 0, &c );
		bli_obj_create( dt_c, m, n, 0, 0, &c_save );

		bli_obj_set_struc( BLIS_TRIANGULAR, a );
		bli_obj_set_uplo( uplo, a );
		//bli_obj_set_diag( BLIS_UNIT_DIAG, a );

		bli_randm( &a );
		bli_randm( &c );
		bli_randm( &b );

/*
		{ 
			obj_t a2;

			bli_obj_alias_to( a, a2 );
			bli_obj_toggle_uplo( a2 );
			bli_obj_inc_diag_off( 1, a2 );
			bli_setm( &BLIS_ZERO, &a2 );
			bli_obj_inc_diag_off( -2, a2 );
			bli_obj_toggle_uplo( a2 );
			bli_obj_set_diag( BLIS_NONUNIT_DIAG, a2 );
			bli_scalm( &BLIS_TWO, &a2 );
			//bli_scalm( &BLIS_TWO, &a );
		} 
*/

		bli_setsc(  (2.0/1.0), 0.0, &alpha );
		bli_setsc(  (1.0/1.0), 0.0, &beta );


		bli_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );

			dtime = bli_clock();


#ifdef PRINT
/*
			obj_t ar, ai;
			bli_obj_alias_to( a, ar );
			bli_obj_alias_to( a, ai );
			bli_obj_set_datatype( BLIS_DOUBLE, ar ); ar.rs *= 2; ar.cs *= 2;
			bli_obj_set_datatype( BLIS_DOUBLE, ai ); ai.rs *= 2; ai.cs *= 2; ai.buffer = ( double* )ai.buffer + 1;

			bli_printm( "ar", &ar, "%4.1f", "" );
			bli_printm( "ai", &ai, "%4.1f", "" );
*/

			bli_invertd( &a );
			bli_printm( "a", &a, "%4.1f", "" );
			bli_invertd( &a );
			bli_printm( "c", &c, "%4.1f", "" );
#endif

#ifdef BLIS
			//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bli_trsm( side,
			//bli_trsm4m( side,
			//bli_trsm3m( side,
			          &alpha,
			          &a,
			          &c );
#else

		if ( bli_is_real( dt_a ) )
		{
			f77_char side   = 'L';
			f77_char uplo   = 'L';
			f77_char transa = 'N';
			f77_char diag   = 'N';
			f77_int  mm     = bli_obj_length( c );
			f77_int  nn     = bli_obj_width( c );
			f77_int  lda    = bli_obj_col_stride( a );
			f77_int  ldc    = bli_obj_col_stride( c );
			float *  alphap = bli_obj_buffer( alpha );
			float *  ap     = bli_obj_buffer( a );
			float *  cp     = bli_obj_buffer( c );

			strsm_( &side,
			        &uplo,
			        &transa,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
		else // if ( bli_is_complex( dt_a ) )
		{
			f77_char  side   = 'L';
			f77_char  uplo   = 'L';
			f77_char  transa = 'N';
			f77_char  diag   = 'N';
			f77_int   mm     = bli_obj_length( c );
			f77_int   nn     = bli_obj_width( c );
			f77_int   lda    = bli_obj_col_stride( a );
			f77_int   ldc    = bli_obj_col_stride( c );
			scomplex* alphap = bli_obj_buffer( alpha );
			scomplex* ap     = bli_obj_buffer( a );
			scomplex* cp     = bli_obj_buffer( c );

			ctrsm_( &side,
			//ztrsm_( &side,
			        &uplo,
			        &transa,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
		
#endif

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		if ( bli_is_left( side ) )
			gflops = ( 1.0 * m * m * n ) / ( dtime_save * 1.0e9 );
		else
			gflops = ( 1.0 * m * n * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt_a ) ) gflops *= 4.0;

#ifdef BLIS
		printf( "data_trsm_blis" );
#else
		printf( "data_trsm_%s", BLAS );
#endif
		printf( "( %2lu, 1:4 ) = [ %4lu %4lu  %10.3e  %6.3f ];\n",
		        ( unsigned long )(p - p_begin + 1)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )n, dtime_save, gflops );


		bli_obj_free( &alpha );
		bli_obj_free( &beta );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	bli_finalize();

	return 0;
}

