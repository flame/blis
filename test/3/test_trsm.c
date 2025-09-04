/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2023, Advanced Micro Devices, Inc. All rights reserved.

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
#include "blis.h"

//#define PRINT

int main( int argc, char** argv )
{
	obj_t    a, c;
	obj_t    c_save;
	obj_t    alpha;
	dim_t    m, n;
	dim_t    p;
	dim_t    p_begin, p_max, p_inc;
	int      m_input, n_input;
	ind_t    ind;
	num_t    dt;
	char     dt_ch;
	int      r, n_repeats;
	side_t   side;
	uplo_t   uploa;
	trans_t  transa;
	diag_t   diaga;
	f77_char f77_side;
	f77_char f77_uploa;
	f77_char f77_transa;
	f77_char f77_diaga;

	double   dtime;
	double   dtime_save;
	double   gflops;

	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 3;

	dt      = DT;

	ind     = IND;

	p_begin = P_BEGIN;
	p_max   = P_MAX;
	p_inc   = P_INC;

	m_input = -1;
	n_input = -1;


	// Supress compiler warnings about unused variable 'ind'.
	( void )ind;

#if 0

	cntx_t* cntx;

	ind_t ind_mod = ind;

	// Initialize a context for the current induced method and datatype.
	cntx = bli_gks_query_ind_cntx( ind_mod, dt );

	// Set k to the kc blocksize for the current datatype.
	k_input = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );

#elif 1

	//k_input = 256;

#endif

	// Choose the char corresponding to the requested datatype.
	if      ( bli_is_float( dt ) )    dt_ch = 's';
	else if ( bli_is_double( dt ) )   dt_ch = 'd';
	else if ( bli_is_scomplex( dt ) ) dt_ch = 'c';
	else                              dt_ch = 'z';

#if 0
	side   = BLIS_LEFT;
#else
	side   = BLIS_RIGHT;
#endif
#if 0
	uploa  = BLIS_LOWER;
#else
	uploa  = BLIS_UPPER;
#endif
	transa = BLIS_NO_TRANSPOSE;
	diaga  = BLIS_NONUNIT_DIAG;

	bli_param_map_blis_to_netlib_side( side, &f77_side );
	bli_param_map_blis_to_netlib_uplo( uploa, &f77_uploa );
	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_diag( diaga, &f77_diaga );

	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_max; p += p_inc ) ;

	printf( "data_%s_%ctrsm_%s", THR_STR, dt_ch, STR );
	printf( "( %2lu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0,
	        ( unsigned long )0, 0.0 );


	for ( p = p_begin; p <= p_max; p += p_inc )
	{

		if ( m_input < 0 ) m = p / ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p / ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );

		if ( bli_is_left( side ) )
			bli_obj_create( dt, m, m, 0, 0, &a );
        else
			bli_obj_create( dt, n, n, 0, 0, &a );
		bli_obj_create( dt, m, n, 0, 0, &c );
		//bli_obj_create( dt, m, n, n, 1, &c );
		bli_obj_create( dt, m, n, 0, 0, &c_save );

		bli_randm( &a );
		bli_randm( &c );

		bli_obj_set_struc( BLIS_TRIANGULAR, &a );
		bli_obj_set_uplo( uploa, &a );
		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_diag( diaga, &a );

		bli_randm( &a );
		bli_mktrim( &a );

		// Load the diagonal of A to make it more likely to be invertible.
		bli_shiftd( &BLIS_TWO, &a );

		bli_setsc(  (2.0/1.0), 0.0, &alpha );

		bli_copym( &c, &c_save );
	
#if 0 //def BLIS
		bli_ind_disable_all_dt( dt );
		bli_ind_enable_dt( ind, dt );
#endif

		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );

			dtime = bli_clock();

#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "" );
			bli_printm( "c", &c, "%4.1f", "" );
#endif

#ifdef BLIS

			bli_trsm( side,
			          &alpha,
			          &a,
			          &c );

#else

			if ( bli_is_float( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldc    = bli_obj_col_stride( &c );
				float*    alphap = ( float* )bli_obj_buffer( &alpha );
				float*    ap     = ( float* )bli_obj_buffer( &a );
				float*    cp     = ( float* )bli_obj_buffer( &c );

				strsm_( &f77_side,
						&f77_uploa,
						&f77_transa,
						&f77_diaga,
						&mm,
						&kk,
						alphap,
						ap, &lda,
						cp, &ldc );
			}
			else if ( bli_is_double( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldc    = bli_obj_col_stride( &c );
				double*   alphap = ( double* )bli_obj_buffer( &alpha );
				double*   ap     = ( double* )bli_obj_buffer( &a );
				double*   cp     = ( double* )bli_obj_buffer( &c );

				dtrsm_( &f77_side,
						&f77_uploa,
						&f77_transa,
						&f77_diaga,
						&mm,
						&kk,
						alphap,
						ap, &lda,
						cp, &ldc );
			}
			else if ( bli_is_scomplex( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldc    = bli_obj_col_stride( &c );
#ifdef EIGEN
				float*    alphap = ( float*    )bli_obj_buffer( &alpha );
				float*    ap     = ( float*    )bli_obj_buffer( &a );
				float*    cp     = ( float*    )bli_obj_buffer( &c );
#else
				scomplex* alphap = ( scomplex* )bli_obj_buffer( &alpha );
				scomplex* ap     = ( scomplex* )bli_obj_buffer( &a );
				scomplex* cp     = ( scomplex* )bli_obj_buffer( &c );
#endif

				ctrsm_( &f77_side,
						&f77_uploa,
						&f77_transa,
						&f77_diaga,
						&mm,
						&kk,
						alphap,
						ap, &lda,
						cp, &ldc );
			}
			else if ( bli_is_dcomplex( dt ) )
			{
				f77_int   mm     = bli_obj_length( &c );
				f77_int   kk     = bli_obj_width( &c );
				f77_int   lda    = bli_obj_col_stride( &a );
				f77_int   ldc    = bli_obj_col_stride( &c );
#ifdef EIGEN
				double*   alphap = ( double*   )bli_obj_buffer( &alpha );
				double*   ap     = ( double*   )bli_obj_buffer( &a );
				double*   cp     = ( double*   )bli_obj_buffer( &c );
#else
				dcomplex* alphap = ( dcomplex* )bli_obj_buffer( &alpha );
				dcomplex* ap     = ( dcomplex* )bli_obj_buffer( &a );
				dcomplex* cp     = ( dcomplex* )bli_obj_buffer( &c );
#endif

				ztrsm_( &f77_side,
						&f77_uploa,
						&f77_transa,
						&f77_diaga,
						&mm,
						&kk,
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

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

		printf( "data_%s_%ctrsm_%s", THR_STR, dt_ch, STR );
		printf( "( %2lu, 1:3 ) = [ %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )n, gflops );

		bli_obj_free( &alpha );

		bli_obj_free( &a );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	//bli_finalize();

	return 0;
}

