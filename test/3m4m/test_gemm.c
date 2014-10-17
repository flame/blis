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

#include <unistd.h>
#include "blis.h"

void zgemm3m_( f77_char*, f77_char*, f77_int*, f77_int*, f77_int*, dcomplex*, dcomplex*, f77_int*, dcomplex*, f77_int*, dcomplex*, dcomplex*, f77_int* );


//#define PRINT

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input, k_input;
	num_t dt;
	char  dt_ch;
	int   r, n_repeats;
	trans_t  transa;
	trans_t  transb;
	f77_char f77_transa;
	f77_char f77_transb;

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	n_repeats = 3;

	p_begin = P_BEGIN;
	p_end   = P_END;
	p_inc   = P_INC;

	m_input = -1;
	n_input = -1;
	k_input = 1;


#if   defined DT_C

	dt = BLIS_SCOMPLEX; dt_ch = 'c';

#if 1
	#if   (defined _4M1) || (defined _3M1)
		k_input = 384/2;
	#elif (defined _4MHW) || (defined _3MHW)
		k_input = 384;
	#else /* asm */
		k_input = 256;
	#endif
#endif

#elif defined DT_Z

	dt = BLIS_DCOMPLEX; dt_ch = 'z';

#if 1
	#if   (defined _4M1) || (defined _3M1)
		k_input = 256/2;
	#elif (defined _4MHW) || (defined _3MHW)
		k_input = 256;
	#else /* asm */
		k_input = 192;
	#endif
#endif

#elif defined DT_S

	dt = BLIS_FLOAT; dt_ch = 's';
	k_input = 384;

#elif defined DT_D

	dt = BLIS_DOUBLE; dt_ch = 'd';
	k_input = 256;

#endif

	transa = BLIS_NO_TRANSPOSE;
	transb = BLIS_NO_TRANSPOSE;

	bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	bli_param_map_blis_to_netlib_trans( transb, &f77_transb );


	for ( p = p_begin; p <= p_end; p += p_inc )
	{
		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dt, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt, 1, 1, 0, 0, &beta );

		bli_obj_create( dt, m, k, 0, 0, &a );
		bli_obj_create( dt, k, n, 0, 0, &b );
		bli_obj_create( dt, m, n, 0, 0, &c );
		//bli_obj_create( dt, m, n, 4, 4*m, &c );
		bli_obj_create( dt, m, n, 0, 0, &c_save );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_conjtrans( transa, a );
		bli_obj_set_conjtrans( transb, b );

		bli_setsc(  (2.0/1.0), 0.0, &alpha );
		bli_setsc( -(1.0/1.0), 0.0, &beta );


		bli_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );


			dtime = bli_clock();


#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "" );
			bli_printm( "b", &b, "%4.1f", "" );
			bli_printm( "c", &c, "%4.1f", "" );
#endif

#ifdef BLIS

	#if   defined _4M1 
			bli_gemm4m( &alpha,
	#elif defined _4MHW 
			bli_gemm4mh( &alpha,
	#elif defined _3M1 
			bli_gemm3m( &alpha,
	#elif defined _3MHW 
			bli_gemm3mh( &alpha,
	#else              
			bli_gemm( &alpha,
	#endif
			          &a,
			          &b,
			          &beta,
			          &c );

#else

		if ( bli_is_float( dt ) )
		{
			f77_int  mm     = bli_obj_length( c );
			f77_int  kk     = bli_obj_width_after_trans( a );
			f77_int  nn     = bli_obj_width( c );
			f77_int  lda    = bli_obj_col_stride( a );
			f77_int  ldb    = bli_obj_col_stride( b );
			f77_int  ldc    = bli_obj_col_stride( c );
			float*   alphap = bli_obj_buffer( alpha );
			float*   ap     = bli_obj_buffer( a );
			float*   bp     = bli_obj_buffer( b );
			float*   betap  = bli_obj_buffer( beta );
			float*   cp     = bli_obj_buffer( c );

			sgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_double( dt ) )
		{
			f77_int  mm     = bli_obj_length( c );
			f77_int  kk     = bli_obj_width_after_trans( a );
			f77_int  nn     = bli_obj_width( c );
			f77_int  lda    = bli_obj_col_stride( a );
			f77_int  ldb    = bli_obj_col_stride( b );
			f77_int  ldc    = bli_obj_col_stride( c );
			double*  alphap = bli_obj_buffer( alpha );
			double*  ap     = bli_obj_buffer( a );
			double*  bp     = bli_obj_buffer( b );
			double*  betap  = bli_obj_buffer( beta );
			double*  cp     = bli_obj_buffer( c );

			dgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_scomplex( dt ) )
		{
			f77_int  mm     = bli_obj_length( c );
			f77_int  kk     = bli_obj_width_after_trans( a );
			f77_int  nn     = bli_obj_width( c );
			f77_int  lda    = bli_obj_col_stride( a );
			f77_int  ldb    = bli_obj_col_stride( b );
			f77_int  ldc    = bli_obj_col_stride( c );
			scomplex*  alphap = bli_obj_buffer( alpha );
			scomplex*  ap     = bli_obj_buffer( a );
			scomplex*  bp     = bli_obj_buffer( b );
			scomplex*  betap  = bli_obj_buffer( beta );
			scomplex*  cp     = bli_obj_buffer( c );

			cgemm_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
		else if ( bli_is_dcomplex( dt ) )
		{
			f77_int  mm     = bli_obj_length( c );
			f77_int  kk     = bli_obj_width_after_trans( a );
			f77_int  nn     = bli_obj_width( c );
			f77_int  lda    = bli_obj_col_stride( a );
			f77_int  ldb    = bli_obj_col_stride( b );
			f77_int  ldc    = bli_obj_col_stride( c );
			dcomplex*  alphap = bli_obj_buffer( alpha );
			dcomplex*  ap     = bli_obj_buffer( a );
			dcomplex*  bp     = bli_obj_buffer( b );
			dcomplex*  betap  = bli_obj_buffer( beta );
			dcomplex*  cp     = bli_obj_buffer( c );

			zgemm_( &f77_transa,
			//zgemm3m_( &f77_transa,
			        &f77_transb,
			        &mm,
			        &nn,
			        &kk,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
#endif

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

		if ( bli_is_complex( dt ) ) gflops *= 4.0;

#ifdef BLIS
		printf( "data_%s_%cgemm_%s_blis", THR_STR, dt_ch, STR );
#else
		printf( "data_%s_%cgemm_%s",      THR_STR, dt_ch, STR );
#endif
		printf( "( %2lu, 1:5 ) = [ %4lu %4lu %4lu  %10.3e  %6.3f ];\n",
		        ( unsigned long )(p - p_begin + 1)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
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

