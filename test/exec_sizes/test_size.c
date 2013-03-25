/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2013, The University of Texas

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

void     dgemm_ ( char* transa, char* transb, int* m, int* n, int* k, double*   alpha, double*   a, int* lda, double*   b, int* ldb, double*   beta, double*   c, int* ldc );
void     zhemm_ ( char* side, char* uplo, int* m, int* n, dcomplex* alpha, dcomplex* a, int* lda, dcomplex* b, int* ldb, dcomplex* beta, dcomplex* c, int* ldc );
void     zherk_ ( char* uplo, char* transa, int* n, int* k, double* alpha, dcomplex* a, int* lda, double* beta, dcomplex* c, int* ldc );
void     zher2k_( char* uplo, char* transa, int* n, int* k, dcomplex* alpha, dcomplex* a, int* lda, dcomplex* b, int* ldb, double* beta, dcomplex* c, int* ldc );
void     dsymm_ ( char* side, char* uplo, int* m, int* n, double*   alpha, double*   a, int* lda, double*   b, int* ldb, double*   beta, double*   c, int* ldc );
void     dsyrk_ ( char* uplo, char* transa, int* n, int* k, double*   alpha, double*   a, int* lda, double*   beta, double*   c, int* ldc );
void     dsyr2k_( char* uplo, char* transa, int* n, int* k, double*   alpha, double*   a, int* lda, double*   b, int* ldb, double*   beta, double*   c, int* ldc );
void     dtrmm_ ( char* side, char* uplo, char* transa, char* diag, int* m, int* n, double*   alpha, double*   a, int* lda, double*   b, int* ldb );
void     dtrsm_ ( char* side, char* uplo, char* transa, char* diag, int* m, int* n, double*   alpha, double*   a, int* lda, double*   b, int* ldb );

int main( int argc, char** argv )
{
	obj_t a, b, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input, k_input;
	num_t dt_a, dt_b, dt_c;
	num_t dt_alpha, dt_beta;
	int   r, n_repeats;
	int   ii;

	blksz_t* mr;
	blksz_t* nr;
	blksz_t* kr;
	blksz_t* mc;
	blksz_t* nc;
	blksz_t* kc;
	blksz_t* ni;

	scalm_t* scalm_cntl;
	packm_t* packm_cntl_a;
	packm_t* packm_cntl_b;

	gemm_t*  gemm_cntl_bp_ke;
	gemm_t*  gemm_cntl_op_bp;
	gemm_t*  gemm_cntl_mm_op;
	gemm_t*  gemm_cntl_vl_mm;

	double dtime;
	double dtime_save;
	double gflops;

	bli_init();

	n_repeats = 3;

	p_begin = 32;
	p_end   = 1600;
	p_inc   = 32;

#if 0
	m_input = -1;
	//m_input = 128;
	//m_input = 256;
	n_input = -1;
	//n_input = 1024;
	k_input = -1;
	k_input = 256;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 16;
	k_input = 16;
	n_input = 16;
#endif

	dt_a = BLIS_DOUBLE;
	dt_b = BLIS_DOUBLE;
	dt_c = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;

	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;


		bli_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bli_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		bli_obj_create( dt_a, m, k, 0, 0, &a );
		bli_obj_create( dt_b, k, n, 0, 0, &b );
		bli_obj_create( dt_c, m, n, 0, 0, &c );
		bli_obj_create( dt_c, m, n, 0, 0, &c_save );

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );


		bli_setsc(  (2.0/1.0), 0.0, &alpha );
		bli_setsc( -(1.0/1.0), 0.0, &beta );

		mr = bli_blksz_obj_create( 2, 4, 2, 2 );
		kr = bli_blksz_obj_create( 1, 2, 1, 1 );
		nr = bli_blksz_obj_create( 1, 2, 1, 1 );
		mc = bli_blksz_obj_create( 128, 128, 128, 128 );
		kc = bli_blksz_obj_create( 256, 256, 256, 256 );
		nc = bli_blksz_obj_create( 512, 512, 512, 512 );
		ni = bli_blksz_obj_create(  16,  16,  16,  16 );

		scalm_cntl =
		bli_scalm_cntl_obj_create( BLIS_UNBLOCKED,
		                           BLIS_VARIANT1 );

		packm_cntl_a =
		bli_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           mr,
		                           kr, 
		                           TRUE,  // scale?
		                           TRUE,  // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_ROW_PANELS );

		packm_cntl_b =
		bli_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           kr,
		                           nr, 
		                           FALSE, // scale?
		                           FALSE, // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_COL_PANELS );

		gemm_cntl_bp_ke =
		bli_gemm_cntl_obj_create( BLIS_UNB_OPT,
		                          BLIS_VARIANT2,
		                          NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		gemm_cntl_op_bp =
		bli_gemm_cntl_obj_create( BLIS_BLOCKED,
		                          //BLIS_VARIANT4,
		                          BLIS_VARIANT1,
		                          mc,
		                          ni,
		                          NULL,
		                          packm_cntl_a,
		                          packm_cntl_b,
		                          NULL,
		                          gemm_cntl_bp_ke,
		                          NULL );

		gemm_cntl_mm_op =
		bli_gemm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT3,
		                          kc,
		                          NULL,
		                          scalm_cntl,
		                          NULL,
		                          NULL,
		                          NULL,
		                          gemm_cntl_op_bp,
		                          NULL );

		gemm_cntl_vl_mm =
		bli_gemm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT2,
		                          nc,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          gemm_cntl_mm_op,
		                          NULL );


		bli_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );


			dtime = bli_clock();


#ifdef NBLIS

	#if NBLIS >= 1
		for ( ii = 0; ii < 2000000000; ++ii )
		{
			bli_gemm( &BLIS_ONE,
			          &a,
			          &b,
			          &BLIS_ONE,
			          &c );
		}
	#endif

	#if NBLIS >= 2
			bli_hemm( BLIS_LEFT,
			          &BLIS_ONE,
			          &a,
			          &b,
			          &BLIS_ONE,
			          &c );
	#endif

	#if NBLIS >= 3
			bli_herk( &BLIS_ONE,
			          &a,
			          &BLIS_ONE,
			          &c );
	#endif

	#if NBLIS >= 4
			bli_her2k( &BLIS_ONE,
			           &a,
			           &b,
			           &BLIS_ONE,
			           &c );
	#endif

	#if NBLIS >= 5
			bli_trmm( BLIS_LEFT,
			          &BLIS_ONE,
			          &a,
			          &c );
	#endif

	#if NBLIS >= 6
			bli_trsm( BLIS_LEFT,
			          &BLIS_ONE,
			          &a,
			          &c );
	#endif

#endif



#ifdef NBLAS

	#if NBLAS >= 1
		for ( ii = 0; ii < 2000000000; ++ii )
		{
			char    transa = 'N';
			char    transb = 'N';
			int     mm     = bli_obj_length( c );
			int     kk     = bli_obj_width_after_trans( a );
			int     nn     = bli_obj_width( c );
			int     lda    = bli_obj_col_stride( a );
			int     ldb    = bli_obj_col_stride( b );
			int     ldc    = bli_obj_col_stride( c );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a );
			double* bp     = bli_obj_buffer( b );
			double* betap  = bli_obj_buffer( beta );
			double* cp     = bli_obj_buffer( c );

			dgemm_( &transa,
			        &transb,
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

	#if NBLAS >= 2
		{
			char    side   = 'L';
			char    uplo   = 'L';
			int     mm     = bli_obj_length( c );
			int     nn     = bli_obj_width( c );
			int     lda    = bli_obj_col_stride( a );
			int     ldb    = bli_obj_col_stride( b );
			int     ldc    = bli_obj_col_stride( c );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a );
			double* bp     = bli_obj_buffer( b );
			double* betap  = bli_obj_buffer( beta );
			double* cp     = bli_obj_buffer( c );

			dsymm_( &side,
			        &uplo,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        bp, &ldb,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 3
		{
			char    uplo   = 'L';
			char    trans  = 'N';
			int     mm     = bli_obj_length( c );
			int     kk     = bli_obj_width( a );
			int     lda    = bli_obj_col_stride( a );
			int     ldc    = bli_obj_col_stride( c );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a );
			double* betap  = bli_obj_buffer( beta );
			double* cp     = bli_obj_buffer( c );

			dsyrk_( &uplo,
			        &trans,
			        &mm,
			        &kk,
			        alphap,
			        ap, &lda,
			        betap,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 4
		{
			char    uplo   = 'L';
			char    trans  = 'N';
			int     mm     = bli_obj_length( c );
			int     kk     = bli_obj_width( a );
			int     lda    = bli_obj_col_stride( a );
			int     ldb    = bli_obj_col_stride( b );
			int     ldc    = bli_obj_col_stride( c );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a );
			double* bp     = bli_obj_buffer( b );
			double* betap  = bli_obj_buffer( beta );
			double* cp     = bli_obj_buffer( c );

			dsyr2k_( &uplo,
			         &trans,
			         &mm,
			         &kk,
			         alphap,
			         ap, &lda,
			         bp, &ldb,
			         betap,
			         cp, &ldc );
		}
	#endif

	#if NBLAS >= 5
		{
			char    side   = 'L';
			char    uplo   = 'L';
			char    trans  = 'N';
			char    diag   = 'N';
			int     mm     = bli_obj_length( c );
			int     nn     = bli_obj_width( c );
			int     lda    = bli_obj_col_stride( a );
			int     ldc    = bli_obj_col_stride( c );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a );
			double* cp     = bli_obj_buffer( c );

			dtrmm_( &side,
			        &uplo,
			        &trans,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
	#endif

	#if NBLAS >= 6
		{
			char    side   = 'L';
			char    uplo   = 'L';
			char    trans  = 'N';
			char    diag   = 'N';
			int     mm     = bli_obj_length( c );
			int     nn     = bli_obj_width( c );
			int     lda    = bli_obj_col_stride( a );
			int     ldc    = bli_obj_col_stride( c );
			double* alphap = bli_obj_buffer( alpha );
			double* ap     = bli_obj_buffer( a );
			double* cp     = bli_obj_buffer( c );

			dtrsm_( &side,
			        &uplo,
			        &trans,
			        &diag,
			        &mm,
			        &nn,
			        alphap,
			        ap, &lda,
			        cp, &ldc );
		}
	#endif

#endif


			dtime = bli_clock() - dtime;

			dtime_save = bli_min( dtime, dtime_save );

		}

		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );

		printf( "data_blis( %2ld, 1:5 ) = [ %4lu %4lu %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, k, n, dtime_save, gflops );

		bli_blksz_obj_free( mr );
		bli_blksz_obj_free( nr );
		bli_blksz_obj_free( kr );
		bli_blksz_obj_free( mc );
		bli_blksz_obj_free( nc );
		bli_blksz_obj_free( kc );
		bli_blksz_obj_free( ni );

		bli_cntl_obj_free( scalm_cntl );
		bli_cntl_obj_free( packm_cntl_a );
		bli_cntl_obj_free( packm_cntl_b );
		bli_cntl_obj_free( gemm_cntl_bp_ke );
		bli_cntl_obj_free( gemm_cntl_op_bp );
		bli_cntl_obj_free( gemm_cntl_mm_op );
		bli_cntl_obj_free( gemm_cntl_vl_mm );

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

