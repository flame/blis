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
#include "blis2.h"

//           uploa  transa m     k     alpha    a        lda   beta     c        ldc
void dsyrk_( char*, char*, int*, int*, double*, double*, int*, double*, double*, int* );

int main( int argc, char** argv )
{
	obj_t a, ah, c;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, k;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, k_input;
	num_t dt_a, dt_b, dt_c;
	num_t dt_alpha, dt_beta;
	int   r, n_repeats;

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

	herk_t*  herk_cntl_bp_ke;
	herk_t*  herk_cntl_op_bp;
	herk_t*  herk_cntl_mm_op;
	herk_t*  herk_cntl_vl_mm;

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
	k_input = 200;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 15;
	k_input = 13;
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
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;


		bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bl2_obj_create( dt_beta,  1, 1, 0, 0, &beta );

		bl2_obj_create( dt_a, m, k, 0, 0, &a );
		bl2_obj_create( dt_c, m, m, 0, 0, &c );
		bl2_obj_create( dt_c, m, m, 0, 0, &c_save );

		bl2_randm( &a );
		bl2_randm( &c );

		bl2_obj_set_struc( BLIS_HERMITIAN, c );
		bl2_obj_set_uplo( BLIS_LOWER, c );

		bl2_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, ah );

		bl2_setsc(  (2.0/1.0), 0.0, &alpha );
		bl2_setsc( -(1.0/1.0), 0.0, &beta );

		mr = bl2_blksz_obj_create( 2, 4, 2, 2 );
		kr = bl2_blksz_obj_create( 1, 1, 1, 1 );
		nr = bl2_blksz_obj_create( 1, 4, 1, 1 );
		mc = bl2_blksz_obj_create( 128, 368, 128, 128 );
		kc = bl2_blksz_obj_create( 256, 256, 256, 256 );
		nc = bl2_blksz_obj_create( 512, 512, 512, 512 );
		ni = bl2_blksz_obj_create(  16,  16,  16,  16 );

		scalm_cntl =
		bl2_scalm_cntl_obj_create( BLIS_UNBLOCKED,
		                           BLIS_VARIANT1 );

		packm_cntl_a =
		bl2_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           mr,
		                           kr, 
		                           FALSE, // scale?
		                           FALSE, // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_ROW_PANELS );

		packm_cntl_b =
		bl2_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           kr,
		                           nr, 
		                           FALSE, // scale?
		                           FALSE, // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_COL_PANELS );

		herk_cntl_bp_ke =
		bl2_herk_cntl_obj_create( BLIS_UNB_OPT,
		                          BLIS_VARIANT2,
		                          NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		herk_cntl_op_bp =
		bl2_herk_cntl_obj_create( BLIS_BLOCKED,
		                          //BLIS_VARIANT4,
		                          BLIS_VARIANT1,
		                          mc,
		                          ni,
		                          NULL,
		                          packm_cntl_a,
		                          packm_cntl_b,
		                          NULL,
		                          herk_cntl_bp_ke,
		                          NULL );

		herk_cntl_mm_op =
		bl2_herk_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT3,
		                          kc,
		                          NULL,
		                          NULL, //scalm_cntl,
		                          NULL,
		                          NULL,
		                          NULL,
		                          herk_cntl_op_bp,
		                          NULL );

		herk_cntl_vl_mm =
		bl2_herk_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT2,
		                          nc,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          herk_cntl_mm_op,
		                          NULL );


		bl2_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bl2_copym( &c_save, &c );


			dtime = bl2_clock();

#ifdef PRINT
			bl2_printm( "a", &a, "%4.1f", "" );
			bl2_printm( "c", &c, "%4.1f", "" );
#endif

#ifdef BLIS

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bl2_herk_int( &alpha,
			              &a,
			              &ah,
			              &beta,
			              &c,
			              herk_cntl_mm_op );

#else

			char    uploa  = 'L';
			char    transa = 'N';
			int     mm     = bl2_obj_length( c );
			int     kk     = bl2_obj_width_after_trans( a );
			int     lda    = bl2_obj_col_stride( a );
			int     ldc    = bl2_obj_col_stride( c );
			double* alphap = bl2_obj_buffer( alpha );
			double* ap     = bl2_obj_buffer( a );
			double* betap  = bl2_obj_buffer( beta );
			double* cp     = bl2_obj_buffer( c );

			dsyrk_( &uploa,
			        &transa,
			        &mm,
			        &kk,
			        alphap,
			        ap, &lda,
			        betap,
			        cp, &ldc );
#endif

#ifdef PRINT
			bl2_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


			dtime_save = bl2_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( 1.0 * m * k * m ) / ( dtime_save * 1.0e9 );

#ifdef BLIS
		printf( "data_herk_blis" );
#else
		printf( "data_herk_%s", BLAS );
#endif
		printf( "( %2ld, 1:4 ) = [ %4lu %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, k, dtime_save, gflops );

		bl2_blksz_obj_free( mr );
		bl2_blksz_obj_free( nr );
		bl2_blksz_obj_free( kr );
		bl2_blksz_obj_free( mc );
		bl2_blksz_obj_free( nc );
		bl2_blksz_obj_free( kc );
		bl2_blksz_obj_free( ni );

		bl2_cntl_obj_free( scalm_cntl );
		bl2_cntl_obj_free( packm_cntl_a );
		bl2_cntl_obj_free( packm_cntl_b );
		bl2_cntl_obj_free( herk_cntl_bp_ke );
		bl2_cntl_obj_free( herk_cntl_op_bp );
		bl2_cntl_obj_free( herk_cntl_mm_op );
		bl2_cntl_obj_free( herk_cntl_vl_mm );

		bl2_obj_free( &alpha );
		bl2_obj_free( &beta );

		bl2_obj_free( &a );
		bl2_obj_free( &c );
		bl2_obj_free( &c_save );
	}

	bl2_finalize();

	return 0;
}

