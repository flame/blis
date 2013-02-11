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

double FLA_Clock( void );

extern gemm_t* gemm_cntl;
//           trans  m     n     alpha    a        lda   x        incx  beta     y       incy
void dgemv_( char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );
void dger_( int*, int*, double*, double*, int*, double*, int*, double*, int* );
void dsymv_( char*, int*, double*, double*, int*, double*, int*, double*, double*, int* );
void dsyr_( char*, int*, double*, double*, int*, double*, int* );
void dsyr2_( char*, int*, double*, double*, int*, double*, int*, double*, int* );
void dtrmv_( char*, char*, char*, int*, double*, int*, double*, int* );
void dtrsv_( char*, char*, char*, int*, double*, int*, double*, int* );

//           trans  trans m     n     k     alpha    a        lda   b        ldb   beta     c        ldc
void dgemm_( char*, char*, int*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int* );

int main( int argc, char** argv )
{

#if 1
{
	obj_t a, b, c, w;
	//obj_t a1, b1, c1;
	//obj_t a11, b11, c11;
	obj_t a_pack, b_pack, c_pack;
	obj_t c_save;
	obj_t alpha, beta;
	dim_t m, n, k;
	//dim_t b_part;
	dim_t p;
	dim_t p_begin, p_end, p_inc;
	int   m_input, n_input, k_input;
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
	packm_t* trmm_packm_cntl_a;
	packm_t* trmm_packm_cntl_b;
	packm_t* trsm_packm_cntl_a;
	packm_t* trsm_packm_cntl_b;
	unpackm_t* trsm_unpackm_cntl_b;

	gemm_t*  gemm_cntl_bp_ke;
	gemm_t*  gemm_cntl_op_bp;
	gemm_t*  gemm_cntl_mm_op;
	gemm_t*  gemm_cntl_vl_mm;
	herk_t*  herk_cntl_bp_ke;
	herk_t*  herk_cntl_op_bp;
	herk_t*  herk_cntl_mm_op;
	herk_t*  herk_cntl_vl_mm;
	her2k_t* her2k_cntl_bp_ke;
	her2k_t* her2k_cntl_op_bp;
	her2k_t* her2k_cntl_mm_op;
	her2k_t* her2k_cntl_vl_mm;
	trmm_t*  trmm_cntl_bp_ke;
	trmm_t*  trmm_cntl_op_bp;
	trmm_t*  trmm_cntl_mm_op;
	trmm_t*  trmm_cntl_vl_mm;
	trmm_t*  trmm3_cntl_mm_op;
	trsm_t*  trsm_cntl_bp_ke;
	trsm_t*  trsm_cntl_op_bp;
	trsm_t*  trsm_cntl_mm_op;
	trsm_t*  trsm_cntl_vl_mm;

	double dtime;
	double dtime_save;
	double gflops;

	bl2_init();

	n_repeats = 3;

#if 1
	p_begin = 32;
	p_end   = 1600;
	p_inc   = 32;
#else
	p_begin = 768;
	p_end   = 768;
	p_inc   = 1;
#endif

#if 1
	m_input = -2;
	//m_input = 128;
	//m_input = 256;
	n_input = -1;
	//n_input = 1024;
	//k_input = -1;
	//k_input = 256;
	k_input = -1;
#else
	p_begin = 16;
	p_end   = 16;
	p_inc   = 1;

	m_input = 15;
	k_input = 13;
	n_input = 15;
#endif

	dt_a = BLIS_DOUBLE;
	//dt_a = BLIS_DCOMPLEX;
	dt_b = BLIS_DOUBLE;
	dt_c = BLIS_DOUBLE;
	//dt_c = BLIS_DCOMPLEX;
	dt_alpha = BLIS_DOUBLE;
	dt_beta = BLIS_DOUBLE;

	//p = p_begin;
	for ( p = p_begin; p <= p_end; p += p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;


		bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
		bl2_obj_create( dt_beta,  1, 1, 0, 0, &beta );

#if 1
		bl2_obj_create( dt_a, m, k, 0, 0, &a );
		bl2_obj_create( dt_b, k, n, 0, 0, &b );
		bl2_obj_create( dt_c, m, n, 0, 0, &c );
		bl2_obj_create( dt_c, m, n, 0, 0, &w );
		bl2_obj_create( dt_c, m, n, 0, 0, &c_save );

		bl2_randm( &a );
		bl2_randm( &b );
		bl2_randm( &c );
		bl2_randm( &w );
#elif 0
		bl2_obj_create( dt_a, m, k, 0, 0, &a );
		bl2_obj_create( dt_b, m, k, 0, 0, &b );
		bl2_obj_create( dt_c, m, m, 0, 0, &c );
		bl2_obj_create( dt_c, m, m, 0, 0, &c_save );

		bl2_randm( &a );
		bl2_randm( &b );
		bl2_randm( &c );
		bl2_setm( &BLIS_ZERO, &c );
#else
		bl2_obj_create( dt_a, m, m, 0, 0, &a );
		bl2_obj_create( dt_b, m, n, 0, 0, &b );
		bl2_obj_create( dt_c, m, n, 0, 0, &c );
		bl2_obj_create( dt_c, m, n, 0, 0, &c_save );

		bl2_obj_set_struc( BLIS_TRIANGULAR, a );
		//bl2_obj_set_uplo( BLIS_UPPER, a );
		bl2_obj_set_uplo( BLIS_LOWER, a );
		//bl2_obj_set_diag_offset( 4, a );

		bl2_randm( &a );
		bl2_randm( &c );
		bl2_randm( &b );
#endif


		bl2_obj_init_pack( &a_pack );
		bl2_obj_init_pack( &b_pack );
		bl2_obj_init_pack( &c_pack );

#if 0
		bl2_obj_set_struc( BLIS_TRIANGULAR, a );
		//bl2_obj_set_uplo( BLIS_LOWER, a );
		bl2_obj_set_uplo( BLIS_UPPER, a );
		bl2_obj_set_diag( BLIS_UNIT_DIAG, a );
		bl2_setm( &BLIS_ZERO, &a );
		bl2_obj_set_struc( BLIS_GENERAL, a );
		bl2_obj_set_uplo( BLIS_DENSE, a );
		bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );
#endif

		bl2_setsc(  (2.0/1.0), 0.0, &alpha );
		bl2_setsc( -(1.0/1.0), 0.0, &beta );

		mr = bl2_blksz_obj_create( 2, 4, 2, 2 );
		kr = bl2_blksz_obj_create( 1, 1, 1, 1 );
		nr = bl2_blksz_obj_create( 1, 4, 1, 1 );
		mc = bl2_blksz_obj_create( 128, 384, 128, 128 );
		kc = bl2_blksz_obj_create( 256, 384, 256, 256 );
		nc = bl2_blksz_obj_create( 512, 512, 512, 512 );
		ni = bl2_blksz_obj_create(  16,  32,  16,  16 );

		scalm_cntl =
		bl2_scalm_cntl_obj_create( BLIS_UNBLOCKED,
		                           BLIS_VARIANT1 );

		packm_cntl_a =
		bl2_packm_cntl_obj_create( BLIS_BLOCKED,
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

		gemm_cntl_bp_ke =
		bl2_gemm_cntl_obj_create( BLIS_UNB_OPT,
		                          BLIS_VARIANT2,
		                          NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		gemm_cntl_op_bp =
		bl2_gemm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT4,
		                          //BLIS_VARIANT1,
		                          mc,
		                          ni,
		                          NULL,
		                          packm_cntl_a,
		                          packm_cntl_b,
		                          NULL,
		                          gemm_cntl_bp_ke,
		                          NULL );

		gemm_cntl_mm_op =
		bl2_gemm_cntl_obj_create( BLIS_BLOCKED,
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
		bl2_gemm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT2,
		                          nc,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          gemm_cntl_mm_op,
		                          NULL );

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
		                          scalm_cntl,
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

		her2k_cntl_bp_ke =
		bl2_her2k_cntl_obj_create( BLIS_UNB_OPT,
		                           BLIS_VARIANT2,
		                           NULL, NULL, NULL, NULL, NULL,
		                           NULL, NULL, NULL, NULL );

		her2k_cntl_op_bp =
		bl2_her2k_cntl_obj_create( BLIS_BLOCKED,
		                           //BLIS_VARIANT4,
		                           BLIS_VARIANT1,
		                           mc,
		                           ni,
		                           NULL,
		                           packm_cntl_a,
		                           packm_cntl_b,
		                           NULL,
		                           her2k_cntl_bp_ke,
		                           herk_cntl_bp_ke,
		                           NULL );

		her2k_cntl_mm_op =
		bl2_her2k_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT3,
		                           kc,
		                           NULL,
		                           scalm_cntl,
		                           NULL,
		                           NULL,
		                           NULL,
		                           her2k_cntl_op_bp,
		                           NULL,
		                           NULL );

		her2k_cntl_vl_mm =
		bl2_her2k_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           nc,
		                           NULL,
		                           NULL,
		                           NULL,
		                           NULL,
		                           NULL,
		                           her2k_cntl_mm_op,
		                           NULL,
		                           NULL );

		trmm_packm_cntl_a =
		bl2_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT3,
		                           mr,
		                           kr, 
		                           TRUE,  // scale?
		                           TRUE,  // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_ROW_PANELS );

		trmm_packm_cntl_b =
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

		trmm_cntl_bp_ke =
		bl2_trmm_cntl_obj_create( BLIS_UNB_OPT,
		                          BLIS_VARIANT2,
		                          NULL, NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		trmm_cntl_op_bp =
		bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
		                          //BLIS_VARIANT4,
		                          BLIS_VARIANT1,
		                          mc,
		                          ni,
		                          NULL,
		                          trmm_packm_cntl_a,
		                          trmm_packm_cntl_b,
		                          NULL,
		                          trmm_cntl_bp_ke,
		                          gemm_cntl_bp_ke,
		                          NULL );

		trmm_cntl_mm_op =
		bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT3,
		                          kc,
		                          NULL,
		                          NULL, //scalm_cntl,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trmm_cntl_op_bp,
		                          NULL,
		                          NULL );

		trmm_cntl_vl_mm =
		bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT2,
		                          nc,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trmm_cntl_mm_op,
		                          NULL,
		                          NULL );

		trmm3_cntl_mm_op =
		bl2_trmm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT3,
		                          kc,
		                          NULL,
		                          scalm_cntl,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trmm_cntl_op_bp,
		                          NULL,
		                          NULL );

		trsm_packm_cntl_a =
		bl2_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT3,
		                           mr,    // IMPORTANT: n dim multiple must be mr to
		                           mr,    // support right and bottom-right edge cases
		                           FALSE, // scale?
		                           TRUE,  // densify?
		                           TRUE,  // invert diagonal?
		                           TRUE,  // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_ROW_PANELS );

		trsm_packm_cntl_b =
		bl2_packm_cntl_obj_create( BLIS_BLOCKED,
		                           BLIS_VARIANT2,
		                           mr,    // IMPORTANT: m dim multiple must be mr since
		                           nr,    // B_pack is updated (ie: serves as C) in trsm
		                           TRUE,  // scale?
		                           FALSE, // densify?
		                           FALSE, // invert diagonal?
		                           FALSE, // reverse iteration if upper?
		                           FALSE, // reverse iteration if lower?
		                           BLIS_PACKED_COL_PANELS );

		trsm_unpackm_cntl_b =
		bl2_unpackm_cntl_obj_create( BLIS_BLOCKED,
		                             BLIS_VARIANT2,
		                             NULL );

		trsm_cntl_bp_ke =
		bl2_trsm_cntl_obj_create( BLIS_UNB_OPT,
		                          //BLIS_VARIANT2,
		                          BLIS_VARIANT3,
		                          NULL, NULL, NULL, NULL, NULL,
		                          NULL, NULL, NULL, NULL );

		trsm_cntl_op_bp =
		bl2_trsm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT4,
		                          //BLIS_VARIANT1,
		                          mc,
		                          ni,
		                          NULL,
		                          trsm_packm_cntl_a,
		                          trsm_packm_cntl_b,
		                          NULL,
		                          trsm_cntl_bp_ke,
		                          gemm_cntl_bp_ke,
		                          NULL );

		trsm_cntl_mm_op =
		bl2_trsm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT3,
		                          kc,
		                          NULL,
		                          NULL, //scalm_cntl,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trsm_cntl_op_bp,
		                          NULL,
		                          NULL );

		trsm_cntl_vl_mm =
		bl2_trsm_cntl_obj_create( BLIS_BLOCKED,
		                          BLIS_VARIANT2,
		                          nc,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          NULL,
		                          trsm_cntl_mm_op,
		                          NULL,
		                          NULL );



		//bl2_printm( "a", &a, "%8.1e", "" );
		//bl2_printm( "b", &b, "%8.1e", "" );
		//bl2_printm( "c", &c, "%8.1e", "" );
		//bl2_printm( "alpha", &alpha, "%8.1e", "" );
		//bl2_printm( "beta", &beta, "%8.1e", "" );

		bl2_copym( &c, &c_save );
	
		dtime_save = 1.0e9;

		for ( r = 0; r < n_repeats; ++r )
		{
			bl2_copym( &c_save, &c );



			dtime = bl2_clock();


#if 0
			//bl2_mm_clear_smem();

			bl2_packm_init( &a, &a_pack, packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a, &a_pack, packm_cntl_a );

			bl2_packm_init( &b, &b_pack, packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &b, &b_pack, packm_cntl_b );

			bl2_gemm_ker_var2( &BLIS_ONE,
			                   &a_pack,
			                   &b_pack,
			                   &BLIS_ONE,
			                   &c,
			                   NULL );
#endif

#if 0
			bl2_packm_init( &b, &b_pack, packm_cntl_b );

			{
				dim_t bn_alg  = 32;
				dim_t bn_use, j;
				dim_t n_trans = bl2_obj_width_after_trans( b );
				obj_t c1_fuse, b1_fuse;

				bl2_packm_init( &a, &a_pack, packm_cntl_a );
				bl2_packm_int( &BLIS_ONE, &a, &a_pack, packm_cntl_a );

				for ( j = 0; j < n_trans; j += bn_alg )
				{
					bn_use = bl2_min( bn_alg, n_trans - j );

					bl2_acquire_mpart_l2r( BLIS_SUBPART1, j, bn_use, &c, &c1_fuse );
					bl2_acquire_mpart_l2r( BLIS_SUBPART1, j, bn_use, &b_pack, &b1_fuse );

					bl2_packm_int( &BLIS_ONE, &b, &b1_fuse, packm_cntl_b );

					bl2_gemm_ker_var2( &BLIS_ONE,
					                   &a_pack,
					                   &b1_fuse,
					                   &BLIS_ONE,
					                   &c1_fuse,
					                   NULL );
				}
			}
#endif

#if 1
			//bl2_obj_set_struc( BLIS_HERMITIAN, a );
			//bl2_obj_set_uplo( BLIS_LOWER, a );
			//bl2_obj_set_uplo( BLIS_UPPER, a );

			//bl2_printm( "a", &a, "%4.1f", "" );
			//bl2_printm( "b", &b, "%4.1f", "" );
			//bl2_printm( "c", &c, "%4.1f", "" );

			bl2_gemm_int( &BLIS_ONE,
			//bl2_gemm_int( &alpha,
			              &a,
			              &b,
			              &BLIS_ONE,
			              &c,
			              //gemm_cntl_op_bp );
			              gemm_cntl_mm_op );

			//bl2_copym( &c, &w );

			//bl2_printm( "c after", &c, "%4.1f", "" );
			//exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_HERMITIAN, a );
			bl2_obj_set_uplo( BLIS_LOWER, a );
			//bl2_obj_set_uplo( BLIS_UPPER, a );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bl2_gemm_int( &BLIS_ONE,
			              &a,
			              &b,
			              &BLIS_ONE,
			              &c,
			              //gemm_cntl_op_bp );
			              gemm_cntl_mm_op );
#endif

#if 0
			bl2_obj_set_struc( BLIS_HERMITIAN, c );
			bl2_obj_set_uplo( BLIS_UPPER, c );

			obj_t a1, c1, ar = a;
			bl2_obj_toggle_conj( ar );
			bl2_obj_toggle_trans( ar );

			dim_t bm = bl2_min( bl2_obj_length( a ), 128 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, bm, &a, &a1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, bm, &c, &c1 );


			dtime = bl2_clock();

			bl2_packm_init( &a1, &a_pack, packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a1, &a_pack, packm_cntl_a );

			bl2_packm_init( &ar, &b_pack, packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &ar, &b_pack, packm_cntl_b );

			//bl2_printm( "a", &a, "%4.1f", "" );
			//bl2_printm( "c", &c, "%4.1f", "" );
			bl2_herk_u_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ONE,
			                     &c1,
			                     NULL );
			//bl2_printm( "c after", &c, "%4.1f", "" );
			//exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_HERMITIAN, c );
			//bl2_obj_set_uplo( BLIS_LOWER, c );
			bl2_obj_set_uplo( BLIS_UPPER, c );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			obj_t ah;
			bl2_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, ah );

			//bl2_printm( "a", &a, "%4.1f", "" );
			//bl2_printm( "c", &c, "%4.1f", "" );
			bl2_herk_int( &BLIS_ONE,
			              &a,
			              &ah,
			              &BLIS_ONE,
			              &c,
			              herk_cntl_op_bp );
			              //herk_cntl_mm_op );
			              //herk_cntl_vl_mm );
			//bl2_printm( "c after", &c, "%4.1f", "" );
			//exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_HERMITIAN, c );
			//bl2_obj_set_uplo( BLIS_LOWER, c );
			bl2_obj_set_uplo( BLIS_UPPER, c );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			obj_t ah, bh;
			bl2_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, ah );
			bl2_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, b, bh );

			//bl2_printm( "a", &a, "%4.1f", "" );
			//bl2_printm( "b", &b, "%4.1f", "" );
			//bl2_printm( "c", &c, "%4.1f", "" );
/*
			bl2_her2k_int( &BLIS_ONE,
			               &a,
			               &bh,
			               &BLIS_ONE,
			               &b,
			               &ah,
			               &BLIS_ONE,
			               &c,
			               her2k_cntl_op_bp );
			               //her2k_cntl_mm_op );
			               //her2k_cntl_vl_mm );
*/

			bl2_herk_int( &BLIS_ONE,
			              &a,
			              &bh,
			              &BLIS_ONE,
			              &c,
			              herk_cntl_op_bp );
			bl2_herk_int( &BLIS_ONE,
			              &b,
			              &ah,
			              &BLIS_ONE,
			              &c,
			              herk_cntl_op_bp );

			//bl2_printm( "c after", &c, "%4.1f", "" );
			//exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_TRIANGULAR, a );
			bl2_obj_set_uplo( BLIS_LOWER, a );
			//bl2_obj_set_diag( BLIS_UNIT_DIAG, a );

			bl2_packm_init( &a, &a_pack, packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a, &a_pack, packm_cntl_a );

			bl2_packm_init( &c, &b_pack, packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &c, &b_pack, packm_cntl_b );


			bl2_printm( "a", &a, "%4.1f", "" );
			bl2_printm( "b", &c, "%4.1f", "" );

			bl2_trmm_l_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c,
			                     NULL );

			bl2_printm( "b after", &c, "%4.1f", "" );
			exit(1);
#endif

#if 0
			bl2_printm( "b", &c, "%4.1f", "" );
			bl2_printm( "a", &a, "%4.1f", "" );

			obj_t a1, a2;
			obj_t c1, c2;
			dim_t off1 = 0;
			dim_t off2 = 8;
			dim_t m1   = off2;
			dim_t m2   = m - m1;

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &a, &a1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &a, &a2 );

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &c, &c1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &c, &c2 );

			bl2_packm_init(           &c, &b_pack, trmm_packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &c, &b_pack, trmm_packm_cntl_b );

			bl2_packm_init(           &a1, &a_pack, trmm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a1, &a_pack, trmm_packm_cntl_a );

			bl2_trmm_u_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c1,
			                     NULL );

			bl2_packm_init(           &a2, &a_pack, trmm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a2, &a_pack, trmm_packm_cntl_a );

			bl2_trmm_u_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c2,
			                     NULL );

			bl2_printm( "b after", &c, "%4.1f", "" );

			exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_TRIANGULAR, a );
			bl2_obj_set_uplo( BLIS_LOWER, a );
			//bl2_obj_set_uplo( BLIS_UPPER, a );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			//bl2_printm( "a", &a, "%4.1f", "" );
			//bl2_printm( "b", &c, "%4.1f", "" );
			bl2_trmm_int( BLIS_LEFT,
			              &BLIS_ONE,
			              &a,
			              &c,
			              &BLIS_ZERO,
			              &c,
			              trmm_cntl_mm_op );
			              //trmm_cntl_vl_mm );
			//bl2_printm( "b after", &c, "%4.1f", "" );
			//exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_TRIANGULAR, a );
			//bl2_obj_set_uplo( BLIS_LOWER, a );
			bl2_obj_set_uplo( BLIS_UPPER, a );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bl2_printm( "a", &a, "%4.1f", "" );
			bl2_printm( "b", &b, "%4.1f", "" );
			bl2_printm( "c", &c, "%4.1f", "" );
			bl2_trmm_int( BLIS_LEFT,
			              &BLIS_ONE,
			              &a,
			              &b,
			              &BLIS_ONE,
			              &c,
			              trmm_cntl_mm_op );
			              //trmm_cntl_vl_mm );
			bl2_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif

#if 0
			bl2_printm( "b", &c, "%4.1f", "" );
			bl2_printm( "a", &a, "%4.1f", "" );

			obj_t a1, a2;
			obj_t c1, c2;
			dim_t off1 = 0;
			dim_t off2 = 8;
			dim_t m1   = off2;
			dim_t m2   = m - m1;

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &a, &a1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &a, &a2 );

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &c, &c1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &c, &c2 );

			bl2_packm_init(           &c, &b_pack, trsm_packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &c, &b_pack, trsm_packm_cntl_b );

			bl2_packm_init(           &a1, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a1, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_l_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c1,
			                     NULL );

			bl2_packm_init(           &a2, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a2, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_l_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c2,
			                     NULL );

			bl2_printm( "b after", &c, "%4.1f", "" );

			//bl2_unpackm_int( &b_pack, &d, trsm_unpackm_cntl_b );
			//bl2_printm( "d (b after unpack2)", &d, "%4.1f", "" );

			exit(1);
#endif

#if 0
			bl2_printm( "b", &c, "%4.1f", "" );
			bl2_printm( "a", &a, "%4.1f", "" );

			obj_t a1, a2;
			obj_t c1, c2;
			dim_t off1 = 0;
			dim_t off2 = 8;
			dim_t m1   = off2;
			dim_t m2   = m - m1;

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &a, &a1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &a, &a2 );

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &c, &c1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &c, &c2 );

			bl2_packm_init(           &c, &b_pack, trsm_packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &c, &b_pack, trsm_packm_cntl_b );

			bl2_packm_init(           &a2, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a2, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_u_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c2,
			                     NULL );

			bl2_packm_init(           &a1, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a1, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_u_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c1,
			                     NULL );


			bl2_printm( "b after", &c, "%4.1f", "" );
			//bl2_printm( "b after", &c, "%7.4f", "" );

			exit(1);
#endif


#if 0
			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );
/*
			obj_t a1, a2;
			obj_t c1, c2;
			dim_t off1 = 0;
			dim_t off2 = 128;
			dim_t m1   = off2;
			dim_t m2   = m - m1;

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &a, &a1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &a, &a2 );

			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off1, m1, &c, &c1 );
			bl2_acquire_mpart_t2b( BLIS_SUBPART1, off2, m2, &c, &c2 );

			bl2_packm_init(           &c, &b_pack, trsm_packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &c, &b_pack, trsm_packm_cntl_b );

			bl2_packm_init(           &a1, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a1, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_l_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c1,
			                     NULL );

			bl2_packm_init(           &a2, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a2, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_l_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c2,
			                     NULL );

			//bl2_unpackm_int( &b_pack, &d, trsm_unpackm_cntl_b );
*/
/*
			bl2_packm_init(           &c, &b_pack, trsm_packm_cntl_b );
			bl2_packm_int( &BLIS_ONE, &c, &b_pack, trsm_packm_cntl_b );

			bl2_packm_init(           &a, &a_pack, trsm_packm_cntl_a );
			bl2_packm_int( &BLIS_ONE, &a, &a_pack, trsm_packm_cntl_a );

			bl2_trsm_l_ker_var2( &BLIS_ONE,
			                     &a_pack,
			                     &b_pack,
			                     &BLIS_ZERO,
			                     &c,
			                     NULL );

			bl2_unpackm_int( &b_pack, &c, trsm_unpackm_cntl_b );
*/

			//bl2_printm( "d (b after unpack2)", &d, "%4.1f", "" );

			//exit(1);
#endif

#if 0
			bl2_obj_set_struc( BLIS_TRIANGULAR, a );
			bl2_obj_set_uplo( BLIS_LOWER, a );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bl2_trsm_int( BLIS_LEFT,
			              &BLIS_ONE,
			              &a,
			              &c,
			              &BLIS_ZERO,
			              &c,
			              trsm_cntl_op_bp );
#endif

#if 0
			//bl2_obj_set_struc( BLIS_TRIANGULAR, a );
			//bl2_obj_set_uplo( BLIS_LOWER, a );

			bl2_printm( "a", &a, "%4.1f", "" );
			bl2_printm( "b", &c, "%4.1f", "" );

			bl2_trsm_int( BLIS_LEFT,
			              &BLIS_ONE,
			              &a,
			              &c,
			              &BLIS_ZERO,
			              &c,
			              trsm_cntl_mm_op );
			              //trmm_cntl_vl_mm );

			bl2_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif


#if 0
			//bl2_obj_set_struc( BLIS_TRIANGULAR, a );
			//bl2_obj_set_uplo( BLIS_LOWER, a );

			bl2_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

			bl2_trsm_int( BLIS_LEFT,
			              &BLIS_ONE,
			              &a,
			              &c,
			              &BLIS_ZERO,
			              &c,
			              //trsm_cntl_op_bp );
			              trsm_cntl_mm_op );
			              //trmm_cntl_vl_mm );
#endif

#if 0
			char transa = 'N';
			char transb = 'N';
			int mm  = bl2_obj_length( c );
			int kk  = bl2_obj_width_after_trans( a );
			int nn  = bl2_obj_width( c );
			int lda  = bl2_obj_col_stride( a );
			int ldb  = bl2_obj_col_stride( b );
			int ldc  = bl2_obj_col_stride( c );
			dgemm_( &transa,
			        &transb,
			        &mm, &nn, &kk,
			        alpha.buffer,
			        a.buffer, &lda,
			        b.buffer, &ldb,
			        beta.buffer,
			        c.buffer, &ldc );
#endif









#if 0
			bl2_gemv( &alpha, &a, &b1, &beta, &c1 );
			//bl2_gemv_unf_var2( &alpha, &a, &b1, &beta, &c1, NULL );
			//bl2_gemv( &alpha, &a11, &b1, &beta, &c11 );
			//bl2_gemv_unf_var1( &alpha, &a, &b1, &beta, &c1, NULL );
			//bl2_gemv_unb_var2( &alpha, &a11, &b1, &beta, &c1, NULL );
			//bl2_gemv_unf_var2( &alpha, &a11, &b11, &beta, &c11, NULL );

			//bl2_gemv( &alpha, &a, &b, &beta, &c );
			//bl2_gemv_unb_var1( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_gemv_unf_var1( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_gemv_unb_var2( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_gemv_unf_var2( &alpha, &a, &b, &beta, &c, NULL );
//#else
			char trans = 'N';
			//char trans = 'T';
			//int mm  = bl2_obj_length( a11 );
			//int kk  = bl2_obj_width( a11 );
			//int lda  = bl2_obj_col_stride( a11 );
			int mm  = bl2_obj_length( a );
			int kk  = bl2_obj_width( a );
			int lda  = bl2_obj_col_stride( a );
			//int one = 1;
			int incb = bl2_obj_vector_inc( b1 );
			int incc = bl2_obj_vector_inc( c1 );
			dgemv_( &trans,
			        &mm, &kk,
			        alpha.buffer,
			        a.buffer, &lda,
			        b.buffer, &incb,
			        beta.buffer,
			        c.buffer, &incc );
#endif


#if 0
			//bl2_ger_unb_var1( &alpha, &a, &b, &c, NULL );
			//bl2_ger_unb_var2( &alpha, &a, &b, &c, NULL );
//#else
			int one = 1; int mm  = m; int nn  = n;
			dger_( &mm, &nn,
			       alpha.buffer,
			       a.buffer, &one,
			       b.buffer, &one,
			       c.buffer, &mm );
#endif


#if 0
			bl2_obj_set_uplo( BLIS_LOWER, a );
			//bl2_hemv_unb_var1( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_hemv_unb_var3( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_hemv_unf_var1a( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_hemv_unf_var3a( &alpha, &a, &b, &beta, &c, NULL );
			bl2_hemv_unf_var1( &alpha, &a, &b, &beta, &c, NULL );
			//bl2_hemv_unf_var3( &alpha, &a, &b, &beta, &c, NULL );
//#else
			int one = 1; int mm  = m; char uplo = 'L';
			dsymv_( &uplo,
			        &mm,
			        alpha.buffer,
			        a.buffer, &mm,
			        b.buffer, &one,
			        beta.buffer,
			        c.buffer, &one );
#endif


#if 0
			bl2_obj_set_uplo( BLIS_LOWER, c );
			bl2_her_unb_var2( &alpha, &a, &c, NULL );
//#else
			int one = 1; int mm  = m; char uplo = 'L';
			dsyr_( &uplo,
			       &mm,
			       alpha.buffer,
			       a.buffer, &one,
			       c.buffer, &mm );
#endif


#if 0
			bl2_obj_set_uplo( BLIS_LOWER, c );
			//bl2_her2( &alpha, &a, &b, &c );
			//bl2_her2_unb_var1( &alpha, &a, &b, &c, NULL );
			//bl2_her2_unb_var4( &alpha, &a, &b, &c, NULL );
			//bl2_her2_unf_var1( &alpha, &a, &b, &c, NULL );
			//bl2_her2_unf_var4( &alpha, &a, &b, &c, NULL );
//#else
			int one = 1; int mm = bl2_obj_col_stride( c ); char uplo = 'L';
			dsyr2_( &uplo,
			        &mm,
			        alpha.buffer,
			        a.buffer, &one,
			        b.buffer, &one,
			        c.buffer, &mm );
#endif


#if 0
			bl2_setsc(  (1.0/1.0), &alpha );
			bl2_obj_set_uplo( BLIS_LOWER, c );
			bl2_obj_set_conjtrans( BLIS_NO_TRANSPOSE, c );
			//bl2_trmv( &alpha, &c, &a );
			//bl2_trmv_unb_var1( &alpha, &c, &a, NULL );
			//bl2_trmv_unb_var2( &alpha, &c, &a, NULL );
			bl2_trmv_unf_var1( &alpha, &c, &a, NULL );
			//bl2_trmv_unf_var2( &alpha, &c, &a, NULL );
//#else
			int one = 1; int mm  = m; char uplo = 'U'; char trans = 'T'; char diag = 'N';
			dtrmv_( &uplo,
			        &trans,
			        &diag,
			        &mm,
			        c.buffer, &mm,
			        a.buffer, &one );
#endif


#if 0
			bl2_setsc(  (1.0/1.0), &alpha );
			bl2_obj_set_uplo( BLIS_UPPER, c );
			bl2_obj_set_conjtrans( BLIS_NO_TRANSPOSE, c );
			//bl2_trsv( &alpha, &c, &a );
			//bl2_trsv_unb_var1( &alpha, &c, &a, NULL );
			//bl2_trsv_unb_var2( &alpha, &c, &a, NULL );
			bl2_trsv_unf_var1( &alpha, &c, &a, NULL );
			//bl2_trsv_unf_var2( &alpha, &c, &a, NULL );
//#else
			int one = 1; int mm  = m; char uplo = 'L'; char trans = 'T'; char diag = 'N';
			dtrsv_( &uplo,
			        &trans,
			        &diag,
			        &mm,
			        c.buffer, &mm,
			        a.buffer, &one );
#endif

			//obj_t c_part;
			//dim_t ij = 3;
			//dim_t bb = 2;
			//bl2_acquire_mpart_tl2br( BLIS_SUBPART22,
			//                         ij,
			//                         bb,
			//                         &c,
			//                         &c_part );
			//bl2_printm( "c_part", &c_part, "%8.1e", "" );

			dtime = bl2_clock() - dtime;

			dtime_save = bl2_min( dtime, dtime_save );

		}

		//bl2_printm( "c after", &c, "%8.1e", "" );
		gflops = ( 2.0 * m * k * n ) / ( dtime_save * 1.0e9 );
		//gflops = ( 2.0 * a_pack.m * k * n - a_pack.m * a_pack.m * k ) / ( dtime_save * 1.0e9 );
		if ( bl2_obj_is_hermitian( c ) || bl2_obj_is_triangular( a ) ) gflops /= 2.0;
		//gflops /= 4.0;

		//gflops = ( 2.0 * bl2_obj_length( a11 ) * bl2_obj_width( a11 ) ) / ( dtime_save * 1.0e9 );
		//gflops = ( 2.0 * bl2_obj_length( a ) * bl2_obj_width( a ) ) / ( dtime_save * 1.0e9 );

		printf( "data_blis( %2ld, 1:5 ) = [ %4lu %4lu %4lu  %10.3e  %6.3f ];\n",
		        (p - p_begin + 1)/p_inc + 1, m, k, n, dtime_save, gflops );

		bl2_obj_release_pack( &a_pack );
		bl2_obj_release_pack( &b_pack );
		bl2_obj_release_pack( &c_pack );

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
		bl2_cntl_obj_free( trmm_packm_cntl_a );
		bl2_cntl_obj_free( trmm_packm_cntl_b );
		bl2_cntl_obj_free( trsm_packm_cntl_a );
		bl2_cntl_obj_free( trsm_packm_cntl_b );
		bl2_cntl_obj_free( trsm_unpackm_cntl_b );
		bl2_cntl_obj_free( gemm_cntl_bp_ke );
		bl2_cntl_obj_free( gemm_cntl_op_bp );
		bl2_cntl_obj_free( gemm_cntl_mm_op );
		bl2_cntl_obj_free( gemm_cntl_vl_mm );
		bl2_cntl_obj_free( herk_cntl_bp_ke );
		bl2_cntl_obj_free( herk_cntl_op_bp );
		bl2_cntl_obj_free( herk_cntl_mm_op );
		bl2_cntl_obj_free( herk_cntl_vl_mm );
		bl2_cntl_obj_free( her2k_cntl_bp_ke );
		bl2_cntl_obj_free( her2k_cntl_op_bp );
		bl2_cntl_obj_free( her2k_cntl_mm_op );
		bl2_cntl_obj_free( her2k_cntl_vl_mm );
		bl2_cntl_obj_free( trmm_cntl_bp_ke );
		bl2_cntl_obj_free( trmm_cntl_op_bp );
		bl2_cntl_obj_free( trmm_cntl_mm_op );
		bl2_cntl_obj_free( trmm_cntl_vl_mm );
		bl2_cntl_obj_free( trmm3_cntl_mm_op );
		bl2_cntl_obj_free( trsm_cntl_bp_ke );
		bl2_cntl_obj_free( trsm_cntl_op_bp );
		bl2_cntl_obj_free( trsm_cntl_mm_op );
		bl2_cntl_obj_free( trsm_cntl_vl_mm );

		bl2_obj_free( &alpha );
		bl2_obj_free( &beta );

		bl2_obj_free( &a );
		bl2_obj_free( &b );
		bl2_obj_free( &c );
		bl2_obj_free( &w );
		bl2_obj_free( &c_save );
	}

	bl2_finalize();

	return 0;
}
#endif




// -- Level 1d -----------------------------------------------------------------



#if 0
{
	num_t dt_alpha, dt_a, dt_b;
	obj_t alpha, a, b;
	dim_t m, n;

	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_b     = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
	bl2_obj_create( dt_a,     n, m, 0, 0, &a );
	bl2_obj_create( dt_b,     m, n, 0, 0, &b );

	bl2_setsc( -(3.0/1.0), &alpha );
	bl2_setm( &BLIS_TWO, &a );
	bl2_setm( &BLIS_ONE, &b );

	bl2_printm( "alpha", &alpha, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );
	bl2_printm( "b before", &b, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	bl2_obj_set_struc( BLIS_GENERAL, a );
	bl2_obj_set_uplo( BLIS_DENSE, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( -7, a );
	bl2_obj_set_diag( BLIS_UNIT_DIAG, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	bl2_axpyd( &alpha, &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );


	bl2_obj_free( &alpha );
	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_a, dt_b;
	obj_t a, b;
	dim_t m, n;

	dt_a     = BLIS_DOUBLE;
	dt_b     = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_a,     n, m, 0, 0, &a );
	bl2_obj_create( dt_b,     m, n, 0, 0, &b );

	bl2_setm( &BLIS_TWO, &a );
	bl2_setm( &BLIS_ZERO, &b );

	bl2_printm( "a before", &a, "%4.1f", "" );
	bl2_printm( "b before", &b, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	bl2_obj_set_struc( BLIS_GENERAL, a );
	bl2_obj_set_uplo( BLIS_DENSE, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( -7, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	bl2_copyd( &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );


	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_beta, dt_a, dt_b;
	obj_t beta, a, b;
	dim_t m, n;

	dt_beta  = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_b     = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_beta, 1, 1, 0, 0, &beta );
	bl2_obj_create( dt_a,    n, m, 0, 0, &a );
	bl2_obj_create( dt_b,    m, n, 0, 0, &b );

	bl2_setsc( -(3.0/1.0), &beta );
	bl2_setm( &BLIS_MINUS_TWO, &a );
	bl2_setm( &BLIS_ONE, &b );

	bl2_printm( "beta", &beta, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );
	bl2_printm( "b before", &b, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	bl2_obj_set_struc( BLIS_GENERAL, a );
	bl2_obj_set_uplo( BLIS_DENSE, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( 2, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	bl2_scal2d( &beta, &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );


	bl2_obj_free( &beta );
	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_a, dt_alpha;
	obj_t alpha, a;
	dim_t m, n;

	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
	bl2_obj_create( dt_a,     m, n, 0, 0, &a );

	bl2_setsc( -(4.0/1.0), &alpha );
	bl2_setm( &BLIS_TWO, &a );

	bl2_printm( "alpha", &alpha, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	bl2_obj_set_struc( BLIS_GENERAL, a );
	bl2_obj_set_uplo( BLIS_DENSE, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( -2, a );
	bl2_obj_set_diag( BLIS_UNIT_DIAG, a );

	bl2_scald( &alpha, &a );

	bl2_printm( "a after", &a, "%4.1f", "" );

	bl2_obj_free( &alpha );
	bl2_obj_free( &a );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_a, dt_alpha;
	obj_t alpha, a;
	dim_t m, n;

	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
	bl2_obj_create( dt_a,     m, n, 0, 0, &a );

	bl2_setsc( -(4.0/1.0), &alpha );
	bl2_setm( &BLIS_TWO, &a );

	bl2_printm( "alpha", &alpha, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	bl2_obj_set_struc( BLIS_GENERAL, a );
	bl2_obj_set_uplo( BLIS_DENSE, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( 7, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );

	bl2_setd( &alpha, &a );

	bl2_printm( "a after", &a, "%4.1f", "" );

	bl2_obj_free( &alpha );
	bl2_obj_free( &a );

	bl2_finalize();

	return 0;
}
#endif



// -- Level 1m -----------------------------------------------------------------



#if 0
{
	num_t dt_alpha, dt_a, dt_b;
	obj_t alpha, a, b;
	dim_t m, n;

	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_b     = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
	bl2_obj_create( dt_a,    n, m, 0, 0, &a );
	bl2_obj_create( dt_b,    m, n, 0, 0, &b );

	bl2_setsc( -(3.0/1.0), &alpha );
	bl2_setm( &BLIS_TWO, &a );
	bl2_setm( &BLIS_ONE, &b );

	bl2_printm( "alpha", &alpha, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );
	bl2_printm( "b before", &b, "%4.1f", "" );

	bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( 2, a );
	bl2_obj_set_diag( BLIS_UNIT_DIAG, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	bl2_axpym( &alpha, &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );

	bl2_obj_free( &alpha );
	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_a, dt_b;
	obj_t a, b;
	dim_t m, n;

	dt_a     = BLIS_DOUBLE;
	dt_b     = BLIS_DOUBLE;

	m = 11;
	n = 6;
	
	bl2_init();

	bl2_obj_create( dt_a,     n, m, 0, 0, &a );
	bl2_obj_create( dt_b,     m, n, 0, 0, &b );

	//bl2_setm( &BLIS_MINUS_ONE, &a );
	bl2_randm( &a );
	bl2_setm( &BLIS_ZERO, &b );

	bl2_printm( "a before", &a, "%4.1f", "" );
	bl2_printm( "b before", &b, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );

	bl2_obj_set_diag_offset( 2, a );
	bl2_obj_set_struc( BLIS_TRIANGULAR, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );

	bl2_copym( &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );

	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_a, dt_alpha;
	obj_t alpha, a;
	dim_t m, n;

	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;

	m = 6;
	n = 1;
	
	bl2_init();

	bl2_obj_create( dt_a,     m, n, 0, 0, &a );
	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

	bl2_setm( &BLIS_ONE, &a );
	bl2_setsc( -(4.0/1.0), &alpha );

	bl2_printm( "alpha", &alpha, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );

	bl2_obj_set_diag_offset( -7, a );
	bl2_obj_set_struc( BLIS_TRIANGULAR, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );

	bl2_scalm( &alpha, &a );

	bl2_printm( "a after", &a, "%4.1f", "" );

	bl2_obj_free( &a );
	bl2_obj_free( &alpha );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t dt_beta, dt_a, dt_b;
	obj_t beta, a, b;
	dim_t m, n;

	dt_beta = BLIS_DOUBLE;
	dt_a    = BLIS_DOUBLE;
	dt_b    = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_beta, 1, 1, 0, 0, &beta );
	bl2_obj_create( dt_a,    n, m, 0, 0, &a );
	bl2_obj_create( dt_b,    m, n, 0, 0, &b );

	bl2_setsc( -(3.0/1.0), &beta );
	bl2_setm( &BLIS_TWO, &a );
	bl2_setm( &BLIS_ONE, &b );

	bl2_printm( "beta", &beta, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );
	bl2_printm( "b before", &b, "%4.1f", "" );

	bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( -2, a );
	bl2_obj_set_diag( BLIS_UNIT_DIAG, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	bl2_scal2m( &beta, &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );

	bl2_obj_free( &beta );
	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t    dt_beta, dt_a;
	obj_t    beta, a, p;
	dim_t    m, n;
	blksz_t* mult_m;
	blksz_t* mult_n;
	packm_t* cntl;

	dt_beta = BLIS_DOUBLE;
	dt_a    = BLIS_DOUBLE;

	m = 6;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_beta, 1, 1, 0, 0, &beta );
	bl2_obj_create( dt_a,    n, m, 0, 0, &a );

	bl2_setsc( -(1.0/1.0), &beta );
	//bl2_setm( &BLIS_TWO, &a );
	bl2_randm( &a );

	bl2_printm( "beta", &beta, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	bl2_obj_set_struc( BLIS_TRIANGULAR, a );
	//bl2_obj_set_struc( BLIS_GENERAL, a );
	//bl2_obj_set_uplo( BLIS_DENSE, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_diag_offset( -2, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );
	bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	mult_m = bl2_blksz_obj_create( 2, 4, 2, 2 );
	mult_n = bl2_blksz_obj_create( 1, 2, 1, 1 );

	cntl = bl2_packm_cntl_obj_create( BLIS_UNBLOCKED,
	                                  BLIS_VARIANT1,
	                                  mult_m,
	                                  mult_n, 
	                                  //TRUE,  // scale?
	                                  FALSE, // scale?
	                                  //FALSE, // densify?
	                                  TRUE,  // densify?
	                                  BLIS_PACKED_ROWS );
	                                  //BLIS_PACKED_COLUMNS );
	bl2_obj_init_pack( &p );

	bl2_packm_int( &beta, &a, &p, cntl );

	bl2_printm( "p after", &p, "%4.1f", "" );
p.m = p.pack_mem->m;
p.n = p.pack_mem->n;
	bl2_printm( "p mem", &p, "%4.1f", "" );

	bl2_obj_release_pack( &p );

	bl2_obj_free( &beta );
	bl2_obj_free( &a );

	bl2_blksz_obj_free( mult_m );
	bl2_blksz_obj_free( mult_n );
	bl2_cntl_obj_free( cntl );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	num_t    dt_beta, dt_a;
	obj_t    beta, a, p;
	dim_t    m, n;
	blksz_t* mult_m;
	blksz_t* mult_n;
	packm_t* cntl;

	dt_beta = BLIS_DOUBLE;
	dt_a    = BLIS_DOUBLE;

	m = 7;
	n = 11;
	
	bl2_init();

	bl2_obj_create( dt_beta, 1, 1, 0, 0, &beta );
	bl2_obj_create( dt_a,    m, n, 0, 0, &a );

	bl2_setsc( -(1.0/1.0), &beta );
	//bl2_setm( &BLIS_TWO, &a );
	bl2_randm( &a );

	bl2_printm( "beta", &beta, "%4.1f", "" );
	bl2_printm( "a before", &a, "%4.1f", "" );

	//bl2_obj_set_struc( BLIS_SYMMETRIC, a );
	//bl2_obj_set_struc( BLIS_TRIANGULAR, a );
	bl2_obj_set_struc( BLIS_GENERAL, a );
	bl2_obj_set_uplo( BLIS_DENSE, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	//bl2_obj_set_diag_offset( -2, a );
	bl2_obj_set_diag_offset( 0, a );
	//bl2_obj_set_diag( BLIS_UNIT_DIAG, a );
	//bl2_obj_set_trans( BLIS_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, a );

	mult_m = bl2_blksz_obj_create( 2, 4, 2, 2 );
	mult_n = bl2_blksz_obj_create( 1, 2, 1, 1 );

	cntl = bl2_packm_cntl_obj_create( BLIS_BLOCKED,
	                                  BLIS_VARIANT1,
	                                  mult_m,
	                                  mult_n, 
	                                  //TRUE,  // scale?
	                                  FALSE, // scale?
	                                  //FALSE, // densify?
	                                  TRUE,  // densify?
	                                  BLIS_PACKED_COL_PANELS );
	bl2_obj_init_pack( &p );

	bl2_packm_int( &beta, &a, &p, cntl );

p.m = p.pack_mem->m;
p.n = p.pack_mem->n;
p.rs = 1;
p.cs = p.pack_mem->m;
	bl2_printm( "p mem", &p, "%4.1f", "" );

	bl2_obj_release_pack( &p );

	bl2_obj_free( &beta );
	bl2_obj_free( &a );

	bl2_blksz_obj_free( mult_m );
	bl2_blksz_obj_free( mult_n );
	bl2_cntl_obj_free( cntl );

	bl2_finalize();

	return 0;
}
#endif



// -- Level 2 ------------------------------------------------------------------



#if 0
{
	obj_t alpha, beta, a, x, y, xK, yK;
	num_t dt_alpha, dt_beta, dt_a, dt_x, dt_y;
	dim_t m, n, K;

	//dt_alpha = BLIS_DCOMPLEX;
	//dt_beta  = BLIS_DCOMPLEX;
	//dt_a     = BLIS_DCOMPLEX;
	//dt_x     = BLIS_DCOMPLEX;
	//dt_y     = BLIS_DCOMPLEX;
	dt_alpha = BLIS_DOUBLE;
	dt_beta  = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_x     = BLIS_DOUBLE;
	dt_y     = BLIS_DOUBLE;

	m = 11;
	n = 13;
	K = 4;

	bl2_init();

	//bl2_obj_create( dt_a, m, n, n, 1, &a );
	//bl2_obj_create( dt_a, m, n, 0, 0, &a );
	bl2_obj_create( dt_a, m, n, 0, 0, &a );
	bl2_obj_create( dt_x, K, n, 0, 0, &xK );
	bl2_obj_create( dt_y, K, m, 0, 0, &yK );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );
	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &yK, &y );

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
	bl2_obj_create( dt_beta,  1, 1, 0, 0, &beta );

	bl2_setm( &BLIS_ZERO, &a );
	bl2_setm( &BLIS_ZERO, &x );
	bl2_setm( &BLIS_ZERO, &y );
	bl2_randm( &a );
	bl2_randm( &x );
	bl2_randm( &y );

	bl2_setsc(  (2.0/1.0), &alpha );
	bl2_setsc( -(1.0/1.0), &beta );
	//bl2_setsc(  (1.0/1.0), &alpha );
	//bl2_setsc(  (0.0/1.0), &beta );

	bl2_printm( "a", &a, "%4.1f", "" );
	bl2_printm( "x", &x, "%4.1f", "" );
	bl2_printm( "y", &y, "%4.1f", "" );

	//bl2_obj_set_conjtrans( BLIS_CONJ_NO_TRANSPOSE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, x );

	bl2_gemv( &alpha, &a, &x, &beta, &y );
	//bl2_gemv_unb_var1( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_gemv_unf_var1( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_gemv_unb_var2( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_gemv_unf_var2( &alpha, &a, &x, &beta, &y, NULL );

	bl2_printm( "y after", &y, "%4.1f", "" );

	bl2_obj_free( &a );
	bl2_obj_free( &xK );
	bl2_obj_free( &yK );
	bl2_obj_free( &alpha );
	bl2_obj_free( &beta );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	obj_t alpha, a, x, y, xK, yK;
	num_t dt_alpha, dt_a, dt_x, dt_y;
	dim_t m, n, K;

	//dt_alpha = BLIS_DCOMPLEX;
	//dt_a = BLIS_DCOMPLEX;
	//dt_x = BLIS_DCOMPLEX;
	//dt_y = BLIS_DCOMPLEX;
	dt_alpha = BLIS_DOUBLE;
	dt_a = BLIS_DOUBLE;
	dt_x = BLIS_DOUBLE;
	dt_y = BLIS_DOUBLE;

	m = 11;
	n = 13;
	K = 4;

	bl2_init();


	bl2_obj_create( dt_a, m, n, 0, 0, &a );
	//bl2_obj_create( dt_a, m, n, n, 1, &a );
	bl2_obj_create( dt_x, K, m, 0, 0, &xK );
	bl2_obj_create( dt_y, K, n, 0, 0, &yK );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );
	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &yK, &y );

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

	bl2_setm( &BLIS_ZERO, &a );
	bl2_setm( &BLIS_ZERO, &x );
	bl2_setm( &BLIS_ZERO, &y );
	bl2_randm( &a );
	bl2_randm( &x );
	bl2_randm( &y );

	bl2_setsc(  (2.0/1.0), &alpha );

	bl2_printm( "a", &a, "%4.1f", "" );
	bl2_printm( "x", &x, "%4.1f", "" );
	bl2_printm( "y", &y, "%4.1f", "" );

	bl2_obj_set_conj( BLIS_NO_CONJUGATE, x );
	bl2_obj_set_conj( BLIS_NO_CONJUGATE, y );

	bl2_ger( &alpha, &x, &y, &a );
	//bl2_ger_unb_var1( &alpha, &x, &y, &a, NULL );
	//bl2_ger_unb_var2( &alpha, &x, &y, &a, NULL );

	bl2_printm( "a after", &a, "%4.1f", "" );

	bl2_obj_free( &a );
	bl2_obj_free( &xK );
	bl2_obj_free( &yK );
	bl2_obj_free( &alpha );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	obj_t alpha, beta, a, x, y, xK, yK;
	num_t dt_alpha, dt_beta, dt_a, dt_x, dt_y;
	dim_t m, K;

	//dt_alpha = BLIS_DCOMPLEX;
	//dt_beta  = BLIS_DCOMPLEX;
	//dt_a     = BLIS_DCOMPLEX;
	//dt_x     = BLIS_DCOMPLEX;
	//dt_y     = BLIS_DCOMPLEX;
	dt_alpha = BLIS_DOUBLE;
	dt_beta  = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_x     = BLIS_DOUBLE;
	dt_y     = BLIS_DOUBLE;

	m = 11;
	K = 4;

	bl2_init();

	bl2_obj_create( dt_a, m, m, 0, 0, &a );
	//bl2_obj_create( dt_a, m, m, m, 1, &a );
	bl2_obj_create( dt_x, K, m, 0, 0, &xK );
	bl2_obj_create( dt_y, K, m, 0, 0, &yK );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );
	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &yK, &y );

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );
	bl2_obj_create( dt_beta,  1, 1, 0, 0, &beta );

	bl2_setm( &BLIS_ZERO, &a );
	bl2_setm( &BLIS_ZERO, &x );
	bl2_setm( &BLIS_ZERO, &y );
	bl2_randm( &a );
	bl2_randm( &x );
	bl2_randm( &y );

	bl2_setsc(  (2.0/1.0), &alpha );
	bl2_setsc( -(1.0/1.0), &beta );
	//bl2_setsc(  (1.0/1.0), &alpha );
	//bl2_setsc(  (1.0/1.0), &beta );

	bl2_printm( "a", &a, "%4.1f", "" );
	bl2_printm( "x", &x, "%4.1f", "" );
	bl2_printm( "y", &y, "%4.1f", "" );

	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	bl2_obj_set_conj( BLIS_NO_CONJUGATE, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, x );

	bl2_hemv( &alpha, &a, &x, &beta, &y );

	//bl2_hemv_unb_var1( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_hemv_unb_var2( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_hemv_unb_var3( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_hemv_unb_var4( &alpha, &a, &x, &beta, &y, NULL );

	//bl2_hemv_unf_var1a( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_hemv_unf_var1( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_hemv_unf_var3a( &alpha, &a, &x, &beta, &y, NULL );
	//bl2_hemv_unf_var3( &alpha, &a, &x, &beta, &y, NULL );

	bl2_printm( "y after", &y, "%4.1f", "" );

	bl2_obj_free( &a );
	bl2_obj_free( &xK );
	bl2_obj_free( &yK );
	bl2_obj_free( &alpha );
	bl2_obj_free( &beta );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	obj_t alpha, a, x, xK;
	num_t dt_alpha, dt_a, dt_x;
	dim_t m, K;

	//dt_x     = BLIS_DCOMPLEX;
	//dt_a     = BLIS_DCOMPLEX;
	dt_x     = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_alpha = bl2_datatype_proj_to_real( dt_x );

	m = 13;
	K = 4;

	bl2_init();

	bl2_obj_create( dt_x, K, m, 0, 0, &xK );
	//bl2_obj_create( dt_a, m, m, 0, 0, &a );
	bl2_obj_create( dt_a, m, m, m, 1, &a );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

	bl2_setm( &BLIS_ZERO, &x );
	bl2_setm( &BLIS_ZERO, &a );
	bl2_randm( &x );
	bl2_randm( &a );

	bl2_setsc(  (2.0/1.0), &alpha );
	//bl2_setsc(  (1.0/1.0), &alpha );

	bl2_printm( "x", &x, "%4.1f", "" );
	bl2_printm( "a", &a, "%4.1f", "" );

	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, x );

	bl2_her( &alpha, &x, &a );

	//bl2_her_unb_var1( &alpha, &x, &a, NULL );
	//bl2_her_unb_var2( &alpha, &x, &a, NULL );

	bl2_printm( "a after", &a, "%4.1f", "" );

	bl2_obj_free( &xK );
	bl2_obj_free( &a );
	bl2_obj_free( &alpha );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	obj_t alpha, a, x, y, xK, yK;
	num_t dt_alpha, dt_a, dt_x, dt_y;
	dim_t m, K;

	//dt_x     = BLIS_DCOMPLEX;
	//dt_y     = BLIS_DCOMPLEX;
	//dt_a     = BLIS_DCOMPLEX;
	//dt_alpha = BLIS_DCOMPLEX;
	dt_x     = BLIS_DOUBLE;
	dt_y     = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_alpha = BLIS_DOUBLE;

	m = 13;
	K = 4;

	bl2_init();

	bl2_obj_create( dt_x, K, m, 0, 0, &xK );
	bl2_obj_create( dt_y, K, m, 0, 0, &yK );
	//bl2_obj_create( dt_a, m, m, m, 1, &a );
	bl2_obj_create( dt_a, m, m, 0, 0, &a );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );
	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &yK, &y );

	bl2_obj_create( dt_alpha, 1, 1, 0, 0, &alpha );

	bl2_setm( &BLIS_ZERO, &x );
	bl2_setm( &BLIS_ZERO, &y );
	bl2_setm( &BLIS_ZERO, &a );
	bl2_randm( &x );
	bl2_randm( &y );
	bl2_randm( &a );

	//bl2_randv( &alpha );
	bl2_setsc(  (2.0/1.0), &alpha );
	//bl2_setsc(  (1.0/1.0), &alpha );

	bl2_printm( "x", &x, "%4.1f", "" );
	bl2_printm( "y", &y, "%4.1f", "" );
	bl2_printm( "a", &a, "%4.1f", "" );
	//bl2_printv( "alpha", &alpha, "%4.1f", "" );

	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_conj( BLIS_CONJUGATE, x );

	bl2_her2( &alpha, &x, &y, &a );

	//bl2_her2_unb_var1( &alpha, &x, &y, &a, NULL );
	//bl2_her2_unb_var2( &alpha, &x, &y, &a, NULL );
	//bl2_her2_unb_var3( &alpha, &x, &y, &a, NULL );
	//bl2_her2_unb_var4( &alpha, &x, &y, &a, NULL );
	//bl2_her2_unf_var4( &alpha, &x, &y, &a, NULL );

	bl2_printm( "a after", &a, "%4.1f", "" );

	bl2_obj_free( &xK );
	bl2_obj_free( &yK );
	bl2_obj_free( &a );
	bl2_obj_free( &alpha );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	obj_t alpha, a, x, xK;
	num_t dt_alpha, dt_a, dt_x;
	dim_t m, K;

	//dt_alpha = BLIS_DCOMPLEX;
	//dt_a     = BLIS_DCOMPLEX;
	//dt_x     = BLIS_DCOMPLEX;
	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_x     = BLIS_DOUBLE;

	m = 13;
	K = 4;

	bl2_init();

	//bl2_obj_create( dt_a, m, m, 0, 0, &a );
	bl2_obj_create( dt_a, m, m, m, 1, &a );
	bl2_obj_create( dt_x, K, m, 0, 0, &xK );
	bl2_obj_create( dt_x, 1, 1, 0, 0, &alpha );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );

	bl2_setm( &BLIS_ZERO, &a );
	bl2_setm( &BLIS_ZERO, &x );
	bl2_randm( &a );
	bl2_randm( &x );

	bl2_setsc(  (1.0/1.0), &alpha );

	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_conjtrans( BLIS_NO_TRANSPOSE, a );
	//bl2_obj_set_conjtrans( BLIS_TRANSPOSE, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );

	bl2_printm( "a", &a, "%4.1f", "" );
	bl2_printm( "x", &x, "%4.1f", "" );

	bl2_trmv( &alpha, &a, &x );

	//bl2_trmv_unb_var2( &alpha, &a, &x, NULL );
	//bl2_trmv_unb_var1( &alpha, &a, &x, NULL );
	//bl2_trmv_unf_var1( &alpha, &a, &x, NULL );
	//bl2_trmv_unf_var2( &alpha, &a, &x, NULL );

	bl2_printm( "x after", &x, "%4.1f", "" );

	bl2_obj_free( &alpha );
	bl2_obj_free( &a );
	bl2_obj_free( &xK );

	bl2_finalize();

	return 0;
}
#endif

#if 0
{
	obj_t alpha, a, x, xK;
	num_t dt_alpha, dt_a, dt_x;
	dim_t m, K;

	dt_alpha = BLIS_DOUBLE;
	dt_a     = BLIS_DOUBLE;
	dt_x     = BLIS_DOUBLE;

	m = 13;
	K = 4;

	bl2_init();

	//bl2_obj_create( dt_a, m, m, 0, 0, &a );
	bl2_obj_create( dt_a, m, m, m, 1, &a );
	bl2_obj_create( dt_x, K, m, 0, 0, &xK );
	bl2_obj_create( dt_x, 1, 1, 0, 0, &alpha );

	bl2_acquire_mpart_t2b( BLIS_SUBPART1, 0, 1, &xK, &x );

	bl2_setm( &BLIS_ZERO, &a );
	bl2_setm( &BLIS_ZERO, &x );
	bl2_randm( &a );
	bl2_randm( &x );

	bl2_setsc(  (2.0/1.0), &alpha );

	bl2_obj_set_uplo( BLIS_UPPER, a );
	//bl2_obj_set_uplo( BLIS_LOWER, a );
	bl2_obj_set_conjtrans( BLIS_NO_TRANSPOSE, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );

	bl2_printm( "a", &a, "%4.1f", "" );
	bl2_printm( "x", &x, "%4.1f", "" );

	bl2_trsv( &alpha, &a, &x );
	//bl2_trsv_unb_var1( &alpha, &a, &x, NULL );
	//bl2_trsv_unb_var2( &alpha, &a, &x, NULL );
	//bl2_trsv_unf_var1( &alpha, &a, &x, NULL );
	//bl2_trsv_unf_var2( &alpha, &a, &x, NULL );

	bl2_printm( "x after", &x, "%4.1f", "" );

	bl2_obj_free( &alpha );
	bl2_obj_free( &a );
	bl2_obj_free( &xK );

	bl2_finalize();

	return 0;
}
#endif



// -- Level 3 ------------------------------------------------------------------



#if 0
{
	obj_t a, b;
	obj_t alpha;
	num_t dt_a, dt_b;
	num_t dt_al;
	dim_t m, n;

	dt_a  = BLIS_DCOMPLEX;
	dt_b  = BLIS_DCOMPLEX;
	dt_al = BLIS_DCOMPLEX;

	m = 9;
	n = 5;

	bl2_init();

	bl2_obj_create( dt_a,  m, m, 0, 0, &a );
	//bl2_obj_create( dt_b,  m, n, 0, 0, &b );
	bl2_obj_create( dt_b,  n, m, 0, 0, &b );
	bl2_obj_create( dt_al, 1, 1, 0, 0, &alpha );

	bl2_setm( &BLIS_ZERO, &a );
	bl2_setm( &BLIS_ZERO, &b );
	bl2_randm( &a );
	bl2_randm( &b );

	//bl2_setsc(  (2.0/1.0), &alpha );
	bl2_setsc(  (1.0/1.0), &alpha );

	bl2_obj_set_uplo( BLIS_LOWER, a );
	//bl2_obj_set_uplo( BLIS_UPPER, a );
	bl2_obj_set_conjtrans( BLIS_TRANSPOSE, a );
	bl2_obj_set_diag( BLIS_NONUNIT_DIAG, a );

	bl2_printm( "a", &a, "%4.1f", "" );
	bl2_printm( "b", &b, "%4.1f", "" );

	//bl2_trmm_lu_unb_var3( &alpha, &a, &b, NULL );
	//bl2_trmm_lu_unb_var2( &alpha, &a, &b, NULL );
	//bl2_trmm_lu_unb_var1( &alpha, &a, &b, NULL );
	//bl2_trmm_ll_unb_var3( &alpha, &a, &b, NULL );
	//bl2_trmm_ll_unb_var2( &alpha, &a, &b, NULL );
	//bl2_trmm_ll_unb_var1( &alpha, &a, &b, NULL );
	//bl2_trmm( BLIS_LEFT, &alpha, &a, &b );
	bl2_trmm( BLIS_RIGHT, &alpha, &a, &b );

	bl2_printm( "b after", &b, "%4.1f", "" );

	bl2_obj_free( &a );
	bl2_obj_free( &b );

	bl2_finalize();

	return 0;
}
#endif


}
