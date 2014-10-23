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

#include "blis.h"


// NOTE: The values handled here may be queried even before bli_init()
// is called!



// -- General library information ----------------------------------------------

// This string gets defined via -D on the command line when BLIS is compiled.
// This string is (or rather, should be) only used here.
static char* bli_version_str       = BLIS_VERSION_STRING;
static char* bli_int_type_size_str = STRINGIFY_INT( BLIS_INT_TYPE_SIZE );

char* bli_info_get_version_str( void )              { return bli_version_str; }
char* bli_info_get_int_type_size_str( void )        { return bli_int_type_size_str; }



// -- bli_config.h -------------------------------------------------------------

gint_t bli_info_get_int_type_size( void )           { return BLIS_INT_TYPE_SIZE; }
gint_t bli_info_get_num_fp_types( void )            { return BLIS_NUM_FP_TYPES; }
gint_t bli_info_get_max_type_size( void )           { return BLIS_MAX_TYPE_SIZE; }
gint_t bli_info_get_max_num_threads( void )         { return BLIS_MAX_NUM_THREADS; }
gint_t bli_info_get_num_mc_x_kc_blocks( void )      { return BLIS_NUM_MC_X_KC_BLOCKS; }
gint_t bli_info_get_num_kc_x_nc_blocks( void )      { return BLIS_NUM_KC_X_NC_BLOCKS; }
gint_t bli_info_get_num_mc_x_nc_blocks( void )      { return BLIS_NUM_MC_X_NC_BLOCKS; }
gint_t bli_info_get_max_preload_byte_offset( void ) { return BLIS_MAX_PRELOAD_BYTE_OFFSET; }
gint_t bli_info_get_simd_align_size( void )         { return BLIS_SIMD_ALIGN_SIZE; }
gint_t bli_info_get_stack_buf_align_size( void )    { return BLIS_STACK_BUF_ALIGN_SIZE; }
gint_t bli_info_get_heap_addr_align_size( void )    { return BLIS_HEAP_ADDR_ALIGN_SIZE; }
gint_t bli_info_get_heap_stride_align_size( void )  { return BLIS_HEAP_STRIDE_ALIGN_SIZE; }
gint_t bli_info_get_contig_addr_align_size( void )  { return BLIS_CONTIG_ADDR_ALIGN_SIZE; }
gint_t bli_info_get_enable_stay_auto_init( void )
{
#ifdef BLIS_ENABLE_STAY_AUTO_INITIALIZED
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_enable_blas2blis( void )
{
#ifdef BLIS_ENABLE_BLAS2BLIS
	return 1;
#else
	return 0;
#endif
}
gint_t bli_info_get_blas2blis_int_type_size( void ) { return BLIS_BLAS2BLIS_INT_TYPE_SIZE; }



// -- bli_kernel.h -------------------------------------------------------------

extern blksz_t* gemm_mc;
extern blksz_t* gemm_nc;
extern blksz_t* gemm_kc;

extern blksz_t* gemm4m_mc;
extern blksz_t* gemm4m_nc;
extern blksz_t* gemm4m_kc;

// -- Default cache blocksizes --

// MC default blocksizes

gint_t bli_info_get_default_mc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_mc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_mc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_mc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_mc_z();
	else                              return 0;
}
gint_t bli_info_get_default_mc_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_mc ); }
gint_t bli_info_get_default_mc_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_mc ); }
gint_t bli_info_get_default_mc_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX,
                                                                       ( bli_4m_is_enabled_c() ? gemm4m_mc
	                                                                                           : gemm_mc ) ); }
gint_t bli_info_get_default_mc_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX,
                                                                       ( bli_4m_is_enabled_z() ? gemm4m_mc
	                                                                                           : gemm_mc ) ); }

// NC default blocksizes

gint_t bli_info_get_default_nc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_nc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_nc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_nc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_nc_z();
	else                              return 0;
}
gint_t bli_info_get_default_nc_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_nc ); }
gint_t bli_info_get_default_nc_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_nc ); }
gint_t bli_info_get_default_nc_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX,
                                                                       ( bli_4m_is_enabled_c() ? gemm4m_nc
	                                                                                           : gemm_nc ) ); }
gint_t bli_info_get_default_nc_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX,
                                                                       ( bli_4m_is_enabled_z() ? gemm4m_nc
	                                                                                           : gemm_nc ) ); }

// KC default blocksizes

gint_t bli_info_get_default_kc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_kc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_kc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_kc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_kc_z();
	else                              return 0;
}
gint_t bli_info_get_default_kc_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_kc ); }
gint_t bli_info_get_default_kc_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_kc ); }
gint_t bli_info_get_default_kc_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX,
                                                                       ( bli_4m_is_enabled_c() ? gemm4m_kc
	                                                                                           : gemm_kc ) ); }
gint_t bli_info_get_default_kc_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX,
                                                                       ( bli_4m_is_enabled_z() ? gemm4m_kc
	                                                                                           : gemm_kc ) ); }


// -- Maximum cache blocksizes --

// MC maximum blocksizes

gint_t bli_info_get_maximum_mc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_maximum_mc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_maximum_mc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_maximum_mc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_maximum_mc_z();
	else                              return 0;
}
gint_t bli_info_get_maximum_mc_s( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_FLOAT,    gemm_mc ); }
gint_t bli_info_get_maximum_mc_d( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DOUBLE,   gemm_mc ); }
gint_t bli_info_get_maximum_mc_c( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_SCOMPLEX,
                                                                           ( bli_4m_is_enabled_c() ? gemm4m_mc
	                                                                                               : gemm_mc ) ); }
gint_t bli_info_get_maximum_mc_z( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DCOMPLEX,
                                                                           ( bli_4m_is_enabled_z() ? gemm4m_mc
	                                                                                               : gemm_mc ) ); }

// NC maximum blocksizes

gint_t bli_info_get_maximum_nc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_maximum_nc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_maximum_nc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_maximum_nc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_maximum_nc_z();
	else                              return 0;
}
gint_t bli_info_get_maximum_nc_s( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_FLOAT,    gemm_nc ); }
gint_t bli_info_get_maximum_nc_d( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DOUBLE,   gemm_nc ); }
gint_t bli_info_get_maximum_nc_c( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_SCOMPLEX,
                                                                           ( bli_4m_is_enabled_c() ? gemm4m_nc
	                                                                                               : gemm_nc ) ); }
gint_t bli_info_get_maximum_nc_z( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DCOMPLEX,
                                                                           ( bli_4m_is_enabled_z() ? gemm4m_nc
	                                                                                               : gemm_nc ) ); }

// KC maximum blocksizes

gint_t bli_info_get_maximum_kc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_maximum_kc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_maximum_kc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_maximum_kc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_maximum_kc_z();
	else                              return 0;
}
gint_t bli_info_get_maximum_kc_s( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_FLOAT,    gemm_kc ); }
gint_t bli_info_get_maximum_kc_d( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DOUBLE,   gemm_kc ); }
gint_t bli_info_get_maximum_kc_c( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_SCOMPLEX,
                                                                           ( bli_4m_is_enabled_c() ? gemm4m_kc
	                                                                                               : gemm_kc ) ); }
gint_t bli_info_get_maximum_kc_z( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DCOMPLEX,
                                                                           ( bli_4m_is_enabled_z() ? gemm4m_kc
	                                                                                               : gemm_kc ) ); }


// -- Default register blocksizes --

extern blksz_t* gemm_mr;
extern blksz_t* gemm_nr;
extern blksz_t* gemm_kr;

extern blksz_t* gemm4m_mr;
extern blksz_t* gemm4m_nr;
extern blksz_t* gemm4m_kr;

// MR default blocksizes

gint_t bli_info_get_default_mr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_mr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_mr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_mr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_mr_z();
	else                              return 0;
}
gint_t bli_info_get_default_mr_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_mr ); }
gint_t bli_info_get_default_mr_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_mr ); }
gint_t bli_info_get_default_mr_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX,
                                                                       ( bli_4m_is_enabled_c() ? gemm4m_mr
	                                                                                           : gemm_mr ) ); }
gint_t bli_info_get_default_mr_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX,
                                                                       ( bli_4m_is_enabled_z() ? gemm4m_mr
	                                                                                           : gemm_mr ) ); }

// NR default blocksizes

gint_t bli_info_get_default_nr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_nr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_nr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_nr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_nr_z();
	else                              return 0;
}
gint_t bli_info_get_default_nr_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_nr ); }
gint_t bli_info_get_default_nr_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_nr ); }
gint_t bli_info_get_default_nr_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX,
                                                                       ( bli_4m_is_enabled_c() ? gemm4m_nr
	                                                                                           : gemm_nr ) ); }
gint_t bli_info_get_default_nr_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX,
                                                                       ( bli_4m_is_enabled_z() ? gemm4m_nr
	                                                                                           : gemm_nr ) ); }

// KR default blocksizes

gint_t bli_info_get_default_kr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_kr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_kr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_kr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_kr_z();
	else                              return 0;
}
gint_t bli_info_get_default_kr_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_kr ); }
gint_t bli_info_get_default_kr_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_kr ); }
gint_t bli_info_get_default_kr_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX,
                                                                       ( bli_4m_is_enabled_c() ? gemm4m_kr
	                                                                                           : gemm_kr ) ); }
gint_t bli_info_get_default_kr_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX,
                                                                       ( bli_4m_is_enabled_z() ? gemm4m_kr
	                                                                                           : gemm_kr ) ); }


// -- Packing register blocksizes --

// MR packing blocksize

gint_t bli_info_get_packdim_mr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_packdim_mr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_packdim_mr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_packdim_mr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_packdim_mr_z();
	else                              return 0;
}
gint_t bli_info_get_packdim_mr_s( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_FLOAT,    gemm_mr ); }
gint_t bli_info_get_packdim_mr_d( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DOUBLE,   gemm_mr ); }
gint_t bli_info_get_packdim_mr_c( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_SCOMPLEX,
                                                                           ( bli_4m_is_enabled_c() ? gemm4m_mr
	                                                                                               : gemm_mr ) ); }
gint_t bli_info_get_packdim_mr_z( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DCOMPLEX,
                                                                           ( bli_4m_is_enabled_z() ? gemm4m_mr
	                                                                                               : gemm_mr ) ); }

// NR packing blocksize

gint_t bli_info_get_packdim_nr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_packdim_nr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_packdim_nr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_packdim_nr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_packdim_nr_z();
	else                              return 0;
}
gint_t bli_info_get_packdim_nr_s( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_FLOAT,    gemm_nr ); }
gint_t bli_info_get_packdim_nr_d( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DOUBLE,   gemm_nr ); }
gint_t bli_info_get_packdim_nr_c( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_SCOMPLEX,
                                                                           ( bli_4m_is_enabled_c() ? gemm4m_nr
	                                                                                               : gemm_nr ) ); }
gint_t bli_info_get_packdim_nr_z( void ) { bli_init(); return bli_blksz_max_for_type( BLIS_DCOMPLEX,
                                                                           ( bli_4m_is_enabled_z() ? gemm4m_nr
	                                                                                               : gemm_nr ) ); }

// -- Micro-panel alignment --

extern blksz_t* gemm_upanel_a_align;
extern blksz_t* gemm_upanel_b_align;

// Micro-panel alignment of A

gint_t bli_info_get_upanel_a_align_size( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_upanel_a_align_size_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_upanel_a_align_size_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_upanel_a_align_size_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_upanel_a_align_size_z();
	else                              return 0;
}
gint_t bli_info_get_upanel_a_align_size_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_upanel_a_align ); }
gint_t bli_info_get_upanel_a_align_size_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_upanel_a_align ); }
gint_t bli_info_get_upanel_a_align_size_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX, gemm_upanel_a_align ); }
gint_t bli_info_get_upanel_a_align_size_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX, gemm_upanel_a_align ); }

// Micro-panel alignment of B

gint_t bli_info_get_upanel_b_align_size( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_upanel_b_align_size_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_upanel_b_align_size_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_upanel_b_align_size_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_upanel_b_align_size_z();
	else                              return 0;
}
gint_t bli_info_get_upanel_b_align_size_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemm_upanel_b_align ); }
gint_t bli_info_get_upanel_b_align_size_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemm_upanel_b_align ); }
gint_t bli_info_get_upanel_b_align_size_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX, gemm_upanel_b_align ); }
gint_t bli_info_get_upanel_b_align_size_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX, gemm_upanel_b_align ); }


// -- Level-2 cache blocksizes --

extern blksz_t* gemv_mc;
extern blksz_t* gemv_nc;

// m dimension default blocksizes

gint_t bli_info_get_default_l2_mc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_l2_mc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_l2_mc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_l2_mc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_l2_mc_z();
	else                              return 0;
}
gint_t bli_info_get_default_l2_mc_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemv_mc ); }
gint_t bli_info_get_default_l2_mc_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemv_mc ); }
gint_t bli_info_get_default_l2_mc_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX, gemv_mc ); }
gint_t bli_info_get_default_l2_mc_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX, gemv_mc ); }


// n dimension default blocksizes

gint_t bli_info_get_default_l2_nc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_l2_nc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_l2_nc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_l2_nc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_l2_nc_z();
	else                              return 0;
}
gint_t bli_info_get_default_l2_nc_s( void ) { bli_init(); return bli_blksz_for_type( BLIS_FLOAT,    gemv_nc ); }
gint_t bli_info_get_default_l2_nc_d( void ) { bli_init(); return bli_blksz_for_type( BLIS_DOUBLE,   gemv_nc ); }
gint_t bli_info_get_default_l2_nc_c( void ) { bli_init(); return bli_blksz_for_type( BLIS_SCOMPLEX, gemv_nc ); }
gint_t bli_info_get_default_l2_nc_z( void ) { bli_init(); return bli_blksz_for_type( BLIS_DCOMPLEX, gemv_nc ); }


// -- Level-1f fusing factors --

// default

gint_t bli_info_get_default_l1f_fuse_fac( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_l1f_fuse_fac_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_l1f_fuse_fac_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_l1f_fuse_fac_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_l1f_fuse_fac_z();
	else                              return 0;
}
gint_t bli_info_get_default_l1f_fuse_fac_s( void ) { return BLIS_L1F_FUSE_FAC_S; }
gint_t bli_info_get_default_l1f_fuse_fac_d( void ) { return BLIS_L1F_FUSE_FAC_D; }
gint_t bli_info_get_default_l1f_fuse_fac_c( void ) { return BLIS_L1F_FUSE_FAC_C; }
gint_t bli_info_get_default_l1f_fuse_fac_z( void ) { return BLIS_L1F_FUSE_FAC_Z; }


// axpyf

gint_t bli_info_get_axpyf_fuse_fac( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_axpyf_fuse_fac_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_axpyf_fuse_fac_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_axpyf_fuse_fac_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_axpyf_fuse_fac_z();
	else                              return 0;
}
gint_t bli_info_get_axpyf_fuse_fac_s( void ) { return BLIS_AXPYF_FUSE_FAC_S; }
gint_t bli_info_get_axpyf_fuse_fac_d( void ) { return BLIS_AXPYF_FUSE_FAC_D; }
gint_t bli_info_get_axpyf_fuse_fac_c( void ) { return BLIS_AXPYF_FUSE_FAC_C; }
gint_t bli_info_get_axpyf_fuse_fac_z( void ) { return BLIS_AXPYF_FUSE_FAC_Z; }


// dotxf

gint_t bli_info_get_dotxf_fuse_fac( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_dotxf_fuse_fac_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_dotxf_fuse_fac_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_dotxf_fuse_fac_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_dotxf_fuse_fac_z();
	else                              return 0;
}
gint_t bli_info_get_dotxf_fuse_fac_s( void ) { return BLIS_DOTXF_FUSE_FAC_S; }
gint_t bli_info_get_dotxf_fuse_fac_d( void ) { return BLIS_DOTXF_FUSE_FAC_D; }
gint_t bli_info_get_dotxf_fuse_fac_c( void ) { return BLIS_DOTXF_FUSE_FAC_C; }
gint_t bli_info_get_dotxf_fuse_fac_z( void ) { return BLIS_DOTXF_FUSE_FAC_Z; }


// dotxaxpyf

gint_t bli_info_get_dotxaxpyf_fuse_fac( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_dotxaxpyf_fuse_fac_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_dotxaxpyf_fuse_fac_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_dotxaxpyf_fuse_fac_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_dotxaxpyf_fuse_fac_z();
	else                              return 0;
}
gint_t bli_info_get_dotxaxpyf_fuse_fac_s( void ) { return BLIS_DOTXAXPYF_FUSE_FAC_S; }
gint_t bli_info_get_dotxaxpyf_fuse_fac_d( void ) { return BLIS_DOTXAXPYF_FUSE_FAC_D; }
gint_t bli_info_get_dotxaxpyf_fuse_fac_c( void ) { return BLIS_DOTXAXPYF_FUSE_FAC_C; }
gint_t bli_info_get_dotxaxpyf_fuse_fac_z( void ) { return BLIS_DOTXAXPYF_FUSE_FAC_Z; }


// -- Level-3 kernel definitions --

static char* ukr_type_str[4] = { "refnce",
                                 "virt4m",
                                 "virt3m",
                                 "optmzd" };

char* bli_info_get_gemm_ukr_type_string( num_t dt )
{
	return ukr_type_str[ bli_gemm_ukernel_impl_type( dt ) ];
}

char* bli_info_get_gemmtrsm_l_ukr_type_string( num_t dt )
{
	return ukr_type_str[ bli_gemmtrsm_l_ukernel_impl_type( dt ) ];
}

char* bli_info_get_gemmtrsm_u_ukr_type_string( num_t dt )
{
	return ukr_type_str[ bli_gemmtrsm_u_ukernel_impl_type( dt ) ];
}

char* bli_info_get_trsm_l_ukr_type_string( num_t dt )
{
	return ukr_type_str[ bli_trsm_l_ukernel_impl_type( dt ) ];
}

char* bli_info_get_trsm_u_ukr_type_string( num_t dt )
{
	return ukr_type_str[ bli_trsm_u_ukernel_impl_type( dt ) ];
}



// -- bli_mem_pool_macro_defs.h ------------------------------------------------

gint_t bli_info_get_mk_pool_size( void ) { return BLIS_MK_POOL_SIZE; }
gint_t bli_info_get_kn_pool_size( void ) { return BLIS_KN_POOL_SIZE; }
gint_t bli_info_get_mn_pool_size( void ) { return BLIS_MN_POOL_SIZE; }



// -- BLIS implementation query (level-3) --------------------------------------

char* bli_info_get_gemm_impl_string( num_t dt )  { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_hemm_impl_string( num_t dt )  { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_herk_impl_string( num_t dt )  { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_her2k_impl_string( num_t dt ) { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_symm_impl_string( num_t dt )  { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_syrk_impl_string( num_t dt )  { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_syr2k_impl_string( num_t dt ) { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_trmm_impl_string( num_t dt )  { bli_init(); return bli_trmm_query_impl_string( dt ); }
char* bli_info_get_trmm3_impl_string( num_t dt ) { bli_init(); return bli_gemm_query_impl_string( dt ); }
char* bli_info_get_trsm_impl_string( num_t dt )  { bli_init(); return bli_trsm_query_impl_string( dt ); }

