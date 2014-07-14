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

// -- 4m status --

gint_t bli_info_get_enable_scomplex_via_4m( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return TRUE;
#else
	return FALSE;
#endif
}
gint_t bli_info_get_enable_dcomplex_via_4m( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return TRUE;
#else
	return FALSE;
#endif
}


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
gint_t bli_info_get_default_mc_s( void ) { return BLIS_DEFAULT_MC_S; }
gint_t bli_info_get_default_mc_d( void ) { return BLIS_DEFAULT_MC_D; }
gint_t bli_info_get_default_mc_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_MC_C;
#else
	return BLIS_DEFAULT_MC_C;
#endif
}
gint_t bli_info_get_default_mc_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_MC_Z;
#else
	return BLIS_DEFAULT_MC_Z;
#endif
}


// KC default blocksizes

gint_t bli_info_get_default_kc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_kc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_kc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_kc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_kc_z();
	else                              return 0;
}
gint_t bli_info_get_default_kc_s( void ) { return BLIS_DEFAULT_KC_S; }
gint_t bli_info_get_default_kc_d( void ) { return BLIS_DEFAULT_KC_D; }
gint_t bli_info_get_default_kc_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_KC_C;
#else
	return BLIS_DEFAULT_KC_C;
#endif
}
gint_t bli_info_get_default_kc_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_KC_Z;
#else
	return BLIS_DEFAULT_KC_Z;
#endif
}


// NC default blocksizes

gint_t bli_info_get_default_nc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_nc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_nc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_nc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_nc_z();
	else                              return 0;
}
gint_t bli_info_get_default_nc_s( void ) { return BLIS_DEFAULT_NC_S; }
gint_t bli_info_get_default_nc_d( void ) { return BLIS_DEFAULT_NC_D; }
gint_t bli_info_get_default_nc_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_NC_C;
#else
	return BLIS_DEFAULT_NC_C;
#endif
}
gint_t bli_info_get_default_nc_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_NC_Z;
#else
	return BLIS_DEFAULT_NC_Z;
#endif
}


// -- Cache blocksize extensions --

// MC blocksize extensions

gint_t bli_info_get_extend_mc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_extend_mc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_extend_mc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_extend_mc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_extend_mc_z();
	else                              return 0;
}
gint_t bli_info_get_extend_mc_s( void ) { return BLIS_EXTEND_MC_S; }
gint_t bli_info_get_extend_mc_d( void ) { return BLIS_EXTEND_MC_D; }
gint_t bli_info_get_extend_mc_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_MC_C;
#else
	return BLIS_EXTEND_MC_C;
#endif
}
gint_t bli_info_get_extend_mc_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_MC_Z;
#else
	return BLIS_EXTEND_MC_Z;
#endif
}


// KC blocksize extensions

gint_t bli_info_get_extend_kc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_extend_kc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_extend_kc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_extend_kc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_extend_kc_z();
	else                              return 0;
}
gint_t bli_info_get_extend_kc_s( void ) { return BLIS_EXTEND_KC_S; }
gint_t bli_info_get_extend_kc_d( void ) { return BLIS_EXTEND_KC_D; }
gint_t bli_info_get_extend_kc_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_KC_C;
#else
	return BLIS_EXTEND_KC_C;
#endif
}
gint_t bli_info_get_extend_kc_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_KC_Z;
#else
	return BLIS_EXTEND_KC_Z;
#endif
}


// NC blocksize extensions

gint_t bli_info_get_extend_nc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_extend_nc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_extend_nc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_extend_nc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_extend_nc_z();
	else                              return 0;
}
gint_t bli_info_get_extend_nc_s( void ) { return BLIS_EXTEND_NC_S; }
gint_t bli_info_get_extend_nc_d( void ) { return BLIS_EXTEND_NC_D; }
gint_t bli_info_get_extend_nc_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_NC_C;
#else
	return BLIS_EXTEND_NC_C;
#endif
}
gint_t bli_info_get_extend_nc_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_NC_Z;
#else
	return BLIS_EXTEND_NC_Z;
#endif
}


// -- Default register blocksizes --

// MR default blocksizes

gint_t bli_info_get_default_mr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_mr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_mr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_mr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_mr_z();
	else                              return 0;
}
gint_t bli_info_get_default_mr_s( void ) { return BLIS_DEFAULT_MR_S; }
gint_t bli_info_get_default_mr_d( void ) { return BLIS_DEFAULT_MR_D; }
gint_t bli_info_get_default_mr_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_MR_C;
#else
	return BLIS_DEFAULT_MR_C;
#endif
}
gint_t bli_info_get_default_mr_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_MR_Z;
#else
	return BLIS_DEFAULT_MR_Z;
#endif
}


// KR default blocksizes

gint_t bli_info_get_default_kr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_kr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_kr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_kr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_kr_z();
	else                              return 0;
}
gint_t bli_info_get_default_kr_s( void ) { return BLIS_DEFAULT_KR_S; }
gint_t bli_info_get_default_kr_d( void ) { return BLIS_DEFAULT_KR_D; }
gint_t bli_info_get_default_kr_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_KR_C;
#else
	return BLIS_DEFAULT_KR_C;
#endif
}
gint_t bli_info_get_default_kr_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_KR_Z;
#else
	return BLIS_DEFAULT_KR_Z;
#endif
}


// NR default blocksizes

gint_t bli_info_get_default_nr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_nr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_nr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_nr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_nr_z();
	else                              return 0;
}
gint_t bli_info_get_default_nr_s( void ) { return BLIS_DEFAULT_NR_S; }
gint_t bli_info_get_default_nr_d( void ) { return BLIS_DEFAULT_NR_D; }
gint_t bli_info_get_default_nr_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_NR_C;
#else
	return BLIS_DEFAULT_NR_C;
#endif
}
gint_t bli_info_get_default_nr_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_DEFAULT_4M_NR_Z;
#else
	return BLIS_DEFAULT_NR_Z;
#endif
}


// -- Register blocksize extensions --

// MR blocksize extensions

gint_t bli_info_get_extend_mr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_extend_mr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_extend_mr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_extend_mr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_extend_mr_z();
	else                              return 0;
}
gint_t bli_info_get_extend_mr_s( void ) { return BLIS_EXTEND_MR_S; }
gint_t bli_info_get_extend_mr_d( void ) { return BLIS_EXTEND_MR_D; }
gint_t bli_info_get_extend_mr_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_MR_C;
#else
	return BLIS_EXTEND_MR_C;
#endif
}
gint_t bli_info_get_extend_mr_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_MR_Z;
#else
	return BLIS_EXTEND_MR_Z;
#endif
}


// NR blocksize extensions

gint_t bli_info_get_extend_nr( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_extend_nr_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_extend_nr_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_extend_nr_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_extend_nr_z();
	else                              return 0;
}
gint_t bli_info_get_extend_nr_s( void ) { return BLIS_EXTEND_NR_S; }
gint_t bli_info_get_extend_nr_d( void ) { return BLIS_EXTEND_NR_D; }
gint_t bli_info_get_extend_nr_c( void ) {
#ifdef BLIS_ENABLE_SCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_NR_C;
#else
	return BLIS_EXTEND_NR_C;
#endif
}
gint_t bli_info_get_extend_nr_z( void ) {
#ifdef BLIS_ENABLE_DCOMPLEX_VIA_4M
	return BLIS_EXTEND_4M_NR_Z;
#else
	return BLIS_EXTEND_NR_Z;
#endif
}


// -- Level-2 cache blocksizes --

// m dimension default blocksizes

gint_t bli_info_get_default_l2_mc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_l2_mc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_l2_mc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_l2_mc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_l2_mc_z();
	else                              return 0;
}
gint_t bli_info_get_default_l2_mc_s( void ) { return BLIS_DEFAULT_L2_MC_S; }
gint_t bli_info_get_default_l2_mc_d( void ) { return BLIS_DEFAULT_L2_MC_D; }
gint_t bli_info_get_default_l2_mc_c( void ) { return BLIS_DEFAULT_L2_MC_C; }
gint_t bli_info_get_default_l2_mc_z( void ) { return BLIS_DEFAULT_L2_MC_Z; }


// n dimension default blocksizes

gint_t bli_info_get_default_l2_nc( num_t dt )
{
	if      ( bli_is_float   ( dt ) ) return bli_info_get_default_l2_nc_s();
	else if ( bli_is_double  ( dt ) ) return bli_info_get_default_l2_nc_d();
	else if ( bli_is_scomplex( dt ) ) return bli_info_get_default_l2_nc_c();
	else if ( bli_is_dcomplex( dt ) ) return bli_info_get_default_l2_nc_z();
	else                              return 0;
}
gint_t bli_info_get_default_l2_nc_s( void ) { return BLIS_DEFAULT_L2_NC_S; }
gint_t bli_info_get_default_l2_nc_d( void ) { return BLIS_DEFAULT_L2_NC_D; }
gint_t bli_info_get_default_l2_nc_c( void ) { return BLIS_DEFAULT_L2_NC_C; }
gint_t bli_info_get_default_l2_nc_z( void ) { return BLIS_DEFAULT_L2_NC_Z; }


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


// dotxf

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



// -- bli_mem_pool_macro_defs.h ------------------------------------------------

gint_t bli_info_get_mk_pool_size( void ) { return BLIS_MK_POOL_SIZE; }
gint_t bli_info_get_kn_pool_size( void ) { return BLIS_KN_POOL_SIZE; }
gint_t bli_info_get_mn_pool_size( void ) { return BLIS_MN_POOL_SIZE; }

