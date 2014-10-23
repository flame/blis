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


// -- General library information ----------------------------------------------

char* bli_info_get_version_str( void );
char* bli_info_get_int_type_size_str( void );


// -- bli_config.h -------------------------------------------------------------

gint_t bli_info_get_int_type_size( void );
gint_t bli_info_get_num_fp_types( void );
gint_t bli_info_get_max_type_size( void );
gint_t bli_info_get_max_num_threads( void );
gint_t bli_info_get_num_mc_x_kc_blocks( void );
gint_t bli_info_get_num_kc_x_nc_blocks( void );
gint_t bli_info_get_num_mc_x_nc_blocks( void );
gint_t bli_info_get_max_preload_byte_offset( void );
gint_t bli_info_get_simd_align_size( void );
gint_t bli_info_get_stack_buf_align_size( void );
gint_t bli_info_get_heap_addr_align_size( void );
gint_t bli_info_get_heap_stride_align_size( void );
gint_t bli_info_get_contig_addr_align_size( void );
gint_t bli_info_get_enable_stay_auto_init( void );
gint_t bli_info_get_enable_blas2blis( void );
gint_t bli_info_get_blas2blis_int_type_size( void );


// -- bli_kernel.h -------------------------------------------------------------

// -- Default cache blocksizes --

gint_t bli_info_get_default_mc( num_t dt );
gint_t bli_info_get_default_mc_s( void );
gint_t bli_info_get_default_mc_d( void );
gint_t bli_info_get_default_mc_c( void );
gint_t bli_info_get_default_mc_z( void );

gint_t bli_info_get_default_kc( num_t dt );
gint_t bli_info_get_default_kc_s( void );
gint_t bli_info_get_default_kc_d( void );
gint_t bli_info_get_default_kc_c( void );
gint_t bli_info_get_default_kc_z( void );

gint_t bli_info_get_default_nc( num_t dt );
gint_t bli_info_get_default_nc_s( void );
gint_t bli_info_get_default_nc_d( void );
gint_t bli_info_get_default_nc_c( void );
gint_t bli_info_get_default_nc_z( void );

// -- Maximum cache blocksizes --

gint_t bli_info_get_maximum_mc( num_t dt );
gint_t bli_info_get_maximum_mc_s( void );
gint_t bli_info_get_maximum_mc_d( void );
gint_t bli_info_get_maximum_mc_c( void );
gint_t bli_info_get_maximum_mc_z( void );

gint_t bli_info_get_maximum_kc( num_t dt );
gint_t bli_info_get_maximum_kc_s( void );
gint_t bli_info_get_maximum_kc_d( void );
gint_t bli_info_get_maximum_kc_c( void );
gint_t bli_info_get_maximum_kc_z( void );

gint_t bli_info_get_maximum_nc( num_t dt );
gint_t bli_info_get_maximum_nc_s( void );
gint_t bli_info_get_maximum_nc_d( void );
gint_t bli_info_get_maximum_nc_c( void );
gint_t bli_info_get_maximum_nc_z( void );

// -- Default register blocksizes --

gint_t bli_info_get_default_mr( num_t dt );
gint_t bli_info_get_default_mr_s( void );
gint_t bli_info_get_default_mr_d( void );
gint_t bli_info_get_default_mr_c( void );
gint_t bli_info_get_default_mr_z( void );

gint_t bli_info_get_default_kr( num_t dt );
gint_t bli_info_get_default_kr_s( void );
gint_t bli_info_get_default_kr_d( void );
gint_t bli_info_get_default_kr_c( void );
gint_t bli_info_get_default_kr_z( void );

gint_t bli_info_get_default_nr( num_t dt );
gint_t bli_info_get_default_nr_s( void );
gint_t bli_info_get_default_nr_d( void );
gint_t bli_info_get_default_nr_c( void );
gint_t bli_info_get_default_nr_z( void );

// -- Packing register blocksizes --

gint_t bli_info_get_packdim_mr( num_t dt );
gint_t bli_info_get_packdim_mr_s( void );
gint_t bli_info_get_packdim_mr_d( void );
gint_t bli_info_get_packdim_mr_c( void );
gint_t bli_info_get_packdim_mr_z( void );

gint_t bli_info_get_packdim_nr( num_t dt );
gint_t bli_info_get_packdim_nr_s( void );
gint_t bli_info_get_packdim_nr_d( void );
gint_t bli_info_get_packdim_nr_c( void );
gint_t bli_info_get_packdim_nr_z( void );

// -- Micro-panel alignment for A --

gint_t bli_info_get_upanel_a_align_size( num_t dt );
gint_t bli_info_get_upanel_a_align_size_s( void );
gint_t bli_info_get_upanel_a_align_size_d( void );
gint_t bli_info_get_upanel_a_align_size_c( void );
gint_t bli_info_get_upanel_a_align_size_z( void );

// -- Micro-panel alignment for B --

gint_t bli_info_get_upanel_b_align_size( num_t dt );
gint_t bli_info_get_upanel_b_align_size_s( void );
gint_t bli_info_get_upanel_b_align_size_d( void );
gint_t bli_info_get_upanel_b_align_size_c( void );
gint_t bli_info_get_upanel_b_align_size_z( void );


// -- Level-2 cache blocksizes --

gint_t bli_info_get_default_l2_mc_s( void );
gint_t bli_info_get_default_l2_mc_d( void );
gint_t bli_info_get_default_l2_mc_c( void );
gint_t bli_info_get_default_l2_mc_z( void );

gint_t bli_info_get_default_l2_nc_s( void );
gint_t bli_info_get_default_l2_nc_d( void );
gint_t bli_info_get_default_l2_nc_c( void );
gint_t bli_info_get_default_l2_nc_z( void );


// -- Level-1f fusing factors --

gint_t bli_info_get_default_l1f_fuse_fac( num_t dt );
gint_t bli_info_get_default_l1f_fuse_fac_s( void );
gint_t bli_info_get_default_l1f_fuse_fac_d( void );
gint_t bli_info_get_default_l1f_fuse_fac_c( void );
gint_t bli_info_get_default_l1f_fuse_fac_z( void );

gint_t bli_info_get_axpyf_fuse_fac( num_t dt );
gint_t bli_info_get_axpyf_fuse_fac_s( void );
gint_t bli_info_get_axpyf_fuse_fac_d( void );
gint_t bli_info_get_axpyf_fuse_fac_c( void );
gint_t bli_info_get_axpyf_fuse_fac_z( void );

gint_t bli_info_get_dotxf_fuse_fac( num_t dt );
gint_t bli_info_get_dotxf_fuse_fac_s( void );
gint_t bli_info_get_dotxf_fuse_fac_d( void );
gint_t bli_info_get_dotxf_fuse_fac_c( void );
gint_t bli_info_get_dotxf_fuse_fac_z( void );

gint_t bli_info_get_dotxaxpyf_fuse_fac( num_t dt );
gint_t bli_info_get_dotxaxpyf_fuse_fac_s( void );
gint_t bli_info_get_dotxaxpyf_fuse_fac_d( void );
gint_t bli_info_get_dotxaxpyf_fuse_fac_c( void );
gint_t bli_info_get_dotxaxpyf_fuse_fac_z( void );


// -- Level-3 kernel definitions --

char* bli_info_get_gemm_ukr_type_string( num_t dt );
char* bli_info_get_gemmtrsm_l_ukr_type_string( num_t dt );
char* bli_info_get_gemmtrsm_u_ukr_type_string( num_t dt );
char* bli_info_get_trsm_l_ukr_type_string( num_t dt );
char* bli_info_get_trsm_u_ukr_type_string( num_t dt );


// -- bli_mem_pool_macro_defs.h ------------------------------------------------

gint_t bli_info_get_mk_pool_size( void );
gint_t bli_info_get_kn_pool_size( void );
gint_t bli_info_get_mn_pool_size( void );


// -- BLIS implementation query (level-3) --------------------------------------

char* bli_info_get_gemm_impl_string( num_t dt );
char* bli_info_get_hemm_impl_string( num_t dt );
char* bli_info_get_herk_impl_string( num_t dt );
char* bli_info_get_her2k_impl_string( num_t dt );
char* bli_info_get_symm_impl_string( num_t dt );
char* bli_info_get_syrk_impl_string( num_t dt );
char* bli_info_get_syr2k_impl_string( num_t dt );
char* bli_info_get_trmm_impl_string( num_t dt );
char* bli_info_get_trmm3_impl_string( num_t dt );
char* bli_info_get_trsm_impl_string( num_t dt );

