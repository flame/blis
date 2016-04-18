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


// -- General configuration-related --------------------------------------------

gint_t bli_info_get_int_type_size( void );
gint_t bli_info_get_num_fp_types( void );
gint_t bli_info_get_max_type_size( void );
gint_t bli_info_get_page_size( void );
gint_t bli_info_get_simd_num_registers( void );
gint_t bli_info_get_simd_size( void );
gint_t bli_info_get_simd_align_size( void );
gint_t bli_info_get_stack_buf_max_size( void );
gint_t bli_info_get_stack_buf_align_size( void );
gint_t bli_info_get_heap_addr_align_size( void );
gint_t bli_info_get_heap_stride_align_size( void );
gint_t bli_info_get_pool_addr_align_size( void );
gint_t bli_info_get_enable_stay_auto_init( void );
gint_t bli_info_get_enable_blas2blis( void );
gint_t bli_info_get_enable_cblas( void );
gint_t bli_info_get_blas2blis_int_type_size( void );


// -- Kernel implementation-related --------------------------------------------


// -- Level-3 kernel definitions --

char* bli_info_get_gemm_ukr_impl_string( ind_t method, num_t dt );
char* bli_info_get_gemmtrsm_l_ukr_impl_string( ind_t method, num_t dt );
char* bli_info_get_gemmtrsm_u_ukr_impl_string( ind_t method, num_t dt );
char* bli_info_get_trsm_l_ukr_impl_string( ind_t method, num_t dt );
char* bli_info_get_trsm_u_ukr_impl_string( ind_t method, num_t dt );


// -- Memory pool-related ------------------------------------------------------

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

