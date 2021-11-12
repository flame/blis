/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
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

//
// thrinfo_t macros specific to various level-3 operations.
//

// gemm

// NOTE: The definition of bli_gemm_get_next_?_upanel() does not need to
// change depending on BLIS_ENABLE_JRIR_SLAB / BLIS_ENABLE_JRIR_RR.
#define bli_gemm_get_next_a_upanel( a1, step, inc ) ( a1 + step * inc )
#define bli_gemm_get_next_b_upanel( b1, step, inc ) ( b1 + step * inc )

// gemmt

// NOTE: The definition of bli_gemmt_get_next_?_upanel() does not need to
// change depending on BLIS_ENABLE_JRIR_SLAB / BLIS_ENABLE_JRIR_RR.
#define bli_gemmt_get_next_a_upanel( a1, step, inc ) ( a1 + step * inc )
#define bli_gemmt_get_next_b_upanel( b1, step, inc ) ( b1 + step * inc )

// trmm

// NOTE: The definition of bli_trmm_get_next_?_upanel() does not need to
// change depending on BLIS_ENABLE_JRIR_SLAB / BLIS_ENABLE_JRIR_RR.
#define bli_trmm_get_next_a_upanel( a1, step, inc ) ( a1 + step * inc )
#define bli_trmm_get_next_b_upanel( b1, step, inc ) ( b1 + step * inc )

#define bli_trmm_my_iter_rr( index, thread ) \
\
	( index % thread->n_way == thread->work_id % thread->n_way )

// trsm

#define bli_trsm_my_iter_rr( index, thread ) \
\
	( index % thread->n_way == thread->work_id % thread->n_way )

//
// thrinfo_t APIs specific to level-3 operations.
//

void bli_l3_thrinfo_init
     (
       thrinfo_t* thread,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id,
       thrinfo_t* sub_node
     );

void bli_l3_thrinfo_init_single
     (
       thrinfo_t* thread
     );

void bli_l3_thrinfo_free
     (
       rntm_t*    rntm,
       thrinfo_t* thread
     );

void bli_l3_sup_thrinfo_free
     (
       rntm_t*    rntm,
       thrinfo_t* thread
     );

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_create_root
     (
       dim_t       id,
       thrcomm_t*  gl_comm,
       rntm_t*     rntm,
       cntl_t*     cntl,
       thrinfo_t** thread
     );

void bli_l3_sup_thrinfo_create_root
     (
       dim_t       id,
       thrcomm_t*  gl_comm,
       rntm_t*     rntm,
       thrinfo_t** thread
     );

void bli_l3_sup_thrinfo_update_root
     (
       rntm_t*    rntm,
       thrinfo_t* thread
     );

void bli_l3_thrinfo_print_gemm_paths
     (
       thrinfo_t** threads
     );

void bli_l3_thrinfo_print_trsm_paths
     (
       thrinfo_t** threads
     );

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_free_paths
     (
       rntm_t*     rntm,
       thrinfo_t** threads
     );

