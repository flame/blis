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

//
// thrinfo_t macros specific to various level-3 operations.
//

// gemm

#define gemm_get_next_a_micropanel( thread, a1, step ) ( a1 + step * thread->n_way )
#define gemm_get_next_b_micropanel( thread, b1, step ) ( b1 + step * thread->n_way )

// herk

#define herk_get_next_a_micropanel( thread, a1, step ) ( a1 + step * thread->n_way )
#define herk_get_next_b_micropanel( thread, b1, step ) ( b1 + step * thread->n_way )

// trmm

#define trmm_r_ir_my_iter( index, thread ) ( index % thread->n_way == thread->work_id % thread->n_way )
#define trmm_r_jr_my_iter( index, thread ) ( index % thread->n_way == thread->work_id % thread->n_way )
#define trmm_l_ir_my_iter( index, thread ) ( index % thread->n_way == thread->work_id % thread->n_way )
#define trmm_l_jr_my_iter( index, thread ) ( index % thread->n_way == thread->work_id % thread->n_way )

// trsm

#define trsm_my_iter( index, thread ) ( index % thread->n_way == thread->work_id % thread->n_way )

//
// thrinfo_t APIs specific to level-3 operations.
//

#if 0
thrinfo_t* bli_l3_thrinfo_create
     (
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       dim_t      n_way,
       dim_t      work_id,
       thrinfo_t* sub_node
     );
#endif

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
       thrinfo_t* thread
     );

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_create_root
     (
       dim_t       id,
       thrcomm_t*  gl_comm,
       cntx_t*     cntx,
       cntl_t*     cntl,
       thrinfo_t** thread
     );

void bli_l3_thrinfo_print_paths
     (
       thrinfo_t** threads
     );

// -----------------------------------------------------------------------------

#if 0
thrinfo_t** bli_l3_thrinfo_create_roots
     (
       cntx_t* cntx,
       cntl_t* cntl
     );

thrinfo_t** bli_l3_thrinfo_create_full_paths
     (
       cntx_t* cntx
     );
#endif

void bli_l3_thrinfo_free_paths
     (
       thrinfo_t** threads
     );

