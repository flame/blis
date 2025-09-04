/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

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

#include "blis.h"

#ifdef BLIS_ENABLE_OPENMP

void* bli_pack_full_thread_entry( void* data_void ) { return NULL; }

void bli_pack_full_thread_decorator
     (
       pack_full_t   func,
       const char*   identifier,
             obj_t*  alpha_obj,
             obj_t*  src_obj,
             obj_t*  dest_obj,
             cntx_t* cntx,
             rntm_t* rntm
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_3);

    dim_t n_threads = bli_rntm_num_threads( rntm );

    /* Ensure n_threads is always greater than or equal to 1 */
    /* Passing BLIS_IC_NT and BLIS_JC_NT for pack can lead to n_threads */
    /* becoming negative. In that case, packing is done using 1 thread */
    n_threads = ( n_threads > 0 ) ? n_threads : 1;

    _Pragma( "omp parallel num_threads(n_threads)" )
    {
        thrinfo_t thread;
        bli_thrinfo_set_n_way( n_threads, &thread );
        bli_thrinfo_set_work_id( omp_get_thread_num(), &thread );

        rntm_t           rntm_l = *rntm;
        rntm_t* restrict rntm_p = &rntm_l;

        func
        (
         identifier,
         alpha_obj,
         src_obj,
         dest_obj,
         cntx,
         rntm_p,
         &thread
        );
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_3);
}
#endif

