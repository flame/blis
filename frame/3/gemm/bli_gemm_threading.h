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


struct gemm_thrinfo_s //implements thrinfo_t
{
    thread_comm_t*      ocomm;       //The thread communicator for the other threads sharing the same work at this level
    dim_t               ocomm_id;    //Our thread id within that thread comm
    thread_comm_t*      icomm;       //The thread communicator for the other threads sharing the same work at this level
    dim_t               icomm_id;    //Our thread id within that thread comm

    dim_t               n_way;       //Number of distinct caucuses used to parallelize the loop
    dim_t               work_id;     //What we're working on

    packm_thrinfo_t*    opackm;
    packm_thrinfo_t*    ipackm;
    struct gemm_thrinfo_s*    sub_gemm;
};
typedef struct gemm_thrinfo_s gemm_thrinfo_t;

#define gemm_thread_sub_gemm( thread )  thread->sub_gemm
#define gemm_thread_sub_opackm( thread )  thread->opackm
#define gemm_thread_sub_ipackm( thread )  thread->ipackm

// For use in gemm micro-kernel
#define gemm_get_next_a_micropanel( thread, a1, step ) ( a1 + step * thread->n_way )
#define gemm_get_next_b_micropanel( thread, b1, step ) ( b1 + step * thread->n_way )

gemm_thrinfo_t** bli_create_gemm_thrinfo_paths( );
void bli_gemm_thrinfo_free_paths( gemm_thrinfo_t**, dim_t n_threads );

void bli_setup_gemm_thrinfo_node( gemm_thrinfo_t* thread,
                                  thread_comm_t* ocomm, dim_t ocomm_id,
                                  thread_comm_t* icomm, dim_t icomm_id,
                                  dim_t n_way, dim_t work_id, 
                                  packm_thrinfo_t* opackm,
                                  packm_thrinfo_t* ipackm,
                                  gemm_thrinfo_t* sub_gemm );

gemm_thrinfo_t* bli_create_gemm_thrinfo_node( thread_comm_t* ocomm, dim_t ocomm_id,
                                              thread_comm_t* icomm, dim_t icomm_id,
                                              dim_t n_way, dim_t work_id, 
                                              packm_thrinfo_t* opackm,
                                              packm_thrinfo_t* ipackm,
                                              gemm_thrinfo_t* sub_gemm );

void bli_setup_gemm_single_threaded_info( gemm_thrinfo_t* thread );
