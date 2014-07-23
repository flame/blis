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


struct trsm_thrinfo_s //implements thrinfo_t
{
    thread_comm_t*      ocomm;       //The thread communicator for the other threads sharing the same work at this level
    dim_t               ocomm_id;    //Our thread id within that thread comm
    thread_comm_t*      icomm;       //The thread communicator for the other threads sharing the same work at this level
    dim_t               icomm_id;    //Our thread id within that thread comm

    dim_t               n_way;       //Number of distinct caucuses used to parallelize the loop
    dim_t               work_id;     //What we're working on

    packm_thrinfo_t*    opackm;
    packm_thrinfo_t*    ipackm;
    struct trsm_thrinfo_s*    sub_trsm;
};
typedef struct trsm_thrinfo_s trsm_thrinfo_t;

#define trsm_thread_sub_trsm( thread )  thread->sub_trsm
#define trsm_thread_sub_opackm( thread )  thread->opackm
#define trsm_thread_sub_ipackm( thread )  thread->ipackm

#define trsm_my_iter( index, thread ) ( index % thread->n_way == thread->work_id % thread->n_way )

trsm_thrinfo_t** bli_create_trsm_thrinfo_paths( bool_t right_sided );
void bli_trsm_thrinfo_free_paths( trsm_thrinfo_t** info, dim_t n_threads );

void bli_setup_trsm_thrinfo_node( trsm_thrinfo_t* thread,
                                  thread_comm_t* ocomm, dim_t ocomm_id,
                                  thread_comm_t* icomm, dim_t icomm_id,
                                  dim_t n_way, dim_t work_id, 
                                  packm_thrinfo_t* opackm,
                                  packm_thrinfo_t* ipackm,
                                  trsm_thrinfo_t* sub_trsm );

trsm_thrinfo_t* bli_create_trsm_thrinfo_node( thread_comm_t* ocomm, dim_t ocomm_id,
                                              thread_comm_t* icomm, dim_t icomm_id,
                                              dim_t n_way, dim_t work_id, 
                                              packm_thrinfo_t* opackm,
                                              packm_thrinfo_t* ipackm,
                                              trsm_thrinfo_t* sub_trsm );

void bli_setup_trsm_single_threaded_info( trsm_thrinfo_t* thread );
