/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_THRINFO_UTILS_H
#define LPGEMM_THRINFO_UTILS_H

// Parallelization only supported along jc and ic loops. Thus not reusing the
// existing thrinfo tree logic, since a light-weight work id generation will
// suffice. However the logic used for thread meta data generation, specific
// to jc and ic loops is borrowed.
BLIS_INLINE void lpgemm_gen_thrinfo
     (
       lpgemm_thrinfo_t* thread,
       thrinfo_t* thread_jc,
       thrinfo_t* thread_ic
     )
{
	if ( thread == NULL )
	{
		// Set n_ways=1 to ensure ST behaviour when thread is not initialized.
		// This is the case when BLIS_ENABLE_OPENMP is not defined.
		bli_thrinfo_set_ocomm_id( 0, thread_jc );
		bli_thrinfo_set_n_way( 1, thread_jc );
		bli_thrinfo_set_work_id( 0, thread_jc );

		bli_thrinfo_set_ocomm_id( 0, thread_ic );
		bli_thrinfo_set_n_way( 1, thread_ic );
		bli_thrinfo_set_work_id( 0, thread_ic );
	}
	else
	{
		// Replicate the logic in bli_l3_sup_thrinfo_create_root for jc thrinfo. 
		bli_thrinfo_set_ocomm_id( thread->tid, thread_jc );
		bli_thrinfo_set_n_way( thread->jc_ways, thread_jc );
		dim_t jc_work_id = thread->tid / thread->ic_ways;
		bli_thrinfo_set_work_id( jc_work_id, thread_jc );

		// Replicate the sub node creation logic in bli_thrinfo_sup_create_for_cntl
		// for ic thrinfo. 
		dim_t ic_comm_id = thread->tid % thread->ic_ways;
		bli_thrinfo_set_ocomm_id( ic_comm_id, thread_ic );
		bli_thrinfo_set_n_way( thread->ic_ways, thread_ic );
		bli_thrinfo_set_work_id( ic_comm_id, thread_ic );
	}
}

#endif //LPGEMM_THRINFO_UTILS_H
