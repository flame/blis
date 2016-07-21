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
#include "assert.h"

thrinfo_t* bli_l3_thrinfo_create
     (
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       thrcomm_t* icomm,
       dim_t      icomm_id,
       dim_t      n_way,
       dim_t      work_id,
       thrinfo_t* opackm,
       thrinfo_t* ipackm,
       thrinfo_t* sub_self
     )
{
	return bli_thrinfo_create
	(
	  ocomm, ocomm_id,
	  icomm, icomm_id,
	  n_way,
	  work_id,
	  opackm,
	  ipackm,
	  sub_self
	);
}

void bli_l3_thrinfo_init
     (
       thrinfo_t* thread,
       thrcomm_t* ocomm,
       dim_t      ocomm_id,
       thrcomm_t* icomm,
       dim_t      icomm_id,
       dim_t      n_way,
       dim_t      work_id,
       thrinfo_t* opackm,
       thrinfo_t* ipackm,
       thrinfo_t* sub_self
     )
{
	bli_thrinfo_init
	(
	  thread,
	  ocomm, ocomm_id,
	  icomm, icomm_id,
	  n_way,
	  work_id,
	  opackm,
	  ipackm,
	  sub_self
	);
}

void bli_l3_thrinfo_init_single
     (
       thrinfo_t* thread
     )
{
	bli_thrinfo_init_single( thread );
}

void bli_l3_thrinfo_free
     (
       thrinfo_t* thread
     )
{
	if ( thread == NULL ||
	     thread == &BLIS_GEMM_SINGLE_THREADED ||
	     thread == &BLIS_HERK_SINGLE_THREADED
	   ) return;

	// Free Communicators
	if ( bli_thread_am_ochief( thread ) )
		bli_thrcomm_free( thread->ocomm );
	if ( bli_thrinfo_sub_self( thread ) == NULL && bli_thread_am_ichief( thread ) )
		bli_thrcomm_free( thread->icomm );

	// Free thrinfo chidren
	bli_packm_thrinfo_free( thread->opackm );
	bli_packm_thrinfo_free( thread->ipackm );
	bli_l3_thrinfo_free( thread->sub_self );
	bli_free_intl( thread );
}

// -----------------------------------------------------------------------------

thrinfo_t** bli_l3_thrinfo_create_paths
     (
       opid_t l3_op,
       side_t side
     )
{
	dim_t jc_in, jc_way;
	dim_t kc_in, kc_way;
	dim_t ic_in, ic_way;
	dim_t jr_in, jr_way;
	dim_t ir_in, ir_way;

#ifdef BLIS_ENABLE_MULTITHREADING
	jc_in = bli_env_read_nway( "BLIS_JC_NT" );
	//kc_way = bli_env_read_nway( "BLIS_KC_NT" );
	kc_in = 1;
	ic_in = bli_env_read_nway( "BLIS_IC_NT" );
	jr_in = bli_env_read_nway( "BLIS_JR_NT" );
	ir_in = bli_env_read_nway( "BLIS_IR_NT" );
#else
	jc_in = 1;
	kc_in = 1;
	ic_in = 1;
	jr_in = 1;
	ir_in = 1;
#endif

	if ( l3_op == BLIS_TRMM )
	{
		// We reconfigure the parallelism for trmm_r due to a dependency in
		// the jc loop. (NOTE: This dependency does not exist for trmm3.)
		if ( bli_is_right( side ) )
		{
			jc_way = 1;
			kc_way = kc_in;
			ic_way = ic_in;
			jr_way = jr_in * jc_in;
			ir_way = ir_in;
		}
		else // if ( bli_is_left( side ) )
		{
			jc_way = jc_in;
			kc_way = kc_in;
			ic_way = ic_in;
			jr_way = jr_in;
			ir_way = ir_in;
		}
	}
	else if ( l3_op == BLIS_TRSM )
	{
		if ( bli_is_right( side ) )
		{

			jc_way = 1;
			kc_way = 1;
			ic_way = jc_in * ic_in * jr_in;
			jr_way = 1;
			ir_way = 1;
		}
		else // if ( bli_is_left( side ) )
		{
			jc_way = 1;
			kc_way = 1;
			ic_way = 1;
			jr_way = ic_in * jr_in * ir_in;
			ir_way = 1;
		}
	}
	else // all other level-3 operations
	{
		jc_way = jc_in;
		kc_way = kc_in;
		ic_way = ic_in;
		jr_way = jr_in;
		ir_way = ir_in;
	}


	dim_t global_num_threads = jc_way * kc_way * ic_way * jr_way * ir_way;
	assert( global_num_threads != 0 );

	dim_t jc_nt  = kc_way * ic_way * jr_way * ir_way;
	dim_t kc_nt  = ic_way * jr_way * ir_way;
	dim_t ic_nt  = jr_way * ir_way;
	dim_t jr_nt  = ir_way;
	dim_t ir_nt  = 1;


	thrinfo_t** paths = bli_malloc_intl( global_num_threads * sizeof( thrinfo_t* ) );

	thrcomm_t* global_comm = bli_thrcomm_create( global_num_threads );

	for( int a = 0; a < jc_way; a++ )
	{
		thrcomm_t* jc_comm = bli_thrcomm_create( jc_nt );

		for( int b = 0; b < kc_way; b++ )
		{
			thrcomm_t* kc_comm = bli_thrcomm_create( kc_nt );

			for( int c = 0; c < ic_way; c++ )
			{
				thrcomm_t* ic_comm = bli_thrcomm_create( ic_nt );

				for( int d = 0; d < jr_way; d++ )
				{
					thrcomm_t* jr_comm = bli_thrcomm_create( jr_nt );

					for( int e = 0; e < ir_way; e++ )
					{
						thrcomm_t* ir_comm        = bli_thrcomm_create( ir_nt );
						dim_t      ir_comm_id     = 0;
						dim_t      jr_comm_id     = e*ir_nt + ir_comm_id;
						dim_t      ic_comm_id     = d*jr_nt + jr_comm_id;
						dim_t      kc_comm_id     = c*ic_nt + ic_comm_id;
						dim_t      jc_comm_id     = b*kc_nt + kc_comm_id;
						dim_t      global_comm_id = a*jc_nt + jc_comm_id;

						// Macrokernel loops
						thrinfo_t* ir_info
						=
						bli_l3_thrinfo_create( jr_comm, jr_comm_id,
						                       ir_comm, ir_comm_id,
						                       ir_way, e,
						                       NULL, NULL, NULL );

						thrinfo_t* jr_info
						=
						bli_l3_thrinfo_create( ic_comm, ic_comm_id,
						                       jr_comm, jr_comm_id,
						                       jr_way, d,
						                       NULL, NULL, ir_info );
						//blk_var_1
						thrinfo_t* pack_ic_in
						=
						bli_packm_thrinfo_create( ic_comm, ic_comm_id,
						                          jr_comm, jr_comm_id,
						                          ic_nt, ic_comm_id );

						thrinfo_t* pack_ic_out
						=
						bli_packm_thrinfo_create( kc_comm, kc_comm_id,
						                          ic_comm, ic_comm_id,
						                          kc_nt, kc_comm_id );

						thrinfo_t* ic_info
						=
						bli_l3_thrinfo_create( kc_comm, kc_comm_id,
						                       ic_comm, ic_comm_id,
						                       ic_way, c,
						                       pack_ic_out, pack_ic_in, jr_info );
						//blk_var_3
						thrinfo_t* pack_kc_in
						=
						bli_packm_thrinfo_create( kc_comm, kc_comm_id,
						                          ic_comm, ic_comm_id,
						                          kc_nt, kc_comm_id );

						thrinfo_t* pack_kc_out
						=
						bli_packm_thrinfo_create( jc_comm, jc_comm_id,
						                          jc_comm, jc_comm_id,
						                          jc_nt, jc_comm_id );

						thrinfo_t* kc_info
						=
						bli_l3_thrinfo_create( jc_comm, jc_comm_id,
						                       kc_comm, kc_comm_id,
						                       kc_way, b,
						                       pack_kc_out, pack_kc_in, ic_info );
						//blk_var_2
						thrinfo_t* pack_jc_in
						=
						bli_packm_thrinfo_create( jc_comm, jc_comm_id,
						                          kc_comm, kc_comm_id,
						                          jc_nt, jc_comm_id );

						thrinfo_t* pack_jc_out
						=
						bli_packm_thrinfo_create( global_comm, global_comm_id,
						                          jc_comm, jc_comm_id,
						                          global_num_threads, global_comm_id );

						thrinfo_t* jc_info
						=
						bli_l3_thrinfo_create( global_comm, global_comm_id,
						                       jc_comm, jc_comm_id,
						                       jc_way, a,
						                       pack_jc_out, pack_jc_in, kc_info );

						paths[global_comm_id] = jc_info;
					}
				}
			}
		}
	}

	return paths;
}

void bli_l3_thrinfo_free_paths
     (
       thrinfo_t** threads,
       dim_t       num
     )
{
	dim_t i;

	for ( i = 0; i < num; ++i )
		bli_l3_thrinfo_free( threads[i] );

	bli_free_intl( threads );
}

