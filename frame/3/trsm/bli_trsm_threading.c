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

void bli_setup_trsm_thrinfo_node( trsm_thrinfo_t* thread,
                                             thread_comm_t* ocomm, dim_t ocomm_id,
                                             thread_comm_t* icomm, dim_t icomm_id,
                                             dim_t n_way, dim_t work_id, 
                                             packm_thrinfo_t* opackm,
                                             packm_thrinfo_t* ipackm,
                                             trsm_thrinfo_t* sub_trsm )
{
    thread->ocomm = ocomm;
    thread->ocomm_id = ocomm_id;
    thread->icomm = icomm;
    thread->icomm_id = icomm_id;
    thread->n_way = n_way;
    thread->work_id = work_id;
    thread->opackm = opackm;
    thread->ipackm = ipackm;
    thread->sub_trsm = sub_trsm;
}

void bli_setup_trsm_single_threaded_info( trsm_thrinfo_t* thread )
{
    thread->ocomm = &BLIS_SINGLE_COMM;
    thread->ocomm_id = 0;
    thread->icomm = &BLIS_SINGLE_COMM;
    thread->icomm_id = 0;
    thread->n_way = 1;
    thread->work_id = 0;
    thread->opackm = &BLIS_PACKM_SINGLE_THREADED;
    thread->ipackm = &BLIS_PACKM_SINGLE_THREADED;
    thread->sub_trsm = thread;
}

trsm_thrinfo_t* bli_create_trsm_thrinfo_node( thread_comm_t* ocomm, dim_t ocomm_id,
                                              thread_comm_t* icomm, dim_t icomm_id,
                                              dim_t n_way, dim_t work_id, 
                                              packm_thrinfo_t* opackm,
                                              packm_thrinfo_t* ipackm,
                                              trsm_thrinfo_t* sub_trsm )
{
    trsm_thrinfo_t* thread = ( trsm_thrinfo_t* ) bli_malloc( sizeof( trsm_thrinfo_t ) );
    bli_setup_trsm_thrinfo_node( thread, ocomm, ocomm_id,
                              icomm, icomm_id,
                              n_way, work_id, 
                              opackm,
                              ipackm,
                              sub_trsm );
    return thread;
}

void bli_trsm_thrinfo_free( trsm_thrinfo_t* thread)
{
    if( thread == NULL ) return;

    // Free Communicators
    if( thread_am_ochief( thread ) )
        bli_free_communicator( thread->ocomm );
    if( thread->sub_trsm == NULL && thread_am_ichief( thread ) ) 
        bli_free_communicator( thread->icomm );

    // Free Sub Thrinfos
    bli_packm_thrinfo_free( thread->opackm );
    bli_packm_thrinfo_free( thread->ipackm );
    bli_trsm_thrinfo_free( thread->sub_trsm );
    bli_free( thread );

    return;
}
void bli_trsm_thrinfo_free_paths( trsm_thrinfo_t** threads, dim_t num )
{
    for( int i = 0; i < num; i++)
        bli_trsm_thrinfo_free( threads[i] );
    bli_free( threads );
}

trsm_thrinfo_t** bli_create_trsm_thrinfo_paths( bool_t right_sided )
{
    dim_t jc_way = 1; 
    dim_t kc_way = 1;
    dim_t ic_way = 1;
    dim_t jr_way = 1;
    dim_t ir_way = 1;

#ifdef BLIS_ENABLE_MULTITHREADING 
    dim_t jc_in = bli_read_nway_from_env( "BLIS_JC_NT" );
    /*dim_t kc_in = bli_read_nway_from_env( "BLIS_KC_NT" );*/
    dim_t ic_in = bli_read_nway_from_env( "BLIS_IC_NT" );
    dim_t jr_in = bli_read_nway_from_env( "BLIS_JR_NT" );
    dim_t ir_in = bli_read_nway_from_env( "BLIS_IR_NT" );

    if(right_sided) {
        ic_way = jc_in * ic_in * jr_in;
        ir_way = ir_in;
    }
    else {
        jc_way = jc_in;
        jr_way  = jr_in * ic_in * ir_in;
    }
#endif
    
    dim_t global_num_threads = jc_way * kc_way * ic_way * jr_way * ir_way;
    assert( global_num_threads != 0 );

    dim_t jc_nt  = kc_way * ic_way * jr_way * ir_way;
    dim_t kc_nt  = ic_way * jr_way * ir_way;
    dim_t ic_nt  = jr_way * ir_way;
    dim_t jr_nt  = ir_way;
    dim_t ir_nt  = 1;

    
    trsm_thrinfo_t** paths = (trsm_thrinfo_t**) bli_malloc( global_num_threads * sizeof( trsm_thrinfo_t* ) );

    thread_comm_t*  global_comm = bli_create_communicator( global_num_threads );
    for( int a = 0; a < jc_way; a++ )
    {   
        thread_comm_t*  jc_comm = bli_create_communicator( jc_nt );
        for( int b = 0; b < kc_way; b++ )
        {   
            thread_comm_t* kc_comm = bli_create_communicator( kc_nt );
            for( int c = 0; c < ic_way; c++ )
            {   
                thread_comm_t* ic_comm = bli_create_communicator( ic_nt );
                for( int d = 0; d < jr_way; d++ )
                {   
                    thread_comm_t* jr_comm = bli_create_communicator( jr_nt );
                    for( int e = 0; e < ir_way; e++) 
                    {   
                        thread_comm_t* ir_comm = bli_create_communicator( ir_nt );
                        dim_t ir_comm_id = 0;
                        dim_t jr_comm_id = e*ir_nt + ir_comm_id;
                        dim_t ic_comm_id = d*jr_nt + jr_comm_id;
                        dim_t kc_comm_id = c*ic_nt + ic_comm_id;
                        dim_t jc_comm_id = b*kc_nt + kc_comm_id;
                        dim_t global_comm_id = a*jc_nt + jc_comm_id;
                        

                        // Macrokernel loops 
                        trsm_thrinfo_t* ir_info = bli_create_trsm_thrinfo_node( jr_comm, jr_comm_id,
                                                                                ir_comm, ir_comm_id,
                                                                                ir_way,  e,
                                                                                NULL, NULL, NULL);

                        trsm_thrinfo_t* jr_info = bli_create_trsm_thrinfo_node( ic_comm, ic_comm_id,
                                                                                jr_comm, jr_comm_id,
                                                                                jr_way,  d,
                                                                                NULL, NULL, ir_info);
                        //blk_var_1
                        packm_thrinfo_t* pack_ic_in  = bli_create_packm_thread_info( ic_comm, ic_comm_id,
                                                                            jr_comm, jr_comm_id,
                                                                            ic_nt, ic_comm_id );

                        packm_thrinfo_t* pack_ic_out  = bli_create_packm_thread_info( kc_comm, kc_comm_id,
                                                                            ic_comm, ic_comm_id,
                                                                            kc_nt, kc_comm_id );

                        trsm_thrinfo_t* ic_info = bli_create_trsm_thrinfo_node( kc_comm, kc_comm_id,
                                                                                ic_comm, ic_comm_id,
                                                                                ic_way,  c,
                                                                                pack_ic_out, pack_ic_in, jr_info);
                        //blk_var_3
                        packm_thrinfo_t* pack_kc_in  = bli_create_packm_thread_info( kc_comm, kc_comm_id,
                                                                            ic_comm, ic_comm_id,
                                                                            kc_nt, kc_comm_id );

                        packm_thrinfo_t* pack_kc_out  = bli_create_packm_thread_info( jc_comm, jc_comm_id,
                                                                            jc_comm, jc_comm_id,
                                                                            jc_nt, jc_comm_id );

                        trsm_thrinfo_t* kc_info = bli_create_trsm_thrinfo_node( jc_comm, jc_comm_id,
                                                                                kc_comm, kc_comm_id,
                                                                                kc_way,  b,
                                                                                pack_kc_out, pack_kc_in, ic_info);
                        //blk_var_2
                        packm_thrinfo_t* pack_jc_in  = bli_create_packm_thread_info( jc_comm, jc_comm_id,
                                                                            kc_comm, kc_comm_id,
                                                                            jc_nt, jc_comm_id );

                        packm_thrinfo_t* pack_jc_out  = bli_create_packm_thread_info( global_comm, global_comm_id,
                                                                            jc_comm, jc_comm_id,
                                                                            global_num_threads, global_comm_id );

                        trsm_thrinfo_t* jc_info = bli_create_trsm_thrinfo_node( global_comm, global_comm_id,
                                                                                jc_comm, jc_comm_id,
                                                                                jc_way,  a,
                                                                                pack_jc_out, pack_jc_in, kc_info);

                        paths[global_comm_id] = jc_info;
                    }
                }
            }
        }
    }
    return paths;
}
