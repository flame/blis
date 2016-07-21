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

#if 0
thrinfo_t** bli_gemm_thrinfo_create_paths( void )
{

#ifdef BLIS_ENABLE_MULTITHREADING
    dim_t jc_way = bli_env_read_nway( "BLIS_JC_NT" );
//    dim_t kc_way = bli_env_read_nway( "BLIS_KC_NT" );
    dim_t kc_way = 1;
    dim_t ic_way = bli_env_read_nway( "BLIS_IC_NT" );
    dim_t jr_way = bli_env_read_nway( "BLIS_JR_NT" );
    dim_t ir_way = bli_env_read_nway( "BLIS_IR_NT" );
#else
    dim_t jc_way = 1;
    dim_t kc_way = 1;
    dim_t ic_way = 1;
    dim_t jr_way = 1;
    dim_t ir_way = 1;
#endif

    dim_t global_num_threads = jc_way * kc_way * ic_way * jr_way * ir_way;
    assert( global_num_threads != 0 );

    dim_t jc_nt  = kc_way * ic_way * jr_way * ir_way;
    dim_t kc_nt  = ic_way * jr_way * ir_way;
    dim_t ic_nt  = jr_way * ir_way;
    dim_t jr_nt  = ir_way;
    dim_t ir_nt  = 1;

    
    thrinfo_t** paths = bli_malloc_intl( global_num_threads * sizeof( thrinfo_t* ) );

    thrcomm_t*  global_comm = bli_thrcomm_create( global_num_threads );
    for( int a = 0; a < jc_way; a++ )
    {
        thrcomm_t*  jc_comm = bli_thrcomm_create( jc_nt );
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
                        thrcomm_t* ir_comm = bli_thrcomm_create( ir_nt );
                        dim_t ir_comm_id = 0;
                        dim_t jr_comm_id = e*ir_nt + ir_comm_id;
                        dim_t ic_comm_id = d*jr_nt + jr_comm_id;
                        dim_t kc_comm_id = c*ic_nt + ic_comm_id;
                        dim_t jc_comm_id = b*kc_nt + kc_comm_id;
                        dim_t global_comm_id = a*jc_nt + jc_comm_id;

                        // Macrokernel loops
                        thrinfo_t* ir_info = bli_l3_thrinfo_create_node( jr_comm, jr_comm_id,
                                                                                ir_comm, ir_comm_id,
                                                                                ir_way,  e,
                                                                                NULL, NULL, NULL);

                        thrinfo_t* jr_info = bli_l3_thrinfo_create_node( ic_comm, ic_comm_id,
                                                                                jr_comm, jr_comm_id,
                                                                                jr_way,  d,
                                                                                NULL, NULL, ir_info);
                        //blk_var_1
                        packm_thrinfo_t* pack_ic_in  = bli_packm_thrinfo_create( ic_comm, ic_comm_id,
                                                                            jr_comm, jr_comm_id,
                                                                            ic_nt, ic_comm_id );

                        packm_thrinfo_t* pack_ic_out  = bli_packm_thrinfo_create( kc_comm, kc_comm_id,
                                                                            ic_comm, ic_comm_id,
                                                                            kc_nt, kc_comm_id );

                        thrinfo_t* ic_info = bli_l3_thrinfo_create_node( kc_comm, kc_comm_id,
                                                                                ic_comm, ic_comm_id,
                                                                                ic_way,  c,
                                                                                pack_ic_out, pack_ic_in, jr_info);
                        //blk_var_3
                        packm_thrinfo_t* pack_kc_in  = bli_packm_thrinfo_create( kc_comm, kc_comm_id,
                                                                            ic_comm, ic_comm_id,
                                                                            kc_nt, kc_comm_id );

                        packm_thrinfo_t* pack_kc_out  = bli_packm_thrinfo_create( jc_comm, jc_comm_id,
                                                                            jc_comm, jc_comm_id,
                                                                            jc_nt, jc_comm_id );

                        thrinfo_t* kc_info = bli_l3_thrinfo_create_node( jc_comm, jc_comm_id,
                                                                                kc_comm, kc_comm_id,
                                                                                kc_way,  b,
                                                                                pack_kc_out, pack_kc_in, ic_info);
                        //blk_var_2
                        packm_thrinfo_t* pack_jc_in  = bli_packm_thrinfo_create( jc_comm, jc_comm_id,
                                                                            kc_comm, kc_comm_id,
                                                                            jc_nt, jc_comm_id );

                        packm_thrinfo_t* pack_jc_out  = bli_packm_thrinfo_create( global_comm, global_comm_id,
                                                                            jc_comm, jc_comm_id,
                                                                            global_num_threads, global_comm_id );

                        thrinfo_t* jc_info = bli_l3_thrinfo_create_node( global_comm, global_comm_id,
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
#endif
