/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

#include "bls_gemm_var.hpp"
#include "bls_packm.hpp"

#include <algorithm>

//
// -- gemm-like microkernel wrapper --------------------------------------------
//

typedef void (*gemm_ukr_ft)
     (
       dim_t      k,
       void*      alpha,
       void*      a,
       void*      b,
       void*      beta,
       void*      c, inc_t rs_c, inc_t cs_c,
       auxinfo_t* data,
       cntx_t*    cntx
     );

typedef void (*xpbys_mxn_ft)
     (
       dim_t m,
       dim_t n,
       void* x, inc_t rs_x, inc_t cs_x,
       void* beta,
       void* y, inc_t rs_y, inc_t cs_y
     );

static xpbys_mxn_ft xpbys_mxn[4] =
{
  (xpbys_mxn_ft)bli_sxpbys_mxn,
  (xpbys_mxn_ft)bli_cxpbys_mxn,
  (xpbys_mxn_ft)bli_dxpbys_mxn,
  (xpbys_mxn_ft)bli_zxpbys_mxn
};

//
// -- gemm-like block-panel algorithm (object interface) -----------------------
//

void bls_gemm_bp_var2
     (
         num_t               dt,
         conj_t              conja,
         conj_t              conjb,
         dim_t               m,
         dim_t               n,
         dim_t               k,
         char*               alpha,
         char*               a, inc_t rs_a, inc_t cs_a,
         char*               b, inc_t rs_b, inc_t cs_b,
         char*               beta,
         char*               c, inc_t rs_c, inc_t cs_c,
         cntx_t*             cntx,
         rntm_t*             rntm,
         const communicator& thread
     )
{
    const auto dt_size = bli_dt_size( dt );

    const auto NC = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx );
    const auto KC = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx );
    const auto MC = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx );
    const auto NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx );
    const auto MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx );

    auto one  = (char*)bli_obj_buffer_for_1x1( dt, &BLIS_ONE );
    auto zero = (char*)bli_obj_buffer_for_1x1( dt, &BLIS_ZERO );

    /* Query the context for the microkernel address and cast it to its
       function pointer type. */
    auto gemm_ukr = (gemm_ukr_ft)bli_cntx_get_l3_nat_ukr_dt( dt, BLIS_GEMM_UKR, cntx );

    const auto col_pref = bli_cntx_l3_nat_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx );
    const auto rs_ct    = ( col_pref ? 1 : NR );
    const auto cs_ct    = ( col_pref ? MR : 1 );

    /* Split the thread communicators repeatedly based on the requested
       parallelism. */
    auto thread_jc = thread   .gang( TCI_EVENLY, bli_rntm_ways_for( BLIS_NC, rntm ) );
    auto thread_ic = thread_jc.gang( TCI_EVENLY, bli_rntm_ways_for( BLIS_MC, rntm ) );

    /* tci_comm_distribute* assigns work by partitioning the range [0,ntask),
       here n. The second paramaeter of tci_range is the granularity, i.e.
       thread work partitions only start and end on multiples of the granularity
       (except the last task whose partition ends at ntask-1 exactly).
       tci_comm_distribute_over_gangs assigns the same work to all threads in
       the sub-communicator, and different work to each sub-communicator that
       was split from the parent communicator. */
    thread_jc.distribute_over_gangs({ n, NR },
    [&,beta](dim_t jc_start, dim_t jc_end) mutable
    {
        auto jcstep_b = cs_b * dt_size;
        auto jcstep_c = cs_c * dt_size;

        MemBuffer mem_b;

        /* Loop over the n dimension (NC rows/columns at a time). */
        for ( auto jj = jc_start; jj < jc_end; jj += NC )
        {
            /* Calculate the thread's current JC block dimension. */
            const auto nc_cur = std::min( NC, jc_end - jj );

            auto b_jc = b + jj * jcstep_b;
            auto c_jc = c + jj * jcstep_c;

            auto pcstep_a = cs_a * dt_size;
            auto pcstep_b = rs_b * dt_size;

            /* Loop over the k dimension (KC rows/columns at a time). */
            for ( auto pp = dim_t{}; pp < k; pp += KC )
            {
                /* Calculate the thread's current PC block dimension. */
                const auto kc_cur = std::min( KC, k - pp );

                auto a_pc = a    + pp * pcstep_a;
                auto b_pc = b_jc + pp * pcstep_b;

                char* b_pc_use;
                inc_t ps_b;

                /* Determine the packing buffer and related parameters for matrix
                   B. Then call the packm implementation. Reverse rs_b and cs_b
                   so that we can pack B^T and A the same way. */
                bls_packm
                (
                  BLIS_N,
                  dt,
                  conjb,
                  NC,       KC,
                  nc_cur,   kc_cur, NR,
                  b_pc,     cs_b, rs_b,
                  b_pc_use, ps_b,
                  cntx,
                  rntm,
                  mem_b,
                  thread_jc
                );

                thread_ic.distribute_over_gangs({m, MR},
                [&](dim_t ic_start, dim_t ic_end)
                {
                    auto icstep_a = rs_a * dt_size;
                    auto icstep_c = rs_c * dt_size;

                    MemBuffer mem_a;

                    /* Loop over the m dimension (MC rows at a time). */
                    for ( auto ii = ic_start; ii < ic_end; ii += MC )
                    {
                        /* Calculate the thread's current IC block dimension. */
                        const auto mc_cur = std::min( MC, ic_end - ii );

                        auto a_ic = a_pc + ii * icstep_a;
                        auto c_ic = c_jc + ii * icstep_c;

                        char* a_ic_use;
                        inc_t ps_a;

                        /* Determine the packing buffer and related parameters for matrix
                           A. Then call the packm implementation. */
                        bls_packm
                        (
                          BLIS_M,
                          dt,
                          conja,
                          MC,       KC,
                          mc_cur,   kc_cur, MR,
                          a_ic,     rs_a, cs_a,
                          a_ic_use, ps_a,
                          cntx,
                          rntm,
                          mem_a,
                          thread_ic
                        );

                        /* The below only partitions work along the jr loop. Partitioning along
                           both ir and jr is easily possible by distributing threads over tasks
                           and computing the work allocation from the task ID. */
                        thread_ic.distribute_over_threads({ nc_cur, NR },
                        [&](dim_t jr_start, dim_t jr_end)
                        {
                            auxinfo_t aux;

                            /* Temporary C buffer for edge cases. Note that the strides of this
                               temporary buffer are set so that they match the storage of the
                               original C matrix. For example, if C is column-stored, ct will be
                               column-stored as well. */
                            char ct[ BLIS_STACK_BUF_MAX_SIZE ]
                                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));

                            auto ps_b_use = ps_b * dt_size;
                            auto jrstep_c = cs_c * NR * dt_size;
                            auto ps_a_use = ps_a * dt_size;
                            auto irstep_c = rs_c * MR * dt_size;

                            /* Compute number of primary and leftover components of the JR loop. */
                            jr_start /= NR;
                            jr_end = ( jr_end + NR - 1 ) / NR;
                            auto jr_left = ( ( nc_cur + NR - 1 ) % NR ) + 1;

                            /* Compute number of primary and leftover components of the IR loop. */
                            auto ir_start = dim_t{};
                            auto ir_end = ( mc_cur + MR - 1 ) / MR;
                            auto ir_left = ( ( mc_cur  + MR - 1 ) % MR ) + 1;

                            /* Loop over the n dimension (NR columns at a time). */
                            for ( auto j = jr_start; j < jr_end; j += 1 )
                            {
                                auto jr_last = j * NR + jr_left == nc_cur;
                                const auto nr_cur = ( jr_last ? jr_left : NR );

                                auto b_jr = b_pc_use + j * ps_b_use;
                                auto c_jr = c_ic     + j * jrstep_c;

                                /* Loop over the m dimension (MR rows at a time). */
                                for ( dim_t i = ir_start; i < ir_end; i += 1 )
                                {
                                    auto ir_last = i * MR + ir_left == mc_cur;
                                    const auto mr_cur = ( ir_last ? ir_left : MR );

                                    auto a_ir = a_ic_use + i * ps_a_use;
                                    auto c_ir = c_jr     + i * irstep_c;

                                    /* Compute the addresses of the next micropanels of A and B. */
                                    auto a2 = bli_gemm_get_next_a_upanel( a_ir, ps_a_use, 1 );
                                    /* Assume for now that our next panel of B to be the current panel
                                       of B. */
                                    auto b2 = b_jr;

                                    if ( ir_last )
                                    {
                                        a2 = a_ic_use;
                                        b2 = bli_gemm_get_next_b_upanel( b_jr, ps_b_use, 1 );
                                        if ( jr_last )
                                            b2 = b_pc_use;
                                    }

                                    /* Save the addresses of next micropanels of A and B to the
                                       auxinfo_t object. */
                                    bli_auxinfo_set_next_a( a2, &aux );
                                    bli_auxinfo_set_next_b( b2, &aux );

                                    /* Handle interior and edge cases separately. */
                                    if ( mr_cur == MR && nr_cur == NR )
                                    {
                                        /* Invoke the gemm microkernel. */
                                        gemm_ukr
                                        (
                                          kc_cur,
                                          alpha,
                                          a_ir,
                                          b_jr,
                                          beta,
                                          c_ir, rs_c, cs_c,
                                          &aux,
                                          cntx
                                        );
                                    }
                                    else
                                    {
                                        /* Invoke the gemm microkernel. */
                                        gemm_ukr
                                        (
                                          kc_cur,
                                          alpha,
                                          a_ir,
                                          b_jr,
                                          zero,
                                          ct, rs_ct, cs_ct,
                                          &aux,
                                          cntx
                                        );

                                        /* Scale the bottom edge of C and add the result from above. */
                                        xpbys_mxn[dt]
                                        (
                                          mr_cur,
                                          nr_cur,
                                          ct, rs_ct, cs_ct,
                                          beta,
                                          c_ir,  rs_c,  cs_c
                                        );
                                    }
                                }
                            }
                        });
                    }
                });

                /* Beta needs to be set to 1.0 for all but the first iteration of the pc loop. */
                beta = one;

                /* This barrier is needed to prevent threads from starting to pack
                   the next row panel of B before the current row panel is fully
                   computed upon. */
                thread_jc.barrier();
            }
        }
    });
}

