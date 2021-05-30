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

#include "bls_gemm_var.h"
#include "bls_packm.h"

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

void bls_gemm_bp_var2_jr_ir
     (
       tci_comm* thread,
       uint64_t jr_start,
       uint64_t jr_end,
       void* parent_cntx
     )
{
    bls_gemm_bp_cntx_t cntx = *(bls_gemm_bp_cntx_t*)parent_cntx;
    num_t dt = cntx.dt;

    const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx.cntx );
    const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx.cntx );

    auxinfo_t aux;

    /* Query the context for the microkernel address and cast it to its
       function pointer type. */
    gemm_ukr_ft gemm_ukr = bli_cntx_get_l3_nat_ukr_dt( dt, BLIS_GEMM_UKR, cntx.cntx );

    /* Temporary C buffer for edge cases. Note that the strides of this
       temporary buffer are set so that they match the storage of the
       original C matrix. For example, if C is column-stored, ct will be
       column-stored as well. */
    char       ct[ BLIS_STACK_BUF_MAX_SIZE
                    / sizeof( char ) ]
                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
    const bool col_pref = bli_cntx_l3_nat_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx.cntx );
    const inc_t rs_ct   = ( col_pref ? 1 : NR );
    const inc_t cs_ct   = ( col_pref ? MR : 1 );

    char* zero = bli_obj_buffer_for_1x1( dt, &BLIS_ZERO );
    char* alpha = cntx.alpha;
    char* beta = cntx.beta;

    dim_t m = cntx.m;
    dim_t n = cntx.n;
    dim_t k = cntx.k;

    inc_t rs_c = cntx.rs_c;
    inc_t cs_c = cntx.cs_c;

    char* a_ic_use = cntx.a;
    char* b_pc_use = cntx.b;
    char* c_ic     = cntx.c;
    inc_t ps_b_use = cntx.ps_b * bli_dt_size( dt );
    inc_t jrstep_c = cs_c * NR * bli_dt_size( dt );
    inc_t ps_a_use = cntx.ps_a * bli_dt_size( dt );
    inc_t irstep_c = rs_c * MR * bli_dt_size( dt );

    /* Compute number of primary and leftover components of the JR loop. */
    jr_start /= NR;
    jr_end = ( jr_end + NR - 1 ) / NR;
    dim_t jr_left = ( ( n + NR - 1 ) % NR ) + 1;

    /* Compute number of primary and leftover components of the IR loop. */
    dim_t ir_start = 0;
    dim_t ir_end = ( m + MR - 1 ) / MR;
    dim_t ir_left = ( ( m  + MR - 1 ) % MR ) + 1;

    /* Loop over the n dimension (NR columns at a time). */
    for ( dim_t j = jr_start; j < jr_end; j += 1 )
    {
        const dim_t nr_cur = ( j * NR + jr_left == n ? jr_left : NR );

        char* b_jr = b_pc_use + j * ps_b_use;
        char* c_jr = c_ic     + j * jrstep_c;

        /* Loop over the m dimension (MR rows at a time). */
        for ( dim_t i = ir_start; i < ir_end; i += 1 )
        {
            const dim_t mr_cur = ( i * MR + ir_left == m ? ir_left : MR );

            char* a_ir = a_ic_use + i * ps_a_use;
            char* c_ir = c_jr     + i * irstep_c;

            /* Compute the addresses of the next micropanels of A and B. */
            char* a2 = bli_gemm_get_next_a_upanel( a_ir, ps_a_use, 1 );
            /* Assume for now that our next panel of B to be the current panel
               of B. */
            char* b2 = b_jr;

            if ( i * MR + ir_left == m )
            {
                a2 = a_ic_use;
                b2 = bli_gemm_get_next_b_upanel( b_jr, ps_b_use, 1 );
                if ( j * NR + jr_left == n )
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
                  k,
                  alpha,
                  a_ir,
                  b_jr,
                  beta,
                  c_ir, rs_c, cs_c,
                  &aux,
                  cntx.cntx
                );
            }
            else
            {
                /* Invoke the gemm microkernel. */
                gemm_ukr
                (
                  k,
                  alpha,
                  a_ir,
                  b_jr,
                  zero,
                  ct, rs_ct, cs_ct,
                  &aux,
                  cntx.cntx
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
}

void bls_gemm_bp_var2_ic
     (
       tci_comm* thread,
       uint64_t ic_start,
       uint64_t ic_end,
       void* parent_cntx
     )
{
    bls_gemm_bp_cntx_t cntx = *(bls_gemm_bp_cntx_t*)parent_cntx;
    num_t dt = cntx.dt;

    const dim_t KC = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx.cntx );
    const dim_t MC = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx.cntx );
    const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx.cntx );
    const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx.cntx );

    mem_t mem_a = BLIS_MEM_INITIALIZER;

    dim_t n = cntx.n;
    dim_t k = cntx.k;

    char* a_pc = cntx.a;
    char* c_jc = cntx.c;
    inc_t icstep_a = cntx.rs_a * bli_dt_size( dt );
    inc_t icstep_c = cntx.rs_c * bli_dt_size( dt );

    /* Loop over the m dimension (MC rows at a time). */
    for ( dim_t ii = ic_start; ii < ic_end; ii += MC )
    {
        /* Calculate the thread's current IC block dimension. */
        const dim_t mc_cur = bli_min( MC, ic_end - ii );

        char* a_ic = a_pc + ii * icstep_a;
        char* c_ic = c_jc + ii * icstep_c;

        char* a_use;
        inc_t ps_a_use;

        /* Determine the packing buffer and related parameters for matrix
           A. Then call the packm implementation. */
        bls_packm
        (
          BLIS_M,
          dt,
          cntx.conja,
          MC,     KC,
          mc_cur, k, MR,
          a_ic,   cntx.rs_a, cntx.cs_a,
          &a_use, &ps_a_use,
          cntx.cntx,
          cntx.rntm,
          &mem_a,
          cntx.thread_ic
        );

        /* Update the context for the next lower loop. */
        cntx.a = a_use;
        cntx.ps_a = ps_a_use;
        cntx.c = c_ic;
        cntx.m = mc_cur;

        /* The below only partitions work along the jr loop. Partitioning along
           both ir and jr is easily possible by distributing threads over tasks
           and computing the work allocation from the task ID. */
        tci_comm_distribute_over_threads( cntx.thread_ic, (tci_range){ n, NR }, bls_gemm_bp_var2_jr_ir, &cntx );
    }

    /* Release any memory that was acquired for packing matrix A. */
    bls_packm_finalize_mem
    (
      cntx.rntm,
      &mem_a,
      cntx.thread_jc
    );
}

void bls_gemm_bp_var2_jc_kc
     (
       tci_comm* thread,
       uint64_t jc_start,
       uint64_t jc_end,
       void* parent_cntx
     )
{
    bls_gemm_bp_cntx_t cntx = *(bls_gemm_bp_cntx_t*)parent_cntx;
    num_t dt = cntx.dt;

    char* one = bli_obj_buffer_for_1x1( dt, &BLIS_ONE );

    const dim_t NC = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx.cntx );
    const dim_t KC = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx.cntx );
    const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx.cntx );
    const dim_t MR = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx.cntx );

    dim_t m = cntx.m;
    dim_t k = cntx.k;

    char* a_00 = cntx.a;
    char* b_00 = cntx.b;
    char* c_00 = cntx.c;
    inc_t jcstep_b = cntx.cs_b * bli_dt_size( dt );
    inc_t jcstep_c = cntx.cs_c * bli_dt_size( dt );
    inc_t pcstep_a = cntx.cs_a * bli_dt_size( dt );
    inc_t pcstep_b = cntx.rs_b * bli_dt_size( dt );

    mem_t mem_b = BLIS_MEM_INITIALIZER;

    /* Loop over the n dimension (NC rows/columns at a time). */
    for ( dim_t jj = jc_start; jj < jc_end; jj += NC )
    {
        /* Calculate the thread's current JC block dimension. */
        const dim_t nc_cur = bli_min( NC, jc_end - jj );

        char* b_jc = b_00 + jj * jcstep_b;
        char* c_jc = c_00 + jj * jcstep_c;

        /* Loop over the k dimension (KC rows/columns at a time). */
        for ( dim_t pp = 0; pp < k; pp += KC )
        {
            /* Calculate the thread's current PC block dimension. */
            const dim_t kc_cur = bli_min( KC, k - pp );

            char* a_pc = a_00 + pp * pcstep_a;
            char* b_pc = b_jc + pp * pcstep_b;

            char* b_use;
            inc_t ps_b_use;

            /* Determine the packing buffer and related parameters for matrix
               B. Then call the packm implementation. Reverse rs_b and cs_b
               so that we can pack B^T and A the same way. */
            bls_packm
            (
              BLIS_N,
              dt,
              cntx.conjb,
              NC,     KC,
              nc_cur, kc_cur, NR,
              b_pc,   cntx.cs_b, cntx.rs_b,
              &b_use, &ps_b_use,
              cntx.cntx,
              cntx.rntm,
              &mem_b,
              cntx.thread_jc
            );

            /* Update the context for the next lower loop. */
            cntx.a = a_pc;
            cntx.b = b_use;
            cntx.ps_b = ps_b_use;
            cntx.c = c_jc;
            cntx.n = nc_cur;
            cntx.k = kc_cur;

            tci_comm_distribute_over_gangs( cntx.thread_ic, (tci_range){ m, MR }, bls_gemm_bp_var2_ic, &cntx );

            /* Beta needs to be set to 1.0 for all but the first iteration of the pc loop. */
            cntx.beta = one;

            /* This barrier is needed to prevent threads from starting to pack
               the next row panel of B before the current row panel is fully
               computed upon. */
            tci_comm_barrier( cntx.thread_jc );
        }
    }

    /* Release any memory that was acquired for packing matrix B. */
    bls_packm_finalize_mem
    (
      cntx.rntm,
      &mem_b,
      cntx.thread_jc
    );
}

void bls_gemm_bp_var2
     (
       tci_comm* thread,
       void* parent_cntx
     )
{
    bls_gemm_bp_cntx_t cntx = *(bls_gemm_bp_cntx_t*)parent_cntx;

	const num_t dt = cntx.dt;
    const dim_t NR = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx.cntx );

    tci_comm thread_jc;
    tci_comm thread_ic;

    /* Split the thread communicators repeatedly based on the requested
       parallelism. */
    tci_comm_gang( thread, &thread_jc, TCI_EVENLY,
                   bli_rntm_ways_for( BLIS_NC, cntx.rntm ), 0 );
    tci_comm_gang( &thread_jc, &thread_ic, TCI_EVENLY,
                   bli_rntm_ways_for( BLIS_MC, cntx.rntm ), 0 );

    cntx.thread_jc = &thread_jc;
    cntx.thread_ic = &thread_ic;

    /* tci_comm_distribute* assigns work by partitioning the range [0,ntask),
       here cntx.n. The second paramaeter of tci_range is the granularity, i.e.
       thread work partitions only start and end on multiples of the granularity
       (except the last task whose partition ends at ntask-1 exactly).
       tci_comm_distribute_over_gangs assigns the same work to all threads in
       the sub-communicator, and different work to each sub-communicator that
       was split from the parent communicator. */
    tci_comm_distribute_over_gangs( &thread_jc, (tci_range){ cntx.n, NR }, bls_gemm_bp_var2_jc_kc, &cntx );
}

