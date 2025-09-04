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
#include "../base/bli_pack_compute_utils.h"

void bli_gemm_compute_init
(
    obj_t* a,
    obj_t* b,
    obj_t* beta,
    obj_t* c,
    cntx_t* cntx,
    rntm_t* rntm
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);

    if ( bli_error_checking_is_enabled() )
    {
        // @todo: Add call to error checking function here
    }

    // Initializing the cntx if one isn't already passed.
    if ( cntx == NULL ) {
        cntx = bli_gks_query_cntx();
    }

    // Initialize a local runtime with global settings if necessary. Note
    // that in the case that a runtime is passed in, we make a local copy.
    rntm_t rntm_l;
    if ( rntm == NULL )
    {
        bli_rntm_init_from_global( &rntm_l );
        rntm = &rntm_l;
    }
    else
    {
        rntm_l = *rntm;
        rntm = &rntm_l;
    }

    // @todo: AOCL Dynamic yet to be implemented for pack-compute APIs.
#ifdef AOCL_DYNAMIC
    // If dynamic-threading is enabled, calculate optimum number
    //  of threads.
    //  rntm will be updated with optimum number of threads.

    // bli_nthreads_optimum(a, b, c, BLIS_GEMM, rntm );
#endif

    bli_rntm_set_ways_from_rntm_sup
    (
      bli_obj_length( c ),
      bli_obj_width( c ),
      bli_obj_width( a ),
      rntm
    );

    bli_l3_compute_thread_decorator
    (
        bli_gemm_compute,
        BLIS_GEMM,
        a,
        b,
        beta,
        c,
        cntx,
        rntm
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);
}

void bli_gemm_compute
(
    obj_t*     a,
    obj_t*     b,
    obj_t*     beta,
    obj_t*     c,
    cntx_t*    cntx,
    rntm_t*    rntm,
    thrinfo_t* thread
)
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4);

    const num_t  dt     = bli_obj_dt( c );
    const dim_t  m      = bli_obj_length( c );
    const dim_t  n      = bli_obj_width( c );
          dim_t  k      = bli_obj_width( a );

    void* restrict buf_a = bli_obj_buffer_at_off( a );
          inc_t    rs_a;
          inc_t    cs_a;

    void* restrict buf_b = bli_obj_buffer_at_off( b );
          inc_t    rs_b;
          inc_t    cs_b;

    stor3_t    stor_id  = bli_obj_stor3_from_strides( c, a, b );
    const bool row_pref = bli_cntx_l3_sup_ker_prefers_rows_dt( dt, stor_id, cntx );

    // packedX defines whether matrix X is pre-packed (reordered) or not.
    bool packeda = bli_obj_is_packed( a );
    bool packedb = bli_obj_is_packed( b );

    // packX defines whether to pack matrix X on-the-go or not.
    bool packa = bli_rntm_pack_a( rntm );
    bool packb = bli_rntm_pack_b( rntm );
    const bool transa = bli_obj_has_trans( a );
    const bool transb = bli_obj_has_trans( b );

    // is_col_stored_a = TRUE when,
    //  A is col stored and not transposed,
    //  or, A is row stored and transposed.
    const bool is_col_stored_a = bli_obj_is_col_stored( a ) && !transa;

    // is_row_stored_b = TRUE when,
    //  B is row stored and not transposed,
    //  or, B is col stored and transposed.
    const bool is_row_stored_b = bli_obj_is_row_stored( b ) && !transb;

    // If kernel is row-preferred but B is not row-stored and unpacked,
    // enable on-the-go packing of B.
    // Else if kernel is col-preferred but A is not col-stored and unpacked,
    // enable on-the-go packing of A.
    if ( row_pref )
    {
        if ( !packedb && !is_row_stored_b ) packb = TRUE;
    }
    else // if ( col_pref )
    {
        if ( !packeda && !is_col_stored_a ) packa = TRUE;
    }

    if ( bli_obj_has_notrans( a ) )
    {
        k     = bli_obj_width( a );

        rs_a  = bli_obj_row_stride( a );
        cs_a  = bli_obj_col_stride( a );
    }
    else // if ( bli_obj_has_trans( a ) )
    {
        // Assign the variables with an implicit transposition.
        k     = bli_obj_length( a );

        rs_a  = bli_obj_col_stride( a );
        cs_a  = bli_obj_row_stride( a );
    }

    if ( bli_obj_has_notrans( b ) )
    {
        rs_b = bli_obj_row_stride( b );
        cs_b = bli_obj_col_stride( b );
    }
    else // if ( bli_obj_has_trans( b ) )
    {
        rs_b = bli_obj_col_stride( b );
        cs_b = bli_obj_row_stride( b );
    }

    void* restrict buf_c    = bli_obj_buffer_at_off( c );
    const inc_t    rs_c     = bli_obj_row_stride( c );
    const inc_t    cs_c     = bli_obj_col_stride( c );

    void* restrict buf_beta = bli_obj_buffer_for_1x1( dt, beta );

    // Setting the packing status in rntm.
    if ( packa ) bli_rntm_set_pack_a( 1, rntm );
    else         bli_rntm_set_pack_a( 0, rntm );

    if ( packb ) bli_rntm_set_pack_b( 1, rntm );
    else         bli_rntm_set_pack_b( 0, rntm );

    if ( bli_is_float( dt ) )
    {
        PASTEMAC( s, gemm_compute )
        (
          packa,
          packb,
          packeda,
          packedb,
          m,
          n,
          k,
          buf_a, rs_a, cs_a,
          buf_b, rs_b, cs_b,
          buf_beta,
          buf_c, rs_c, cs_c,
          BLIS_RRR,     // Using BLIS_RRR since we want to redirect to m kernels.
          cntx,
          rntm,
          thread
        );
    }
    else if ( bli_is_double( dt ) )
    {
        PASTEMAC( d, gemm_compute )
        (
          packa,
          packb,
          packeda,
          packedb,
          m,
          n,
          k,
          buf_a, rs_a, cs_a,
          buf_b, rs_b, cs_b,
          buf_beta,
          buf_c, rs_c, cs_c,
          BLIS_RRR,     // Using BLIS_RRR since we want to redirect to m kernels.
          cntx,
          rntm,
          thread
        );
    }

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4);

}

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC( ch, varname ) \
      ( \
        bool             packa, \
        bool             packb, \
        bool             packeda, \
        bool             packedb, \
        dim_t            m, \
        dim_t            n, \
        dim_t            k, \
        void*   restrict a, inc_t rs_a, inc_t cs_a, \
        void*   restrict b, inc_t rs_b, inc_t cs_b, \
        void*   restrict beta, \
        void*   restrict c, inc_t rs_c, inc_t cs_c, \
        stor3_t          stor_id, \
        cntx_t* restrict cntx, \
        rntm_t* restrict rntm, \
        thrinfo_t* restrict thread \
      ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5); \
\
    const num_t dt = PASTEMAC( ch, type ); \
\
    /* If m or n is zero, return immediately. */ \
    if ( bli_zero_dim2( m, n ) ) return; \
\
    /* @todo Add early return for k < 1 or alpha = 0 here. */ \
\
    /* Query the context for various blocksizes. */ \
    const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
    const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
    const dim_t NC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
    const dim_t MC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
    const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
    /* @note: Modifications of KC are just a part of optimizations.
        Such optimizations have been removed for simplicity and will be a part
        of the optimizations patch. */ \
    dim_t KC; \
    KC = KC0; \
\
    /* Query the maximum blocksize for NR, which implies a maximum blocksize
       extension for the final iteration. */ \
    const dim_t NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx ); \
    const dim_t NRE = NRM - NR; \
\
    /* Compute partitioning step values for each matrix of each loop. */ \
    const inc_t jcstep_c = cs_c; \
    const inc_t jcstep_b = cs_b; \
\
    const inc_t jcstep_b_use = k; \
\
    const inc_t pcstep_a = cs_a; \
    const inc_t pcstep_b = rs_b; \
\
    const inc_t icstep_c = rs_c; \
    const inc_t icstep_a = rs_a; \
\
    const inc_t pcstep_a_use = ( ( m + MR - 1 ) / MR ) * MR; \
\
    const inc_t jrstep_c = cs_c * NR; \
\
    PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
\
    ctype* restrict a_00       = a; \
    ctype* restrict b_00       = b; \
    ctype* restrict c_00       = c; \
    ctype* restrict beta_cast  = beta; \
\
    /* Make local copies of beta and one scalars to prevent any unnecessary
       sharing of cache lines between the cores' caches. */ \
    ctype           beta_local = *beta_cast; \
    ctype           one_local  = *PASTEMAC(ch,1); \
\
    auxinfo_t       aux; \
    mem_t mem_a = BLIS_MEM_INITIALIZER; \
    mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
    /* Define an array of bszid_t ids, which will act as our substitute for
       the cntl_t tree. */ \
    /*                           5thloop  4thloop         packb  3rdloop         packa  2ndloop  1stloop  ukrloop */ \
    bszid_t bszids_nopack[6] = { BLIS_NC, BLIS_KC,               BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
    bszid_t bszids_packa [7] = { BLIS_NC, BLIS_KC,               BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
    bszid_t bszids_packb [7] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC,               BLIS_NR, BLIS_MR, BLIS_KR }; \
    bszid_t bszids_packab[8] = { BLIS_NC, BLIS_KC, BLIS_NO_PART, BLIS_MC, BLIS_NO_PART, BLIS_NR, BLIS_MR, BLIS_KR }; \
    bszid_t* restrict bszids; \
\
    /* Set the bszids pointer to the correct bszids array above based on which
       matrices (if any) are being packed. */ \
\
    if ( packa ) { if ( packb ) bszids = bszids_packab; \
                   else         bszids = bszids_packa; } \
    else         { if ( packb ) bszids = bszids_packb; \
                   else         bszids = bszids_nopack; } \
\
    /* Determine whether we are using more than one thread. */ \
    const bool is_mt = ( bli_rntm_calc_num_threads( rntm ) > 1 ); \
\
    thrinfo_t* restrict thread_jc = NULL; \
    thrinfo_t* restrict thread_pc = NULL; \
    thrinfo_t* restrict thread_pb = NULL; \
    thrinfo_t* restrict thread_ic = NULL; \
    thrinfo_t* restrict thread_pa = NULL; \
    thrinfo_t* restrict thread_jr = NULL; \
\
    /* Grow the thrinfo_t tree. */ \
    bszid_t*   restrict bszids_jc = bszids; \
                        thread_jc = thread; \
    bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
    /* Compute the JC loop thread range for the current thread. */ \
    dim_t jc_start, jc_end; \
    bli_thread_range_sub( thread_jc, n, NR, FALSE, &jc_start, &jc_end ); \
\
    /* Loop over the n dimension (NC rows/columns at a time). */ \
    /*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
    for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
    { \
        /* Calculate the thread's current JC block dimension. */ \
        dim_t nc_cur = ( NC <= ( jc_end - jj ) ? NC : ( jc_end - jj ) ); \
\
        /* For MT correctness- to ensure full packing order of packed buffer */ \
        /* for Single and Multi Threaded executions are same. */ \
        dim_t jc_cur_loop = jj;\
        dim_t jc_cur_loop_rem = 0;\
        dim_t n_sub_updated = 0;\
\
        if ( packedb ) \
        { \
            get_B_panel_reordered_start_offset_width \
            ( \
              jj, n, NC, NR, \
              &jc_cur_loop, &jc_cur_loop_rem, \
              &nc_cur, &n_sub_updated \
            ); \
        } \
\
        ctype* restrict b_jc = b_00 + jj * jcstep_b; \
        ctype* restrict b_jc_use = b_00 + jc_cur_loop * jcstep_b_use; \
        ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
        /* Grow the thrinfo_t tree. */ \
        bszid_t*   restrict bszids_pc = &bszids_jc[1]; \
                            thread_pc = bli_thrinfo_sub_node( thread_jc ); \
        bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
        /* Compute the PC loop thread range for the current thread. */ \
        const dim_t pc_start = 0, pc_end = k; \
        const dim_t k_local = k; \
\
        /* Compute number of primary and leftover components of the PC loop. */ \
        /*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
        const dim_t pc_left =   k_local % KC; \
\
        /* Loop over the k dimension (KC rows/columns at a time). */ \
        /*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
        for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
        { \
            /* Calculate the thread's current PC block dimension. */ \
            const dim_t kc_cur = ( KC <= ( pc_end - pp ) ? KC : pc_left ); \
            const inc_t icstep_a_use = kc_cur; \
\
            ctype* restrict a_pc = a_00 + pp * pcstep_a; \
            ctype* restrict b_pc = b_jc + pp * pcstep_b; \
            ctype* restrict b_pc_use; \
            ctype* restrict a_pc_use = a_00 + pp * pcstep_a_use; \
\
            /* Only apply beta to the first iteration of the pc loop. */ \
            ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
            ctype* b_use; \
            inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
            /* Set the bszid_t array and thrinfo_t pointer based on whether
               we will be packing B. If we won't be packing B, we alias to
               the _pc variables so that code further down can unconditionally
               reference the _pb variables. Note that *if* we will be packing
               B, the thrinfo_t node will have already been created by a
               previous call to bli_thrinfo_grow(), since bszid values of
               BLIS_NO_PART cause the tree to grow by two (e.g. to the next
               bszid that is a normal bszid_t value). */ \
            bszid_t*   restrict bszids_pb; \
            if ( packb ) { bszids_pb = &bszids_pc[1]; \
                           thread_pb = bli_thrinfo_sub_node( thread_pc ); } \
            else         { bszids_pb = &bszids_pc[0]; \
                           thread_pb = thread_pc; } \
\
            /* Determine the packing buffer and related parameters for matrix
               B. (If B will not be packed, then a_use will be set to point to
               b and the _b_use strides will be set accordingly.) Then call
               the packm sup variant chooser, which will call the appropriate
               implementation based on the schema deduced from the stor_id. */ \
\
            /* packedb == TRUE indicates that B is reordered thus, update the
               necessary pointers.
               Else, call packm routine to pack B on-the-go. */ \
            if ( packedb ) \
            { \
                rs_b_use = NR; \
                cs_b_use = 1; \
                ps_b_use = kc_cur * NR; \
                b_pc_use = b_jc_use + pp * n_sub_updated + jc_cur_loop_rem * kc_cur; \
            } else \
            { \
                PASTEMAC(ch,packm_sup_b) \
                ( \
                packb, \
                BLIS_BUFFER_FOR_B_PANEL,  \
                stor_id,                  \
                BLIS_NO_TRANSPOSE, \
                KC,     NC,       \
                kc_cur, nc_cur, NR, \
                &one_local, \
                b_pc,   rs_b,      cs_b, \
                &b_use, &rs_b_use, &cs_b_use, \
                                   &ps_b_use, \
                cntx, \
                rntm, \
                &mem_b, \
                thread_pb  \
                ); \
\
                b_pc_use = b_use; \
            } \
\
            /* We don't need to embed the panel stride of B within the auxinfo_t
               object because this variant iterates through B in the jr loop,
               which occurs here, within the macrokernel, not within the
               millikernel. */ \
            bli_auxinfo_set_ps_b( ps_b_use, &aux ); \
\
            /* Grow the thrinfo_t tree. */ \
            bszid_t*   restrict bszids_ic = &bszids_pb[1]; \
                                thread_ic = bli_thrinfo_sub_node( thread_pb ); \
            bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
            /* Compute the IC loop thread range for the current thread. */ \
            dim_t ic_start, ic_end; \
            bli_thread_range_sub( thread_ic, m, MR, FALSE, &ic_start, &ic_end ); \
            const dim_t m_local = ic_end - ic_start; \
\
            /* Compute number of primary and leftover components of the IC loop. */ \
            /*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/ \
            const dim_t ic_left =   m_local % MC; \
\
            /* Loop over the m dimension (MC rows at a time). */ \
            /*for ( dim_t ii = 0; ii < ic_iter; ii += 1 )*/ \
            for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
            { \
                /* Calculate the thread's current IC block dimension. */ \
                const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
\
                ctype* restrict a_ic = a_pc + ii * icstep_a; \
                ctype* restrict c_ic = c_jc + ii * icstep_c; \
                ctype* restrict a_ic_use; \
\
                ctype* a_use; \
                inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
                /* Set the bszid_t array and thrinfo_t pointer based on whether
                   we will be packing B. If we won't be packing A, we alias to
                   the _ic variables so that code further down can unconditionally
                   reference the _pa variables. Note that *if* we will be packing
                   A, the thrinfo_t node will have already been created by a
                   previous call to bli_thrinfo_grow(), since bszid values of
                   BLIS_NO_PART cause the tree to grow by two (e.g. to the next
                   bszid that is a normal bszid_t value). */ \
                bszid_t*   restrict bszids_pa; \
                if ( packa ) { bszids_pa = &bszids_ic[1]; \
                               thread_pa = bli_thrinfo_sub_node( thread_ic ); } \
                else         { bszids_pa = &bszids_ic[0]; \
                               thread_pa = thread_ic; } \
\
                /* Determine the packing buffer and related parameters for matrix
                   A. (If A will not be packed, then a_use will be set to point to
                   a and the _a_use strides will be set accordingly.) Then call
                   the packm sup variant chooser, which will call the appropriate
                   implementation based on the schema deduced from the stor_id. */ \
                /* packedb == TRUE indicates that B is reordered thus, update the
                   necessary pointers.
                   Else, call packm routine to pack B on-the-go. */ \
                if ( packeda ) \
                { \
                    rs_a_use = 1; \
                    cs_a_use = MR; \
                    ps_a_use = MR * kc_cur; \
                    a_ic_use = a_pc_use + ii * icstep_a_use; \
                } \
                else \
                { \
                    PASTEMAC(ch,packm_sup_a) \
                    ( \
                    packa, \
                    BLIS_BUFFER_FOR_A_BLOCK, /* This algorithm packs matrix A to */ \
                    stor_id,                 /* a "block of A."                  */ \
                    BLIS_NO_TRANSPOSE, \
                    MC,     KC,       /* This "block of A" is (at most) MC x KC. */ \
                    mc_cur, kc_cur, MR, \
                    &one_local, \
                    a_ic,   rs_a,      cs_a, \
                    &a_use, &rs_a_use, &cs_a_use, \
                                       &ps_a_use, \
                    cntx, \
                    rntm, \
                    &mem_a, \
                    thread_pa  \
                    ); \
                    /* Alias a_use so that it's clear this is our current block of
                     matrix A. */ \
                    a_ic_use = a_use; \
                } \
\
                /* Embed the panel stride of A within the auxinfo_t object. The
                   millikernel will query and use this to iterate through
                   micropanels of A (if needed). */ \
                bli_auxinfo_set_ps_a( ps_a_use, &aux ); \
\
                /* Grow the thrinfo_t tree. */ \
                bszid_t*   restrict bszids_jr = &bszids_pa[1]; \
                                    thread_jr = bli_thrinfo_sub_node( thread_pa ); \
                bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
                /* Compute number of primary and leftover components of the JR loop. */ \
                dim_t jr_iter = ( nc_cur + NR - 1 ) / NR; \
                dim_t jr_left =   nc_cur % NR; \
\
                /* An optimization: allow the last jr iteration to contain up to NRE
                   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
                   these cases.) Note that this prevents us from declaring jr_iter and
                   jr_left as const. NOTE: We forgo this optimization when packing B
                   since packing an extended edge case is not yet supported. */ \
                if ( !packb && !is_mt ) \
                if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE ) \
                { \
                    jr_iter--; jr_left += NR; \
                } \
\
                /* Compute the JR loop thread range for the current thread. */ \
                dim_t jr_start, jr_end; \
                bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
                /* Loop over the n dimension (NR columns at a time). */ \
                /*for ( dim_t j = 0; j < jr_iter; j += 1 )*/ \
                for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
                { \
                    const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
                    ctype* restrict b_jr = b_pc_use + j * ps_b_use; \
                    ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
                    /* Loop over the m dimension (MR rows at a time). */ \
                    { \
                        /* Invoke the gemmsup millikernel. */ \
                        gemmsup_ker \
                        ( \
                          BLIS_NO_CONJUGATE, \
                          BLIS_NO_CONJUGATE, \
                          mc_cur, \
                          nr_cur, \
                          kc_cur, \
                          &one_local, \
                          a_ic_use, rs_a_use, cs_a_use, \
                          b_jr,     rs_b_use, cs_b_use, \
                          beta_use, \
                          c_jr,     rs_c,     cs_c, \
                          &aux, \
                          cntx  \
                        ); \
                    } \
                } \
            } \
\
            /* NOTE: This barrier is only needed if we are packing B (since
               that matrix is packed within the pc loop of this variant). */ \
            if ( packb ) bli_thread_barrier( thread_pb ); \
        } \
        if ( packedb ) \
        { \
            adjust_B_panel_reordered_jc( &jj, jc_cur_loop ); \
        } \
    } \
\
    /* Release any memory that was acquired for packing matrices A and B. */ \
    PASTEMAC(ch,packm_sup_finalize_mem_a) \
    ( \
      packa, \
      rntm, \
      &mem_a, \
      thread_pa  \
    ); \
    PASTEMAC(ch,packm_sup_finalize_mem_b) \
    ( \
      packb, \
      rntm, \
      &mem_b, \
      thread_pb  \
    ); \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5); \
\
}

INSERT_GENTFUNC_BASIC0_SD( gemm_compute )
