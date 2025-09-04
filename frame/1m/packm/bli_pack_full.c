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
#include "../../base/bli_pack_compute_utils.h"

void bli_pack_full_init
     (
       const char*   identifier,
             obj_t*  alpha_obj,
             obj_t*  src_obj,
             obj_t*  dest_obj,
             cntx_t* cntx,
             rntm_t* rntm
     )
{
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_2);

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

    const num_t dt = bli_obj_dt( src_obj );
    
    bli_pack_full_thread_decorator
    (
     bli_is_float( dt ) ? bli_spackm_full: bli_dpackm_full,
     identifier,
     alpha_obj,
     src_obj,
     dest_obj,
     cntx,
     rntm
    );

    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_2);

}

// Full pack function for A matrix

#undef GENTFUNC
#define GENTFUNC( ctype, ch, tfuncname ) \
\
void PASTEMAC(ch,tfuncname) \
     ( \
       dim_t           m, \
       dim_t           k, \
       ctype* restrict alpha, \
       ctype* restrict src, \
       inc_t           rs, \
       inc_t           cs, \
       ctype* restrict dest, \
       cntx_t*         cntx, \
       rntm_t*         rntm, \
       thrinfo_t* thread \
     ) \
{\
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5); \
\
    const num_t dt = PASTEMAC(ch,type); \
\
    /* Query the context for various blocksizes. */ \
    const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
    const dim_t MC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
    const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
    dim_t KC = KC0; \
\
    const inc_t pcstep_a = cs; \
    const inc_t icstep_a = rs; \
\
    const inc_t pcstep_a_use = ( ( m + MR - 1 ) / MR ) * MR; \
\
    thrinfo_t* restrict thread_pa = NULL; \
    thrinfo_t* restrict thread_ic = NULL; \
\
    /* Compute the PC loop thread range for the current thread. */ \
    const dim_t pc_start = 0, pc_end = k; \
    const dim_t k_local = k; \
\
    /* Compute number of primary and leftover components of the PC loop. */ \
    /*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
    const dim_t pc_left =   k_local % KC; \
\
    inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
    /* Loop over the k dimension (KC rows/columns at a time). */ \
    /*for ( dim_t pp = 0; pp < pc_iter; pp += 1 )*/ \
    for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
    { \
        /* Calculate the thread's current PC block dimension. */ \
        const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
        const inc_t icstep_a_use = kc_cur; \
\
        ctype* restrict a_pc = src + pp * pcstep_a; \
        ctype* restrict a_pc_use = dest + pp * pcstep_a_use; \
\
        /* Grow the thrinfo_t tree. */ \
        thread_ic = thread; \
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
             ctype* restrict a_ic_use = a_pc_use + ii * icstep_a_use; \
\
             /* Packing is parallelized only at IC loop */ \
             thread_pa = &BLIS_GEMM_SINGLE_THREADED; \
\
             pack_t schema  = BLIS_PACKED_ROW_PANELS; \
             dim_t m_max = ( mc_cur / MR + ( mc_cur % MR ? 1 : 0 ) ) * MR; \
             dim_t k_max = kc_cur; \
\
             rs_a_use = 1; \
             cs_a_use = MR; \
\
             inc_t pd_a_use = MR; \
             ps_a_use = MR * kc_cur; \
\
             /* For packing to column-stored row panels, use var1. */ \
             PASTEMAC(ch,packm_sup_var1) \
             ( \
               BLIS_NO_TRANSPOSE, \
               schema, \
               mc_cur, \
               kc_cur, \
               m_max, \
               k_max, \
               alpha, \
               a_ic,  rs,  cs, \
               a_ic_use, rs_a_use, cs_a_use, \
               pd_a_use,  ps_a_use, \
               cntx, \
               thread_pa  \
             ); \
\
        } \
    } \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5); \
\
} \

INSERT_GENTFUNC_BASIC0_SD( pack_full_a )



// Full pack function for B matrix

#undef GENTFUNC
#define GENTFUNC( ctype, ch, tfuncname ) \
\
void PASTEMAC(ch,tfuncname) \
    ( \
        dim_t           k, \
        dim_t           n, \
        ctype* restrict alpha, \
        ctype* restrict src, \
        inc_t           rs, \
        inc_t           cs, \
        ctype* restrict dest, \
        cntx_t*         cntx, \
        rntm_t*         rntm, \
        thrinfo_t* thread \
    ) \
{ \
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_5); \
\
    const num_t dt = PASTEMAC(ch,type); \
\
    /* Query the context for various blocksizes. */ \
    const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
    const dim_t NC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
    const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
    dim_t KC = KC0; \
\
    const inc_t jcstep_b = cs; \
    const inc_t pcstep_b = rs; \
\
    const inc_t jcstep_b_use = k; \
\
    thrinfo_t* restrict thread_jc = NULL; \
    thrinfo_t* restrict thread_pb = NULL; \
\
    thread_jc = thread; \
\
    /* Compute the JC loop thread range for the current thread. */ \
    dim_t jc_start, jc_end; \
    bli_thread_range_sub( thread_jc, n, NR, FALSE, &jc_start, &jc_end ); \
\
    inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
    /* Loop over the n dimension (NC rows/columns at a time). */ \
    /*for ( dim_t jj = 0; jj < jc_iter; jj += 1 )*/ \
    for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
    { \
        /* Calculate the thread's current JC block dimension. */ \
        dim_t nc_cur = ( NC <= ( jc_end - jj ) ? NC : ( jc_end - jj ) ); \
\
        dim_t jc_cur_loop = jj;\
        dim_t jc_cur_loop_rem = 0;\
        dim_t n_sub_updated = 0;\
\
        /* This function returns the offsets that are computed for */ \
        /* thread workload distribution in MT execution. */           \
        get_B_panel_reordered_start_offset_width \
        ( \
          jj, n, NC, NR, \
          &jc_cur_loop, &jc_cur_loop_rem, \
          &nc_cur, &n_sub_updated \
        ); \
\
        /* The offsets are calculated in such a way that it resembles */ \
        /* the reorder buffer traversal in single threaded reordering. */ \
        /* The panel boundaries (KCxNC) remain as it is accessed in */ \
        /* single thread, and as a consequence a thread with jc_start */ \
        /* inside the panel cannot consider NC range for reorder. It */ \
        /* has to work with NC' < NC, and the offset is calulated using */ \
        /* prev NC panels spanning k dim + cur NC panel spaning pc loop */ \
        /* cur iteration + (NC - NC') spanning current kc0 (<= KC). */ \
        /* */ \
        /* Eg: Consider the following reordered buffer diagram: */ \
        /*          t1              t2                     */ \
        /*          |               |                      */ \
        /*          |           |..NC..|                   */ \
        /*          |           |      |                   */ \
        /*          |.NC. |.NC. |NC'|NC"                   */ \
        /*     pc=0-+-----+-----+---+--+                   */ \
        /*        KC|     |     |   |  |                   */ \
        /*          |  1  |  3  |   5  |                   */ \
        /*    pc=KC-+-----+-----+---st-+                   */ \
        /*        KC|     |     |   |  |                   */ \
        /*          |  2  |  4  | 6 | 7|                   */ \
        /* pc=k=2KC-+-----+-----+---+--+                   */ \
        /*          |jc=0 |jc=NC|jc=2NC|                   */ \
        /* */ \
        /* The numbers 1,2..6,7 denotes the order in which reordered */ \
        /* KCxNC blocks are stored in memory, ie: block 1 followed by 2 */ \
        /* followed by 3, etc. Given two threads t1 and t2, and t2 needs */ \
        /* to acces point st in the reorder buffer to write the data: */ \
        /* The offset calulation logic will be: */ \
        /* jc_cur_loop = 2NC, jc_cur_loop_rem = NC', pc = KC, */ \
        /* n_sub_updated = NC, k = 2KC, kc0_updated = KC */ \
        /* */ \
        /* st = ( jc_cur_loop * k )    <traverse blocks 1,2,3,4> */ \
        /*    + ( n_sub_updated * pc ) <traverse block 5>        */ \
        /*    + ( NC' * kc0_updated)   <traverse block 6>        */ \
\
        ctype* restrict b_jc = src + jj * jcstep_b; \
        ctype* restrict b_jc_use = dest + jc_cur_loop * jcstep_b_use; \
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
\
            ctype* restrict b_pc = b_jc + pp * pcstep_b; \
            ctype* restrict b_pc_use = b_jc_use + pp * n_sub_updated + jc_cur_loop_rem * kc_cur; \
\
            /* Packing is parallelized only at JC loop */ \
            thread_pb = &BLIS_GEMM_SINGLE_THREADED; \
\
            pack_t schema  = BLIS_PACKED_COL_PANELS; \
            dim_t k_max = kc_cur; \
            dim_t n_max = ( nc_cur / NR + ( nc_cur % NR ? 1 : 0 ) ) * NR; \
\
            rs_b_use = NR; \
            cs_b_use = 1; \
\
            inc_t pd_b_use = NR; \
            ps_b_use = kc_cur * NR; \
\
            /* For packing to row-stored column panels, use var1. */ \
            PASTEMAC(ch,packm_sup_var1) \
            ( \
              BLIS_NO_TRANSPOSE, \
              schema, \
              kc_cur, \
              nc_cur, \
              k_max, \
              n_max, \
              alpha, \
              b_pc,  rs,  cs, \
              b_pc_use, rs_b_use, cs_b_use, \
              pd_b_use, ps_b_use, \
              cntx, \
              thread_pb  \
            ); \
\
        } \
\
        adjust_B_panel_reordered_jc( &jj, jc_cur_loop ); \
\
    } \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_5); \
\
} \

INSERT_GENTFUNC_BASIC0_SD( pack_full_b )


#undef GENTFUNC
#define GENTFUNC( ctype, ch, tfuncname ) \
\
void PASTEMAC(ch,tfuncname) \
     ( \
       const char*   identifier, \
             obj_t*  alpha_obj, \
             obj_t*  src_obj, \
             obj_t*  dest_obj, \
             cntx_t* cntx, \
             rntm_t* rntm, \
             thrinfo_t* thread \
     ) \
{ \
\
    AOCL_DTL_TRACE_ENTRY(AOCL_DTL_LEVEL_TRACE_4); \
\
    const num_t dt = bli_obj_dt( src_obj ); \
\
    inc_t rs = bli_obj_row_stride( src_obj ); \
    inc_t cs = bli_obj_col_stride( src_obj ); \
    void* restrict src    = bli_obj_buffer_at_off( src_obj ); \
    void* restrict dest    = bli_obj_buffer_at_off( dest_obj ); \
    void* restrict alpha = bli_obj_buffer_for_1x1( dt, alpha_obj ); \
    dim_t length = bli_obj_length( src_obj ); \
    dim_t width  = bli_obj_width(src_obj); \
\
    if ( bli_obj_has_trans( src_obj ) ) \
    { \
        rs = cs; \
        cs = 1; \
        dim_t temp = length; \
        length = width; \
        width = temp; \
    } \
\
    /*---------------------------------------A-----------------------------------*/\
    if (*identifier == 'a' || *identifier == 'A') \
    {\
        PASTEMAC(ch, pack_full_a) \
        ( \
          length, \
          width, \
          alpha, \
          src, \
          rs, \
          cs, \
          dest, \
          cntx, \
          rntm, \
          thread \
        ); \
    } \
\
/*---------------------------------------B-----------------------------------*/\
    if (*identifier == 'b' || *identifier == 'B') \
    {\
        PASTEMAC(ch, pack_full_b) \
        ( \
          length, \
          width, \
          alpha, \
          src, \
          rs, \
          cs, \
          dest, \
          cntx, \
          rntm, \
          thread \
        ); \
    } \
\
    AOCL_DTL_TRACE_EXIT(AOCL_DTL_LEVEL_TRACE_4); \
\
} \

INSERT_GENTFUNC_BASIC0_SD( packm_full )


