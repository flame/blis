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

#include "bls_packm.hpp"

#include <algorithm>

MemBuffer::MemBuffer( MemBuffer&& other )
{
    swap( other );
}

MemBuffer::MemBuffer( dim_t size, packbuf_t pack_buf_type, rntm_t* rntm, const communicator& thread )
{
    is_master_ = thread.master();

    if ( is_master_ )
    {
        /* Acquire directly to the chief thread's mem_t that was passed in.
           It needs to be that mem_t struct, and not a local (temporary)
           mem_t, since there is no barrier until after packing is finished,
           which could allow a race condition whereby the chief thread exits
           the current function before the other threads have a chance to
           copy from it. (A barrier would fix that race condition, but then
           again, I prefer to keep barriers to a minimum.) */
        bli_membrk_acquire_m
        (
          rntm,
          size,
          pack_buf_type,
          &mem_
        );
    }

    /* Broadcast the address of the chief thread's passed-in mem_t to all
       threads. */
    mem_t* mem_p = &mem_;
    thread.broadcast_value(mem_p);

    /* Non-chief threads: Copy the contents of the chief thread's
       passed-in mem_t to the passed-in mem_t for this thread. (The
       chief thread already has the mem_t, so it does not need to
       perform any copy.) */
    if ( ! is_master_ )
    {
        mem_ = *mem_p;
    }
}

MemBuffer& MemBuffer::operator=( MemBuffer&& other )
{
    if ( this != &other ) swap( other );
    return *this;
}

MemBuffer::~MemBuffer()
{
    if ( *this && is_master_ )
    {
        bli_membrk_release
        (
          rntm_,
          &mem_
        );
    }
}

void MemBuffer::swap( MemBuffer& other )
{
    using std::swap;
    swap( mem_, other.mem_ );
    swap( rntm_, other.rntm_ );
    swap( is_master_, other.is_master_ );
}

typedef void (*packm_cxk_ft)
    (
      conj_t  conja,
      pack_t  schema,
      dim_t   panel_dim,
      dim_t   panel_dim_max,
      dim_t   panel_len,
      dim_t   panel_len_max,
      void*   kappa,
      void*   a, inc_t inca, inc_t lda,
      void*   p,             inc_t ldp,
      cntx_t* cntx
    );

static packm_cxk_ft packm_cxk[4] =
{
  (packm_cxk_ft)bli_spackm_cxk,
  (packm_cxk_ft)bli_cpackm_cxk,
  (packm_cxk_ft)bli_dpackm_cxk,
  (packm_cxk_ft)bli_zpackm_cxk
};

void bls_packm
     (
       mdim_t              dim,
       num_t               dt,
       conj_t              conj,
       dim_t               mn_alloc,
       dim_t               k_alloc,
       dim_t               mn,
       dim_t               k,
       dim_t               mrnr,
       char*               ab, inc_t  rs_ab, inc_t cs_ab,
       char*&              p,  inc_t& ps_p,
       cntx_t*             cntx,
       rntm_t*             rntm,
       MemBuffer&          mem,
       const communicator& thread
     )
{
    /* Set the pack buffer type so that we are obtaining memory blocks from
       the pool dedicated to blocks of A or B. */
    const auto pack_buf_type = ( dim == BLIS_M ? BLIS_BUFFER_FOR_A_BLOCK
                                               : BLIS_BUFFER_FOR_B_PANEL );

    /* Barrier to make sure all threads are caught up and ready to begin the
       packm stage. */
    thread.barrier();

    /* Check the mem_t entry provided by the caller. If it is unallocated,
       then we need to acquire a block from the memory broker. Here we assume
       that is a block *has* already been allocated, then it is large enough
       for the current block. */
    if ( ! mem )
        mem = MemBuffer( mn_alloc * k_alloc * bli_dt_size( dt ),
                         pack_buf_type, rntm, thread );

    /* Determine the dimensions and strides for the packed matrix. */
    /* Pack A to column-stored row-panels or B to row-stored column-panels. */
    ps_p = mrnr * k;

    /* Set the buffer address provided by the caller to point to the memory
       associated with the mem_t entry acquired from the memory pool. */
    p = (char*)bli_mem_buffer( mem );

    /* Compute the total number of iterations we'll need. */
    auto n_iter = mn / mrnr + ( mn % mrnr ? 1 : 0 );

    auto schema = ( dim == BLIS_M ? BLIS_PACKED_ROW_PANELS
                                  : BLIS_PACKED_COL_PANELS );

    thread.distribute_over_threads(n_iter,
    [&](dim_t it_start,
        dim_t it_end)
    {
        auto vs_ab_use = rs_ab * bli_dt_size( dt );
        auto ps_p_use = ps_p * bli_dt_size( dt );

        auto one = bli_obj_buffer_for_1x1( dt, &BLIS_ONE );
        auto ab_use = ab + it_start * mrnr * vs_ab_use;
        auto p_use = p + it_start * ps_p_use;

        /* Iterate over every logical micropanel in the source matrix. */
        for ( auto it = it_start; it < it_end; it += 1 )
        {
            auto panel_dim_i = std::min( mrnr, mn - it * mrnr );

            packm_cxk[dt]
            (
              conj,
              schema,
              panel_dim_i,
              mrnr,
              k,
              k,
              one,
              ab_use, rs_ab, cs_ab,
              p_use,         mrnr,
              cntx
            );

            ab_use += mrnr * vs_ab_use;
            p_use += ps_p_use;
        }
    });

    /* Barrier so that packing is done before computation. */
    thread.barrier();
}

