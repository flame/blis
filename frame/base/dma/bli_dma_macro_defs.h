/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, Kalray Inc.

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

#ifndef BLIS_DMA_MACRO_DEFS_H
#define BLIS_DMA_MACRO_DEFS_H


// -- Default DMA-backend functions --------------------------------------------

#ifndef BLIS_DMA_BACKEND_INIT
#define BLIS_DMA_BACKEND_INIT                 bli_dma_backend_init_ref
#endif

#ifndef BLIS_DMA_BACKEND_FINALIZE
#define BLIS_DMA_BACKEND_FINALIZE             bli_dma_backend_finalize_ref
#endif

#ifndef BLIS_DMA_BACKEND_GET2D
#define BLIS_DMA_BACKEND_GET2D                bli_dma_backend_get2D_ref
#endif

#ifndef BLIS_DMA_BACKEND_PUT2D
#define BLIS_DMA_BACKEND_PUT2D                bli_dma_backend_put2D_ref
#endif

#ifndef BLIS_DMA_BACKEND_WAIT
#define BLIS_DMA_BACKEND_WAIT                 bli_dma_backend_wait_ref
#endif

BLIS_INLINE int bli_dma_backend_init()
{
    return BLIS_DMA_BACKEND_INIT();
}

BLIS_INLINE int bli_dma_backend_finalize()
{
    return BLIS_DMA_BACKEND_FINALIZE();
}

BLIS_INLINE int bli_dma_backend_get2D(
    const void*  global,
    void*        local,
    size_t       elem_size,
    int32_t      width,
    int32_t      height,
    point2d_t*   global_point,
    point2d_t*   local_point,
    dma_event_t* event
)
{
    return BLIS_DMA_BACKEND_GET2D(global, local, elem_size, width, height,
                                  global_point, local_point, event);
}

BLIS_INLINE int bli_dma_backend_put2D(
    void*        global,
    const void*  local,
    size_t       elem_size,
    int32_t      width,
    int32_t      height,
    point2d_t*   global_point,
    point2d_t*   local_point,
    dma_event_t* event
)
{
    return BLIS_DMA_BACKEND_PUT2D(global, local, elem_size, width, height,
                                  global_point, local_point, event);
}

BLIS_INLINE int bli_dma_backend_wait( dma_event_t *event )
{
    return BLIS_DMA_BACKEND_WAIT( event );
}


#endif // BLIS_DMA_MACRO_DEFS_H
