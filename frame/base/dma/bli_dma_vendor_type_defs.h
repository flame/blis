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

#ifndef BLIS_DMA_VENDOR_TYPE_DEFS_H
#define BLIS_DMA_VENDOR_TYPE_DEFS_H


// -- Vendor-specific DMA headers ----------------------------------------------
//
// This is the place where vendors define the `dma_event_t` type, based on
// their own DMA library, as well as overriding the default DMA backend API:
//
// #define BLIS_DMA_BACKEND_INIT          your_favorite_dma_func_init
// #define BLIS_DMA_BACKEND_FINALIZE      your_favorite_dma_func_finalize
// #define BLIS_DMA_BACKEND_GET2D         your_favorite_dma_func_get2d
// #define BLIS_DMA_BACKEND_PUT2D         your_favorite_dma_func_put2d
// #define BLIS_DMA_BACKEND_WAIT          your_favorite_dma_func_wait
//
// NOTE:
// - The current DMA support calls bli_pba_acquire_m() (bli_dma_oapi.c) to
// allocate a DMA buffer in an expected-to-be local/scratchpad memory (SMEM).
// Developer should accordingly map the PBA allocator onto SMEM.
// Generally, having packed buffers in such a near-core, fast scratchpad memory
// is always worth for performance.

#if defined(BLIS_OS_YOUR_FAVORITE_ARCH)

	// Define your vendor-specific dma_event_t here
	// ...

#else  // Reference DMA
	// No vendor-specific DMA library, define a reference `dma_event_t` to
	// work with bli_pthread.
	typedef dma_event_ref_t dma_event_t;
#endif


#endif // BLIS_DMA_VENDOR_TYPE_DEFS_H
