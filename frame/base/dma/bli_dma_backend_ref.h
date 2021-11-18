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

#ifndef BLIS_DMA_BACKEND_REF_H
#define BLIS_DMA_BACKEND_REF_H

// -- Define a reference `dma_event_t` to work with bli_pthread ----------------

// Arguments for a DMA transfer. References:
// - https://github.com/kalray/opencl_optim_examples
// - https://hal.univ-grenoble-alpes.fr/hal-01652614/document
typedef struct bli_dma_thread_arg_s
{
   void*     global;         // begin address of global buffer
   void*     local;          // begin address of local buffer
   size_t    elem_size;      // size of an element in byte
   int32_t   width;          // block width in element
   int32_t   height;         // block height in element
   point2d_t global_point;   // global_point
   point2d_t local_point;    // local_point
} bli_dma_thread_arg_t;

typedef struct dma_event_s
{
   bli_dma_thread_arg_t arg;
   bli_pthread_t        thread;
} dma_event_ref_t;


// -- Reference DMA backend API ------------------------------------------------

/**
 * DMA-backend initialization
 * @return 0 on success, non-zero otherwise
 */
int bli_dma_backend_init_ref();

/**
 * DMA-backend finalize
 * @return 0 on success, non-zero otherwise
 */
int bli_dma_backend_finalize_ref();

/**
 * DMA-backend copy 2D2D from global memory to scratchpad
 * @param  global       begin address of global buffer
 * @param  local        begin address of local buffer
 * @param  elem_size    size of an element in byte
 * @param  width        block width in element
 * @param  height       block height in element
 * @param  global_point global_point
 * @param  local_point  local_point
 * @param  event        event. If non-NULL, the call configures an asynchronous
 *                      transfer and returns immediately. One must later call
                        wait() on this event.
                        If NULL, the call is blocking until the transfer is done.
 * @return 0 on success, non-zero otherwise
 */
int bli_dma_backend_get2D_ref(
	const void*      global,
	void*            local,
	size_t           elem_size,
	int32_t          width,
	int32_t          height,
	point2d_t*       global_point,
	point2d_t*       local_point,
	dma_event_ref_t* event
);

/**
 * DMA-backend copy 2D2D from scratchpad to global memory
 * @param  global       begin address of global buffer
 * @param  local        begin address of local buffer
 * @param  elem_size    size of an element in byte
 * @param  width        block width in element
 * @param  height       block height in element
 * @param  global_point global_point
 * @param  local_point  local_point
 * @param  event        event. If non-NULL, the call configures an asynchronous
 *                      transfer and returns immediately. One must later call
                        wait() on this event.
                        If NULL, the call is blocking until the transfer is done.
 * @return 0 on success, non-zero otherwise
 */
int bli_dma_backend_put2D_ref(
	void*            global,
	const void*      local,
	size_t           elem_size,
	int32_t          width,
	int32_t          height,
	point2d_t*       global_point,
	point2d_t*       local_point,
	dma_event_ref_t* event
);

/**
 * DMA-backend wait for asynchronous transfer
 * @param  event event
 * @return 0 on success, non-zero otherwise
 */
int bli_dma_backend_wait_ref( dma_event_ref_t *event );


#endif // BLIS_DMA_BACKEND_REF_H
