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

#include "blis.h"

#ifdef BLIS_ENABLE_DMA

// =============================================================================
// -- Reference implementation of DMA backend using memcpy and bli_pthread
// =============================================================================

// Backend init, called by bli_dma_sys_init()
int bli_dma_backend_init_ref()
{
	return 0;
}

// Backend finalize, called by bli_dma_sys_finalize()
int bli_dma_backend_finalize_ref()
{
	return 0;
}

static void* get2D_routine( void* arg_ )
{
	bli_dma_thread_arg_t* arg          = (bli_dma_thread_arg_t*) arg_;
	void*                 global       = arg->global;
	void*                 local        = arg->local;
	size_t                elem_size    = arg->elem_size;
	int32_t               width        = arg->width;
	int32_t               height       = arg->height;
	point2d_t             global_point = arg->global_point;
	point2d_t             local_point  = arg->local_point;

	char* local_ptr        = ((char*) local)  +
	                         (((local_point.ypos * local_point.xdim) +
	                            local_point.xpos) * elem_size);
	const char* global_ptr = ((const char*) global) +
	                         (((global_point.ypos * global_point.xdim) +
	                            global_point.xpos) * elem_size);

	for( int i = 0; i < height; ++i )
	{
		memcpy(local_ptr, global_ptr, width*elem_size);
		local_ptr  += (local_point.xdim * elem_size);
		global_ptr += (global_point.xdim * elem_size);
	}

	return NULL;
}

static void* put2D_routine( void* arg_ )
{
	bli_dma_thread_arg_t* arg          = (bli_dma_thread_arg_t*) arg_;
	void*                 global       = arg->global;
	void*                 local        = arg->local;
	size_t                elem_size    = arg->elem_size;
	int32_t               width        = arg->width;
	int32_t               height       = arg->height;
	point2d_t             global_point = arg->global_point;
	point2d_t             local_point  = arg->local_point;

	const char* local_ptr  = ((const char*) local)  +
	                         (((local_point.ypos * local_point.xdim) +
	                            local_point.xpos) * elem_size);
	char* global_ptr       = ((char*) global) +
	                         (((global_point.ypos * global_point.xdim) +
	                            global_point.xpos) * elem_size);

	for( int i = 0; i < height; ++i )
	{
		memcpy(global_ptr, local_ptr, width*elem_size);
		local_ptr  += (local_point.xdim * elem_size);
		global_ptr += (global_point.xdim * elem_size);
	}

	return NULL;
}

// 2D (asynchronous) copy between scratchpad and global memory
int bli_dma_backend_get2D_ref(
	const void*      global,
	void*            local,
	size_t           elem_size,
	int32_t          width,
	int32_t          height,
	point2d_t*       global_point,
	point2d_t*       local_point,
	dma_event_ref_t* event
)
{
	int ret = 0;

	dma_event_ref_t  event_local;
	dma_event_ref_t* event_used = event ? event : &event_local;

	event_used->arg.global       = (void* )global;
	event_used->arg.local        = (void* )local;
	event_used->arg.elem_size    = elem_size;
	event_used->arg.width        = width;
	event_used->arg.height       = height;
	event_used->arg.global_point = *global_point;
	event_used->arg.local_point  = *local_point;

	if (event) {
		// copy asynchronously by another thread
		ret = bli_pthread_create( &(event->thread), NULL,
		                          &get2D_routine, &(event_used->arg) );
		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "        %s(): bli_pthread_create() returned %d event %p\n",
		         __FUNCTION__, ret, event );
		#endif // BLIS_DMA_DEBUG
	} else {
		// blocking: copy myself
		get2D_routine( &(event_used->arg) );
		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "        %s(): get2D_routine() event %p\n", __FUNCTION__, event );
		#endif // BLIS_DMA_DEBUG
	}

	return ret;
}

// 2D (asynchronous) copy between scratchpad and global memory
int bli_dma_backend_put2D_ref(
	void*            global,
	const void*      local,
	size_t           elem_size,
	int32_t          width,
	int32_t          height,
	point2d_t*       global_point,
	point2d_t*       local_point,
	dma_event_ref_t* event
)
{
	int ret = 0;

	dma_event_ref_t  event_local;
	dma_event_ref_t* event_used = event ? event : &event_local;

	event_used->arg.global       = (void* )global;
	event_used->arg.local        = (void* )local;
	event_used->arg.elem_size    = elem_size;
	event_used->arg.width        = width;
	event_used->arg.height       = height;
	event_used->arg.global_point = *global_point;
	event_used->arg.local_point  = *local_point;

	if (event) {
		// copy asynchronously by another thread
		ret = bli_pthread_create( &(event->thread), NULL,
		                          &put2D_routine, &(event_used->arg) );
		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "        %s(): bli_pthread_create() returned %d event %p\n",
			    __FUNCTION__, ret, event );
		#endif // BLIS_DMA_DEBUG
	} else {
		// blocking: copy myself
		put2D_routine( &(event_used->arg) );
		#ifdef BLIS_DMA_DEBUG
		fprintf( stdout, "        %s(): put2D_routine() event %p\n", __FUNCTION__, event );
		#endif // BLIS_DMA_DEBUG
	}

	return ret;
}

// Wait for termination of a DMA transfer
int bli_dma_backend_wait_ref( dma_event_ref_t *event )
{
	#ifdef BLIS_DMA_DEBUG
	fprintf( stdout, "        %s(): event %p\n", __FUNCTION__, event );
	#endif // BLIS_DMA_DEBUG
	return ( event ? bli_pthread_join( event->thread, NULL ) : 0 );
}



#endif // BLIS_ENABLE_DMA
