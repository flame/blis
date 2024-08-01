/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#if defined(__linux__)
#include <signal.h>
#include <stdexcept>
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <stdlib.h>
#include "blis.h"
#include "common/protected_buffer.h"

/*
*  Returns aligned or unaligned memory of required size
*/
void* testinghelpers::ProtectedBuffer::get_mem(dim_t size, bool is_aligned)
{
    void* mem = nullptr;
#if defined(__linux__)
    mem = is_aligned ? aligned_alloc(BLIS_HEAP_STRIDE_ALIGN_SIZE, size) : malloc(size);
#else
    mem = is_aligned ? _aligned_malloc(BLIS_HEAP_STRIDE_ALIGN_SIZE, size) : malloc(size);
#endif
    if (mem == NULL)
    {
        printf("Protected Buffer: Memory not allocated.\n");
        exit(EXIT_FAILURE);
    }
    return mem;
}

/**
 * @brief Allocate memory for greenzones and redzones, and add protection to redzones
 *
 * @param size                size of buffer required
 * @param is_aligned          should allocated memory be aligned
 * @param is_mem_test         is memory allocated for memory test.
 */
testinghelpers::ProtectedBuffer::ProtectedBuffer(dim_t size, bool is_aligned, bool is_mem_test)
{
#if defined(__linux__)
    this->is_mem_test = is_mem_test;
    if (is_mem_test)
    {
        // query page size
        size_t page_size = sysconf(_SC_PAGESIZE);

        // calculate minimum number of pages needed for requested size
        // we make buffer at least twice the requested size to make sure
        // that greenzone_1 and greenzone_2 do not overlap
        size_t buffer_size = ((( size * 2 ) / page_size) + 1) * page_size;

        // allocate memory (buffer_size + 1 page to ensure 1st redzone can be started at page bounday
        // + 2 * REDZONE_SIZE pages for 1 redzone on each end of buffer)
        mem = (char*)get_mem(buffer_size + ((1 + (REDZONE_SIZE * 2)) * page_size), is_aligned);

        // set redzone_1 to mem+page_size to make sure that
        // atleast one page boundary exist between mem and redzone_1
        redzone_1 = (void*)((char*)mem + page_size);

        // find page boundary ( address which is multiple of pagesize and less than redzone_1 )
        // say page_size is Nth power of 2 therefore only (N+1)th LSB is set in page_size
        // (-page_size) implies 2's complement therefore in (-page_size) N LSBs are unset, all
        // other bits are set.
        // (redzone_1 & -page_size) will unset N LSBs of redzone_1, therefore making redzone_1 a
        // multiple of page_size.
        // this line is equivalent to (redzone_1 - (redzone_1 % page_size))
        // where page_size is power of two.
        redzone_1 = (void*)((uintptr_t)(redzone_1) & -page_size);

        // redzone_2 = redzone_1 + sizeof redzone_1 + sizeof buffer
        redzone_2 = (void*)((char*)redzone_1 + (page_size * REDZONE_SIZE)  + buffer_size);

        // make redzones read/write/execute protected
        int res = mprotect(redzone_1, page_size * REDZONE_SIZE, PROT_NONE);
        if (res == -1)
        {
            do { perror("mprotect"); exit(EXIT_FAILURE); } while (0);
        }
        res = mprotect(redzone_2, page_size * REDZONE_SIZE, PROT_NONE);
        if (res == -1)
        {
            do { perror("mprotect"); exit(EXIT_FAILURE); } while (0);
        }

        // get address to the first "size" bytes of buffer
        greenzone_1 = (void*)((char*)redzone_1 + (page_size * REDZONE_SIZE));

        // get address to the last "size" bytes of buffer
        greenzone_2 = (void*)((char*)redzone_2 - size);
    }
    else
#endif
    {
        mem = get_mem(size, is_aligned);
        greenzone_1 = mem, greenzone_2 = mem;
    }

}

/**
 * @brief Remove Protection from redzones and free allocated memory
 */
testinghelpers::ProtectedBuffer::~ProtectedBuffer()
{
#if defined(__linux__)
    if(is_mem_test)
    {
        size_t page_size = sysconf(_SC_PAGESIZE);

        int res = mprotect(redzone_1, page_size * REDZONE_SIZE, PROT_READ | PROT_WRITE );
        if (res == -1)
        {
            do { perror("mprotect"); exit(EXIT_FAILURE); } while (0);
        }
        res = mprotect(redzone_2, page_size * REDZONE_SIZE, PROT_READ | PROT_WRITE );
        if (res == -1)
        {
            do { perror("mprotect"); exit(EXIT_FAILURE); } while (0);
        }
    }
#endif
    free(mem);
}

/**
 * Function to handle segfault during memory test and convert it to a exception
 */
void testinghelpers::ProtectedBuffer::handle_mem_test_fail(int signal)
{
#if defined(__linux__)
    // unmask the segmentation fault signal
    sigset_t signal_set;
    sigemptyset(&signal_set);
    sigaddset(&signal_set, SIGSEGV);
    sigprocmask(SIG_UNBLOCK, &signal_set, NULL);

    throw std::out_of_range("err invalid");
#endif
}

void testinghelpers::ProtectedBuffer::start_signal_handler()
{
#if defined(__linux__)
    // add signal handler for segmentation fault
    signal(SIGSEGV, ProtectedBuffer::handle_mem_test_fail);
#endif
}


void testinghelpers::ProtectedBuffer::stop_signal_handler()
{
#if defined(__linux__)
    // reset to default signal handler
    signal(SIGSEGV, SIG_DFL);
#endif
}
