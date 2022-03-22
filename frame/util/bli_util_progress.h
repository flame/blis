/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef BLI_UTIL_PROGRESS_H
#define BLI_UTIL_PROGRESS_H

// Public interface for the end user.

typedef dim_t (*AOCL_progress_callback)(char *api,
                                      dim_t lapi,
                                      dim_t progress,
                                      dim_t current_thread,
                                      dim_t total_threads);

BLIS_EXPORT_BLIS void AOCL_BLIS_set_progress(AOCL_progress_callback func);

// Private interfaces for internal use

extern AOCL_progress_callback AOCL_progress_ptr;

extern BLIS_TLS_TYPE dim_t tls_aoclprogress_counter;
extern BLIS_TLS_TYPE dim_t tls_aoclprogress_last_update;

// Define the frequency of reporting (number of elements).
// Progress update will be sent only after these many 
// elements are processed in the current thread.
#define AOCL_PROGRESS_FREQUENCY 1e+9

#define MAX_API_NAME_LEN 20

// Macro to send update using datatype character and the api name
#define AOCL_PROGRESS_DT(dt, api, progress, tid, nt) \
        char buf[MAX_API_NAME_LEN]; \
        snprintf(buf, MAX_API_NAME_LEN, "%c%s", dt, api); \
        (*AOCL_progress_ptr) (buf, strlen(buf), progress, tid, nt); \

// Macro to send update using api name alone.
#define AOCL_PROGRESS_NAME(api, progress, tid, nt) \
        char buf[MAX_API_NAME_LEN]; \
        snprintf(buf, MAX_API_NAME_LEN, "%s", dt, api); \
        (*AOCL_progress_ptr) (buf, strlen(buf), progress, tid, nt); \

#endif // BLI_UTIL_PROGRESS_H
