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

#include <iostream>
#ifdef REF_IS_MKL
#include <omp.h>
#endif
#include "common/refCBLAS.h"

namespace testinghelpers {
refCBLAS::refCBLAS() {
    std::cout << "refCBLAS constructor\n";
    if (!refCBLASModule)
    {
#ifdef REF_IS_MKL
        // Dummy call to force linker, link OpenMP library if MKL is used.
        omp_get_num_threads();
        MKLCoreModule = dlopen(MKL_CORE, RTLD_GLOBAL | RTLD_LAZY);
        MKLGNUThreadModule = dlopen(MKL_GNU_THREAD, RTLD_GLOBAL | RTLD_LAZY);
#endif
#ifdef ENABLE_ASAN
        refCBLASModule = dlopen(REFERENCE_BLAS, RTLD_LOCAL | RTLD_LAZY);
#else
        refCBLASModule = dlopen(REFERENCE_BLAS, RTLD_DEEPBIND | RTLD_LAZY);
#endif
    }

    if (refCBLASModule == nullptr)
    {
      std::cout<<dlerror();
      throw std::runtime_error("Reference Library cannot be found. LIB_PATH=" REFERENCE_BLAS );
    }
}

refCBLAS::~refCBLAS() {
    std::cout << "refCBLAS destructor\n" <<std::endl;
#ifdef REF_IS_MKL
    dlclose(MKLCoreModule);
    dlclose(MKLGNUThreadModule);
#endif
    dlclose(refCBLASModule);
}
void* refCBLAS::get() { return refCBLASModule; }
} //end of testinghelpers namespace

thread_local testinghelpers::refCBLAS refCBLASModule;
