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

/**
 * This is a helper class that we use to load the symbols
 * from the reference library dynamically so that we get
 * the reference solution.
 * Since dynamic loading can be time consuming this class works
 * in the following manner.
 * - We have a thread local instance of this object. That means
 *   that for each executable there is a global variable called
 *   refCBLASModule.
 * - The constructor of refCBLASModule (which is called automatically)
 *   loads the library either with a call to dlopen (Linux) or with
 *   a call to LoadLibrary (Windows).
 * - Similarly the destructor unloads the library.
 * - The member function loadSymbol() is used to return the pointer
 *   to that symbol in the library, either with a call to ldsym (Linux)
 *   or with a call to GetProcAddress (Windows).
 * This means that the library is only loaded once per executable
 * due to having the global variable refCBLASModule and unloaded once
 * at the end. Multiple calls to loadSymbol are used to access the
 * corresponding API used for reference.
*/

namespace testinghelpers {
refCBLAS::refCBLAS() {

    std::cout << "refCBLAS constructor\n";
    if (!refCBLASModule)
    {
#ifdef WIN32
    refCBLASModule = LoadLibraryEx(REFERENCE_BLAS, NULL, LOAD_LIBRARY_SAFE_CURRENT_DIRS);
#else
  #ifdef ENABLE_ASAN
    refCBLASModule = dlopen(REFERENCE_BLAS, RTLD_LOCAL | RTLD_LAZY);
  #else
    refCBLASModule = dlopen(REFERENCE_BLAS, RTLD_DEEPBIND | RTLD_LAZY);
  #endif
#endif
    }

    if (refCBLASModule == nullptr)
    {
#ifndef WIN32
      std::cout<<dlerror();
#endif
      throw std::runtime_error("Reference Library cannot be found. LIB_PATH=" REFERENCE_BLAS );
    }
}

refCBLAS::~refCBLAS() {
    std::cout << "refCBLAS destructor\n" <<std::endl;
#ifdef WIN32
    FreeLibrary((HMODULE)refCBLASModule);
#else
    dlclose(refCBLASModule);
#endif
}

void* refCBLAS::loadSymbol(const char* symbol)
{
#ifdef WIN32
    return reinterpret_cast<void*>(GetProcAddress((HMODULE)refCBLASModule, symbol));
#else
    return dlsym(refCBLASModule, symbol);
#endif
}
} //end of testinghelpers namespace

thread_local testinghelpers::refCBLAS refCBLASModule;
