#include <iostream>
#include "common/refCBLAS.h"

namespace testinghelpers {
refCBLAS::refCBLAS() {
    std::cout << "refCBLAS constructor\n";
    if (!refCBLASModule)
    {
#ifdef REF_IS_MKL
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