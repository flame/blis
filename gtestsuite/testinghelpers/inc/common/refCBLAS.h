#include <dlfcn.h>
#include <stdexcept>

namespace testinghelpers {
class refCBLAS
{
  private:
#ifdef REF_IS_MKL
    void *MKLCoreModule = nullptr;
    void *MKLGNUThreadModule = nullptr;
#endif
    void *refCBLASModule = nullptr;

  public:
    refCBLAS();
    ~refCBLAS();
    void* get();
};
} //end of testinghelpers namespace

extern thread_local testinghelpers::refCBLAS refCBLASModule;