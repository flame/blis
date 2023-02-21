#include "blis.h"
#include <dlfcn.h>
#include "util/ref_nrm2.h"

/*
 * ==========================================================================
 * NORMFV performs vector operations
 *    Compute the euclidean norm of a vector
 *    of the elements in a vector x of length n. The resulting norm is stored to norm
 * ========================================================================
**/

namespace testinghelpers {

template <typename T, typename Treal>
Treal ref_nrm2(gtint_t n, T* x, gtint_t incx) {

  typedef Treal (*Fptr_ref_cblas_nrm2)( f77_int, const T *, f77_int );
  Fptr_ref_cblas_nrm2 ref_cblas_nrm2;

  // Call C function
  /* Check the typename T passed to this function template and call respective function.*/
  if (typeid(T) == typeid(float))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)dlsym(refCBLASModule.get( ), "cblas_snrm2");
  }
  else if (typeid(T) == typeid(double))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)dlsym(refCBLASModule.get(), "cblas_dnrm2");
  }
  else if (typeid(T) == typeid(scomplex))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)dlsym(refCBLASModule.get(), "cblas_scnrm2");
  }
  else if (typeid(T) == typeid(dcomplex))
  {
      ref_cblas_nrm2 = (Fptr_ref_cblas_nrm2)dlsym(refCBLASModule.get(), "cblas_dznrm2");
  }
  else
  {
    throw std::runtime_error("Error in ref_nrm2.cpp: Invalid typename is passed function template.");
  }
  if (!ref_cblas_nrm2) {
    throw std::runtime_error("Error in ref_nrm2.cpp: Function pointer == 0 -- symbol not found.");
  }

  return ref_cblas_nrm2(n, x, incx);
}

// Explicit template instantiations
template float  ref_nrm2<float, float>(gtint_t n, float* x, gtint_t incx);
template double ref_nrm2<double, double>(gtint_t n, double* x, gtint_t incx);
template float  ref_nrm2<scomplex, float>(gtint_t n, scomplex* x, gtint_t incx);
template double ref_nrm2<dcomplex, double>(gtint_t n, dcomplex* x, gtint_t incx);

} //end of namespace testinghelpers