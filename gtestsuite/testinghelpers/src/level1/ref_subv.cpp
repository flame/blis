#include "blis.h"
#include "level1/ref_subv.h"

namespace testinghelpers {

// Since subv is not supported by BLAS/CBLAS, we have a local reference implementation.
template<typename T>
void ref_subv( char conj_x, gtint_t n, const T* xp, gtint_t incx,
                                             T* y, gtint_t incy ) {
    gtint_t i, ix, iy;
    bool cfx    = chkconj( conj_x );
    gtint_t svx = buff_dim(n, incx);

    if (n == 0) {
        return;
    }

    std::vector<T> X( svx );
    memcpy(X.data(), xp, svx*sizeof(T));

    if( cfx ) {
        conj<T>( X.data(), n, incx );
    }

    ix = 0;
    iy = 0;
    for(i = 0 ; i < n ; i++) {
        y[iy] = y[iy] - X[ix];
        ix    = ix + incx;
        iy    = iy + incy;
    }

    return;
}

// Explicit template instantiations
template void ref_subv<float>(char, gtint_t, const float*, gtint_t, float*, gtint_t);
template void ref_subv<double>(char, gtint_t, const double*, gtint_t, double*, gtint_t);
template void ref_subv<scomplex>(char, gtint_t, const scomplex*, gtint_t, scomplex*, gtint_t);
template void ref_subv<dcomplex>(char, gtint_t, const dcomplex*, gtint_t, dcomplex*, gtint_t);

} //end of namespace testinghelpers