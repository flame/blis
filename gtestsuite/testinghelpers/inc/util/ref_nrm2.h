#pragma once

#include "common/testing_helpers.h"

/*
 * ==========================================================================
 * NRM2 returns the euclidean norm of a vector via the function
 * name, so that
 *    NRM2 := sqrt( x'*x ).
 * ==========================================================================
 */

namespace testinghelpers {

template <typename Tf, typename T>
T ref_nrm2(gtint_t n, Tf* x, gtint_t incx);

} //end of namespace testinghelpers