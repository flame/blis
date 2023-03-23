#include "blis.h"

#ifdef __cplusplus
#include "complex_math.hpp"
extern "C"
#endif
void syrk_diag_ref( obj_t* alpha, obj_t* a, obj_t* d, obj_t* beta, obj_t* c );

