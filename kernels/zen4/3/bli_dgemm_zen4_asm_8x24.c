#include "blis.h"
#include "bli_x86_asm_macros.h"

void bli_dgemm_zen4_asm_8x24(
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
       const auxinfo_t* data,
       const cntx_t*    cntx
                            )
                            {
                                ;
                            }