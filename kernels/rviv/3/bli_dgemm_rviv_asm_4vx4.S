#define REALNAME bli_dgemm_rviv_asm_4vx4
#define DATASIZE 8
#define DATASHIFT 3
#define VTYPE e64
#define FLOAD fld
#define FZERO(fr) fcvt.d.w fr, x0
#define FEQ feq.d
#define VLE vle64.v
#define VSE vse64.v

#include "bli_sdgemm_rviv_asm_4vx4.inc"
