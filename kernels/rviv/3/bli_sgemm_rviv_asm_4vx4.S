#define REALNAME bli_sgemm_rviv_asm_4vx4
#define DATASIZE 4
#define DATASHIFT 2
#define VTYPE e32
#define FLOAD flw
#define FZERO(fr) fcvt.s.w fr, x0
#define FEQ feq.s
#define VLE vle32.v
#define VSE vse32.v

#include "bli_sdgemm_rviv_asm_4vx4.inc"
