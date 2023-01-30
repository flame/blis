#define REALNAME bli_cgemm_rviv_asm_4vx4
#define DATASIZE 8
#define DATASHIFT 3
#define VTYPE e32
#define FLOAD flw
#define FZERO(fr) fcvt.s.w fr, x0
#define FEQ feq.s
#define VLE vlseg2e32.v
#define VSE vsseg2e32.v

#include "bli_czgemm_rviv_asm_4vx4.inc"
