#define REALNAME bli_zgemm_rviv_asm_4vx4
#define DATASIZE 16
#define DATASHIFT 4
#define VTYPE e64
#define FLOAD fld
#define FZERO(fr) fcvt.d.w fr, x0
#define FEQ feq.d
#define VLE vlseg2e64.v
#define VSE vsseg2e64.v

#include "bli_czgemm_rviv_asm_4vx4.inc"
