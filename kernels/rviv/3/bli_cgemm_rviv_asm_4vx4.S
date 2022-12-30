#define REALNAME bli_cgemm_rviv_asm_4vx4
    .text
    .align      2
    .global     REALNAME

// void REALNAME(uint64_t k, scomplex* alpha, scomplex* a, scomplex* b,
//               scomplex* beta, scomplex* c, uint64_t rs_c, uint64_t cs_c)
//
// register arguments:
// a0   k
// a1   alpha
// a2   a
// a3   b
// a4   beta
// a5   c
// a6   rs_c
// a7   cs_c
//
// Assume rs_c = 1

#define loop_counter a0

#define A00_ptr a2
#define A10_ptr t0
#define A01_ptr t1
#define A11_ptr t2

#define B_row_ptr  a3

#define C00_ptr  a5
#define C01_ptr  t3
#define C02_ptr  t4
#define C03_ptr  t5
#define C10_ptr  s1
#define C11_ptr  s2
#define C12_ptr  s3
#define C13_ptr  s4

#define tmp   t6

#define ALPHA_re fa0
#define ALPHA_im fa1
#define BETA_re  fa2
#define BETA_im  fa3

#define B00_re fa4
#define B00_im fa5
#define B01_re fa6
#define B01_im fa7
#define B02_re fa0
#define B02_im fa1
#define B03_re fa2
#define B03_im fa3

#define B10_re ft0
#define B10_im ft1
#define B11_re ft2
#define B11_im ft3
#define B12_re ft4
#define B12_im ft5
#define B13_re ft6
#define B13_im ft7

#define A00_re v24
#define A00_im v25
#define A10_re v26
#define A10_im v27
#define A01_re v28
#define A01_im v29
#define A11_re v30
#define A11_im v31

#define C0_re v24
#define C0_im v25
#define C1_re v26
#define C1_im v27
#define C2_re v28
#define C2_im v29
#define C3_re v30
#define C3_im v31

#define AB00_re v0
#define AB00_im v1
#define AB01_re v2
#define AB01_im v3
#define AB02_re v4
#define AB02_im v5
#define AB03_re v6
#define AB03_im v7
#define AB10_re v8
#define AB10_im v9
#define AB11_re v10
#define AB11_im v11
#define AB12_re v12
#define AB12_im v13
#define AB13_re v14
#define AB13_im v15

#define tmp0_re v16
#define tmp0_im v17
#define tmp1_re v18
#define tmp1_im v19
#define tmp2_re v20
#define tmp2_im v21
#define tmp3_re v22
#define tmp3_im v23

#define rs_c  a6
#define cs_c  a7

REALNAME:
    addi sp,sp,-64
    sd s7,56(sp)
    sd s6,48(sp)
    sd s5,40(sp)
    sd s4,32(sp)
    sd s3,24(sp)
    sd s2,16(sp)
    sd s1,8(sp)
    sd s0,0(sp)

    li tmp, 256
    vsetvli s0, tmp, e32, m1, ta, ma
    slli s0, s0, 3      // count in 2*sizeof(float) steps
    mv rs_c, s0         // Assumes unit row skip
    slli cs_c, cs_c, 3

    // Set up pointers
    add C01_ptr, C00_ptr, cs_c
    add C02_ptr, C01_ptr, cs_c
    add C03_ptr, C02_ptr, cs_c
    add C10_ptr, C00_ptr, rs_c
    add C11_ptr, C01_ptr, rs_c
    add C12_ptr, C02_ptr, rs_c
    add C13_ptr, C03_ptr, rs_c

    add A10_ptr, A00_ptr, s0
    slli s0, s0, 1      // length of a column of A in bytes
    add A01_ptr, A00_ptr, s0
    add A11_ptr, A10_ptr, s0

    // Zero-initialize accumulators
    fmv.w.x fa3, zero
    vfmv.v.f AB00_re, fa3
    vfmv.v.f AB00_im, fa3
    vfmv.v.f AB01_re, fa3
    vfmv.v.f AB01_im, fa3
    vfmv.v.f AB02_re, fa3
    vfmv.v.f AB02_im, fa3
    vfmv.v.f AB03_re, fa3
    vfmv.v.f AB03_im, fa3
    vfmv.v.f AB10_re, fa3
    vfmv.v.f AB10_im, fa3
    vfmv.v.f AB11_re, fa3
    vfmv.v.f AB11_im, fa3
    vfmv.v.f AB12_re, fa3
    vfmv.v.f AB12_im, fa3
    vfmv.v.f AB13_re, fa3
    vfmv.v.f AB13_im, fa3

    li tmp, 3
    ble loop_counter, tmp, TAIL_UNROLL_2

    // Preload A and B
    // Load and deinterleave A(:,l)
    vlseg2e32.v A00_re, (A00_ptr)
    vlseg2e32.v A10_re, (A10_ptr)

    // Load B(l,0:3)
    flw B00_re, (B_row_ptr)
    flw B00_im, 4(B_row_ptr)
    flw B01_re, 8(B_row_ptr)
    flw B01_im, 12(B_row_ptr)
    flw B02_re, 16(B_row_ptr)
    flw B02_im, 20(B_row_ptr)
    flw B03_re, 24(B_row_ptr)
    flw B03_im, 28(B_row_ptr)

    // Load and deinterleave A(:,l+1)
    vlseg2e32.v A01_re, (A01_ptr)
    vlseg2e32.v A11_re, (A11_ptr)

LOOP_UNROLL_4: // loop_counter >= 4
    addi loop_counter, loop_counter, -4

    vfmacc.vf AB00_re, B00_re, A00_re   // AB(:,0) += A(:,l) * B(l,0)
    vfnmsac.vf AB00_re, B00_im, A00_im
    vfmacc.vf AB00_im, B00_re, A00_im
    vfmacc.vf AB00_im, B00_im, A00_re
    vfmacc.vf AB10_re, B00_re, A10_re
    vfnmsac.vf AB10_re, B00_im, A10_im
    vfmacc.vf AB10_im, B00_re, A10_im
    vfmacc.vf AB10_im, B00_im, A10_re

    vfmacc.vf AB01_re, B01_re, A00_re   // AB(:,1) += A(:,l) * B(l,1)
    vfnmsac.vf AB01_re, B01_im, A00_im
    vfmacc.vf AB01_im, B01_re, A00_im
    vfmacc.vf AB01_im, B01_im, A00_re
    vfmacc.vf AB11_re, B01_re, A10_re
    vfnmsac.vf AB11_re, B01_im, A10_im
    vfmacc.vf AB11_im, B01_re, A10_im
    vfmacc.vf AB11_im, B01_im, A10_re

    // Point to A(:,l+2), A(:,l+3)
    add A00_ptr, A01_ptr, s0
    add A10_ptr, A11_ptr, s0
    add A01_ptr, A00_ptr, s0
    add A11_ptr, A10_ptr, s0

    // Load B(l+1,0:3)
    flw B10_re, 32(B_row_ptr)
    flw B10_im, 36(B_row_ptr)
    flw B11_re, 40(B_row_ptr)
    flw B11_im, 44(B_row_ptr)
    flw B12_re, 48(B_row_ptr)
    flw B12_im, 52(B_row_ptr)
    flw B13_re, 56(B_row_ptr)
    flw B13_im, 60(B_row_ptr)
    addi B_row_ptr, B_row_ptr, 64

    vfmacc.vf AB00_re, B10_re, A01_re   // AB(:,0) += A(:,l+1) * B(l+1,0)
    vfnmsac.vf AB00_re, B10_im, A01_im
    vfmacc.vf AB00_im, B10_re, A01_im
    vfmacc.vf AB00_im, B10_im, A01_re
    vfmacc.vf AB10_re, B10_re, A11_re
    vfnmsac.vf AB10_re, B10_im, A11_im
    vfmacc.vf AB10_im, B10_re, A11_im
    vfmacc.vf AB10_im, B10_im, A11_re

    vfmacc.vf AB02_re, B02_re, A00_re   // AB(:,2) += A(:,l) * B(l,2)
    vfnmsac.vf AB02_re, B02_im, A00_im
    vfmacc.vf AB02_im, B02_re, A00_im
    vfmacc.vf AB02_im, B02_im, A00_re
    vfmacc.vf AB12_re, B02_re, A10_re
    vfnmsac.vf AB12_re, B02_im, A10_im
    vfmacc.vf AB12_im, B02_re, A10_im
    vfmacc.vf AB12_im, B02_im, A10_re

    vfmacc.vf AB03_re, B03_re, A00_re   // AB(:,3) += A(:,l) * B(l,3)
    vfnmsac.vf AB03_re, B03_im, A00_im
    vfmacc.vf AB03_im, B03_re, A00_im
    vfmacc.vf AB03_im, B03_im, A00_re
    vfmacc.vf AB13_re, B03_re, A10_re
    vfnmsac.vf AB13_re, B03_im, A10_im
    vfmacc.vf AB13_im, B03_re, A10_im
    vfmacc.vf AB13_im, B03_im, A10_re

    // Load and deinterleave A(:,l+2)
    vlseg2e32.v A00_re, (A00_ptr)
    vlseg2e32.v A10_re, (A10_ptr)

    // Load B(l+2, 0:3)
    flw B00_re, (B_row_ptr)
    flw B00_im, 4(B_row_ptr)
    flw B01_re, 8(B_row_ptr)
    flw B01_im, 12(B_row_ptr)
    flw B02_re, 16(B_row_ptr)
    flw B02_im, 20(B_row_ptr)
    flw B03_re, 24(B_row_ptr)
    flw B03_im, 28(B_row_ptr)

    vfmacc.vf AB01_re, B11_re, A01_re   // AB(:,1) += A(:,l+1) * B(l+1,1)
    vfnmsac.vf AB01_re, B11_im, A01_im
    vfmacc.vf AB01_im, B11_re, A01_im
    vfmacc.vf AB01_im, B11_im, A01_re
    vfmacc.vf AB11_re, B11_re, A11_re
    vfnmsac.vf AB11_re, B11_im, A11_im
    vfmacc.vf AB11_im, B11_re, A11_im
    vfmacc.vf AB11_im, B11_im, A11_re

    vfmacc.vf AB02_re, B12_re, A01_re   // AB(:,2) += A(:,l+1) * B(l+1,2)
    vfnmsac.vf AB02_re, B12_im, A01_im
    vfmacc.vf AB02_im, B12_re, A01_im
    vfmacc.vf AB02_im, B12_im, A01_re
    vfmacc.vf AB12_re, B12_re, A11_re
    vfnmsac.vf AB12_re, B12_im, A11_im
    vfmacc.vf AB12_im, B12_re, A11_im
    vfmacc.vf AB12_im, B12_im, A11_re

    vfmacc.vf AB03_re, B13_re, A01_re   // AB(:,3) += A(:,l+1) * B(l+1,3)
    vfnmsac.vf AB03_re, B13_im, A01_im
    vfmacc.vf AB03_im, B13_re, A01_im
    vfmacc.vf AB03_im, B13_im, A01_re
    vfmacc.vf AB13_re, B13_re, A11_re
    vfnmsac.vf AB13_re, B13_im, A11_im
    vfmacc.vf AB13_im, B13_re, A11_im
    vfmacc.vf AB13_im, B13_im, A11_re

    // Load and deinterleave A(:,l+3)
    vlseg2e32.v A01_re, (A01_ptr)
    vlseg2e32.v A11_re, (A11_ptr)

    // Point to A(:,l+2), A(:,l+3)
    add A00_ptr, A01_ptr, s0
    add A10_ptr, A11_ptr, s0
    add A01_ptr, A00_ptr, s0
    add A11_ptr, A10_ptr, s0

    // Load B(l+3, 0:3)
    flw B10_re, 32(B_row_ptr)
    flw B10_im, 36(B_row_ptr)
    flw B11_re, 40(B_row_ptr)
    flw B11_im, 44(B_row_ptr)
    flw B12_re, 48(B_row_ptr)
    flw B12_im, 52(B_row_ptr)
    flw B13_re, 56(B_row_ptr)
    flw B13_im, 60(B_row_ptr)
    addi B_row_ptr, B_row_ptr, 64

    vfmacc.vf AB00_re, B00_re, A00_re   // AB(:,0) += A(:,l+2) * B(l+2,0)
    vfnmsac.vf AB00_re, B00_im, A00_im
    vfmacc.vf AB00_im, B00_re, A00_im
    vfmacc.vf AB00_im, B00_im, A00_re
    vfmacc.vf AB10_re, B00_re, A10_re
    vfnmsac.vf AB10_re, B00_im, A10_im
    vfmacc.vf AB10_im, B00_re, A10_im
    vfmacc.vf AB10_im, B00_im, A10_re

    vfmacc.vf AB00_re, B10_re, A01_re   // AB(:,0) += A(:,l+3) * B(l+3,0)
    vfnmsac.vf AB00_re, B10_im, A01_im
    vfmacc.vf AB00_im, B10_re, A01_im
    vfmacc.vf AB00_im, B10_im, A01_re
    vfmacc.vf AB10_re, B10_re, A11_re
    vfnmsac.vf AB10_re, B10_im, A11_im
    vfmacc.vf AB10_im, B10_re, A11_im
    vfmacc.vf AB10_im, B10_im, A11_re

    vfmacc.vf AB01_re, B01_re, A00_re   // AB(:,1) += A(:,l+2) * B(l+2,1)
    vfnmsac.vf AB01_re, B01_im, A00_im
    vfmacc.vf AB01_im, B01_re, A00_im
    vfmacc.vf AB01_im, B01_im, A00_re
    vfmacc.vf AB11_re, B01_re, A10_re
    vfnmsac.vf AB11_re, B01_im, A10_im
    vfmacc.vf AB11_im, B01_re, A10_im
    vfmacc.vf AB11_im, B01_im, A10_re

    vfmacc.vf AB01_re, B11_re, A01_re   // AB(:,1) += A(:,l+3) * B(l+3,1)
    vfnmsac.vf AB01_re, B11_im, A01_im
    vfmacc.vf AB01_im, B11_re, A01_im
    vfmacc.vf AB01_im, B11_im, A01_re
    vfmacc.vf AB11_re, B11_re, A11_re
    vfnmsac.vf AB11_re, B11_im, A11_im
    vfmacc.vf AB11_im, B11_re, A11_im
    vfmacc.vf AB11_im, B11_im, A11_re

    vfmacc.vf AB02_re, B02_re, A00_re   // AB(:,2) += A(:,l+2) * B(l+2,2)
    vfnmsac.vf AB02_re, B02_im, A00_im
    vfmacc.vf AB02_im, B02_re, A00_im
    vfmacc.vf AB02_im, B02_im, A00_re
    vfmacc.vf AB12_re, B02_re, A10_re
    vfnmsac.vf AB12_re, B02_im, A10_im
    vfmacc.vf AB12_im, B02_re, A10_im
    vfmacc.vf AB12_im, B02_im, A10_re

    vfmacc.vf AB02_re, B12_re, A01_re   // AB(:,2) += A(:,l+3) * B(l+3,2)
    vfnmsac.vf AB02_re, B12_im, A01_im
    vfmacc.vf AB02_im, B12_re, A01_im
    vfmacc.vf AB02_im, B12_im, A01_re
    vfmacc.vf AB12_re, B12_re, A11_re
    vfnmsac.vf AB12_re, B12_im, A11_im
    vfmacc.vf AB12_im, B12_re, A11_im
    vfmacc.vf AB12_im, B12_im, A11_re

    vfmacc.vf AB03_re, B03_re, A00_re   // AB(:,3) += A(:,l+2) * B(l+2,3)
    vfnmsac.vf AB03_re, B03_im, A00_im
    vfmacc.vf AB03_im, B03_re, A00_im
    vfmacc.vf AB03_im, B03_im, A00_re
    vfmacc.vf AB13_re, B03_re, A10_re
    vfnmsac.vf AB13_re, B03_im, A10_im
    vfmacc.vf AB13_im, B03_re, A10_im
    vfmacc.vf AB13_im, B03_im, A10_re

    vfmacc.vf AB03_re, B13_re, A01_re   // AB(:,3) += A(:,l+3) * B(l+3,3)
    vfnmsac.vf AB03_re, B13_im, A01_im
    vfmacc.vf AB03_im, B13_re, A01_im
    vfmacc.vf AB03_im, B13_im, A01_re
    vfmacc.vf AB13_re, B13_re, A11_re
    vfnmsac.vf AB13_re, B13_im, A11_im
    vfmacc.vf AB13_im, B13_re, A11_im
    vfmacc.vf AB13_im, B13_im, A11_re

    li tmp, 3
    ble loop_counter, tmp, TAIL_UNROLL_2

    // Load A and B for the next iteration
    vlseg2e32.v A00_re, (A00_ptr)
    vlseg2e32.v A10_re, (A10_ptr)
    vlseg2e32.v A01_re, (A01_ptr)
    vlseg2e32.v A11_re, (A11_ptr)

    flw B00_re, (B_row_ptr)
    flw B00_im, 4(B_row_ptr)
    flw B01_re, 8(B_row_ptr)
    flw B01_im, 12(B_row_ptr)
    flw B02_re, 16(B_row_ptr)
    flw B02_im, 20(B_row_ptr)
    flw B03_re, 24(B_row_ptr)
    flw B03_im, 28(B_row_ptr)

    j LOOP_UNROLL_4


TAIL_UNROLL_2: // loop_counter <= 3
    li tmp, 1
    ble loop_counter, tmp, TAIL_UNROLL_1

    addi loop_counter, loop_counter, -2

    // Load and deinterleave A(:,l)
    vlseg2e32.v A00_re, (A00_ptr)
    vlseg2e32.v A10_re, (A10_ptr)

    // Load B(l, 0:3)
    flw B00_re, (B_row_ptr)
    flw B00_im, 4(B_row_ptr)
    flw B01_re, 8(B_row_ptr)
    flw B01_im, 12(B_row_ptr)
    flw B02_re, 16(B_row_ptr)
    flw B02_im, 20(B_row_ptr)
    flw B03_re, 24(B_row_ptr)
    flw B03_im, 28(B_row_ptr)

    vfmacc.vf AB00_re, B00_re, A00_re   // AB(:,0) += A(:,l) * B(l,0)
    vfnmsac.vf AB00_re, B00_im, A00_im
    vfmacc.vf AB00_im, B00_re, A00_im
    vfmacc.vf AB00_im, B00_im, A00_re
    vfmacc.vf AB10_re, B00_re, A10_re
    vfnmsac.vf AB10_re, B00_im, A10_im
    vfmacc.vf AB10_im, B00_re, A10_im
    vfmacc.vf AB10_im, B00_im, A10_re

    vfmacc.vf AB01_re, B01_re, A00_re   // AB(:,1) += A(:,l) * B(l,1)
    vfnmsac.vf AB01_re, B01_im, A00_im
    vfmacc.vf AB01_im, B01_re, A00_im
    vfmacc.vf AB01_im, B01_im, A00_re
    vfmacc.vf AB11_re, B01_re, A10_re
    vfnmsac.vf AB11_re, B01_im, A10_im
    vfmacc.vf AB11_im, B01_re, A10_im
    vfmacc.vf AB11_im, B01_im, A10_re

    // Load and deinterleave A(:,l+1)
    vlseg2e32.v A01_re, (A01_ptr)
    vlseg2e32.v A11_re, (A11_ptr)

    // Load B(l+1, 0:3)
    flw B10_re, 32(B_row_ptr)
    flw B10_im, 36(B_row_ptr)
    flw B11_re, 40(B_row_ptr)
    flw B11_im, 44(B_row_ptr)
    flw B12_re, 48(B_row_ptr)
    flw B12_im, 52(B_row_ptr)
    flw B13_re, 56(B_row_ptr)
    flw B13_im, 60(B_row_ptr)

    vfmacc.vf AB00_re, B10_re, A01_re   // AB(:,0) += A(:,l+1) * B(l+1,0)
    vfnmsac.vf AB00_re, B10_im, A01_im
    vfmacc.vf AB00_im, B10_re, A01_im
    vfmacc.vf AB00_im, B10_im, A01_re
    vfmacc.vf AB10_re, B10_re, A11_re
    vfnmsac.vf AB10_re, B10_im, A11_im
    vfmacc.vf AB10_im, B10_re, A11_im
    vfmacc.vf AB10_im, B10_im, A11_re

    vfmacc.vf AB01_re, B11_re, A01_re   // AB(:,1) += A(:,l+1) * B(l+1,1)
    vfnmsac.vf AB01_re, B11_im, A01_im
    vfmacc.vf AB01_im, B11_re, A01_im
    vfmacc.vf AB01_im, B11_im, A01_re
    vfmacc.vf AB11_re, B11_re, A11_re
    vfnmsac.vf AB11_re, B11_im, A11_im
    vfmacc.vf AB11_im, B11_re, A11_im
    vfmacc.vf AB11_im, B11_im, A11_re

    vfmacc.vf AB02_re, B02_re, A00_re   // AB(:,2) += A(:,l) * B(l,2)
    vfnmsac.vf AB02_re, B02_im, A00_im
    vfmacc.vf AB02_im, B02_re, A00_im
    vfmacc.vf AB02_im, B02_im, A00_re
    vfmacc.vf AB12_re, B02_re, A10_re
    vfnmsac.vf AB12_re, B02_im, A10_im
    vfmacc.vf AB12_im, B02_re, A10_im
    vfmacc.vf AB12_im, B02_im, A10_re

    vfmacc.vf AB03_re, B03_re, A00_re   // AB(:,3) += A(:,l) * B(l,3)
    vfnmsac.vf AB03_re, B03_im, A00_im
    vfmacc.vf AB03_im, B03_re, A00_im
    vfmacc.vf AB03_im, B03_im, A00_re
    vfmacc.vf AB13_re, B03_re, A10_re
    vfnmsac.vf AB13_re, B03_im, A10_im
    vfmacc.vf AB13_im, B03_re, A10_im
    vfmacc.vf AB13_im, B03_im, A10_re

    vfmacc.vf AB02_re, B12_re, A01_re   // AB(:,2) += A(:,l+1) * B(l+1,2)
    vfnmsac.vf AB02_re, B12_im, A01_im
    vfmacc.vf AB02_im, B12_re, A01_im
    vfmacc.vf AB02_im, B12_im, A01_re
    vfmacc.vf AB12_re, B12_re, A11_re
    vfnmsac.vf AB12_re, B12_im, A11_im
    vfmacc.vf AB12_im, B12_re, A11_im
    vfmacc.vf AB12_im, B12_im, A11_re

    vfmacc.vf AB03_re, B13_re, A01_re   // AB(:,3) += A(:,l+1) * B(l+1,3)
    vfnmsac.vf AB03_re, B13_im, A01_im
    vfmacc.vf AB03_im, B13_re, A01_im
    vfmacc.vf AB03_im, B13_im, A01_re
    vfmacc.vf AB13_re, B13_re, A11_re
    vfnmsac.vf AB13_re, B13_im, A11_im
    vfmacc.vf AB13_im, B13_re, A11_im
    vfmacc.vf AB13_im, B13_im, A11_re

    beqz loop_counter, MULTIPLYALPHA

    // Advance pointers
    add A00_ptr, A01_ptr, s0
    add A10_ptr, A11_ptr, s0
    addi B_row_ptr, B_row_ptr, 64

TAIL_UNROLL_1: // loop_counter <= 1
    beqz loop_counter, MULTIPLYALPHA

    // Load and deinterleave A(:,l)
    vlseg2e32.v A00_re, (A00_ptr)
    vlseg2e32.v A10_re, (A10_ptr)

    // Load B(l,0:3)
    flw B00_re, (B_row_ptr)
    flw B00_im, 4(B_row_ptr)
    flw B01_re, 8(B_row_ptr)
    flw B01_im, 12(B_row_ptr)
    flw B02_re, 16(B_row_ptr)
    flw B02_im, 20(B_row_ptr)
    flw B03_re, 24(B_row_ptr)
    flw B03_im, 28(B_row_ptr)

    vfmacc.vf AB00_re, B00_re, A00_re   // AB(:,0) += A(:,l) * B(l,0)
    vfnmsac.vf AB00_re, B00_im, A00_im
    vfmacc.vf AB00_im, B00_re, A00_im
    vfmacc.vf AB00_im, B00_im, A00_re
    vfmacc.vf AB10_re, B00_re, A10_re
    vfnmsac.vf AB10_re, B00_im, A10_im
    vfmacc.vf AB10_im, B00_re, A10_im
    vfmacc.vf AB10_im, B00_im, A10_re

    vfmacc.vf AB01_re, B01_re, A00_re   // AB(:,1) += A(:,l) * B(l,1)
    vfnmsac.vf AB01_re, B01_im, A00_im
    vfmacc.vf AB01_im, B01_re, A00_im
    vfmacc.vf AB01_im, B01_im, A00_re
    vfmacc.vf AB11_re, B01_re, A10_re
    vfnmsac.vf AB11_re, B01_im, A10_im
    vfmacc.vf AB11_im, B01_re, A10_im
    vfmacc.vf AB11_im, B01_im, A10_re

    vfmacc.vf AB02_re, B02_re, A00_re   // AB(:,2) += A(:,l) * B(l,2)
    vfnmsac.vf AB02_re, B02_im, A00_im
    vfmacc.vf AB02_im, B02_re, A00_im
    vfmacc.vf AB02_im, B02_im, A00_re
    vfmacc.vf AB12_re, B02_re, A10_re
    vfnmsac.vf AB12_re, B02_im, A10_im
    vfmacc.vf AB12_im, B02_re, A10_im
    vfmacc.vf AB12_im, B02_im, A10_re

    vfmacc.vf AB03_re, B03_re, A00_re   // AB(:,3) += A(:,l) * B(l,3)
    vfnmsac.vf AB03_re, B03_im, A00_im
    vfmacc.vf AB03_im, B03_re, A00_im
    vfmacc.vf AB03_im, B03_im, A00_re
    vfmacc.vf AB13_re, B03_re, A10_re
    vfnmsac.vf AB13_re, B03_im, A10_im
    vfmacc.vf AB13_im, B03_re, A10_im
    vfmacc.vf AB13_im, B03_im, A10_re

MULTIPLYALPHA:
    flw ALPHA_re, (a1)
    flw ALPHA_im, 4(a1)
    flw BETA_re, (a4)
    flw BETA_im, 4(a4)

    fmv.w.x fa4, zero
    feq.s tmp, ALPHA_im, fa4
    bne tmp, zero, ALPHAREAL

    // [AB00, ..., AB03] * alpha
    vfmul.vf tmp0_re, AB00_im, ALPHA_im
    vfmul.vf tmp0_im, AB00_re, ALPHA_im
    vfmul.vf tmp1_re, AB01_im, ALPHA_im
    vfmul.vf tmp1_im, AB01_re, ALPHA_im
    vfmul.vf tmp2_re, AB02_im, ALPHA_im
    vfmul.vf tmp2_im, AB02_re, ALPHA_im
    vfmul.vf tmp3_re, AB03_im, ALPHA_im
    vfmul.vf tmp3_im, AB03_re, ALPHA_im
    vfmsub.vf AB00_re, ALPHA_re, tmp0_re
    vfmsub.vf AB01_re, ALPHA_re, tmp1_re
    vfmsub.vf AB02_re, ALPHA_re, tmp2_re
    vfmsub.vf AB03_re, ALPHA_re, tmp3_re
    vfmadd.vf AB00_im, ALPHA_re, tmp0_im
    vfmadd.vf AB01_im, ALPHA_re, tmp1_im
    vfmadd.vf AB02_im, ALPHA_re, tmp2_im
    vfmadd.vf AB03_im, ALPHA_re, tmp3_im

    // [AB10, ..., AB13] * alpha
    vfmul.vf tmp0_re, AB10_im, ALPHA_im
    vfmul.vf tmp0_im, AB10_re, ALPHA_im
    vfmul.vf tmp1_re, AB11_im, ALPHA_im
    vfmul.vf tmp1_im, AB11_re, ALPHA_im
    vfmul.vf tmp2_re, AB12_im, ALPHA_im
    vfmul.vf tmp2_im, AB12_re, ALPHA_im
    vfmul.vf tmp3_re, AB13_im, ALPHA_im
    vfmul.vf tmp3_im, AB13_re, ALPHA_im
    vfmsub.vf AB10_re, ALPHA_re, tmp0_re
    vfmsub.vf AB11_re, ALPHA_re, tmp1_re
    vfmsub.vf AB12_re, ALPHA_re, tmp2_re
    vfmsub.vf AB13_re, ALPHA_re, tmp3_re
    vfmadd.vf AB10_im, ALPHA_re, tmp0_im
    vfmadd.vf AB11_im, ALPHA_re, tmp1_im
    vfmadd.vf AB12_im, ALPHA_re, tmp2_im
    vfmadd.vf AB13_im, ALPHA_re, tmp3_im

    j MULTIPLYBETA

ALPHAREAL:
    vfmul.vf AB00_re, AB00_re, ALPHA_re
    vfmul.vf AB00_im, AB00_im, ALPHA_re
    vfmul.vf AB01_re, AB01_re, ALPHA_re
    vfmul.vf AB01_im, AB01_im, ALPHA_re
    vfmul.vf AB02_re, AB02_re, ALPHA_re
    vfmul.vf AB02_im, AB02_im, ALPHA_re
    vfmul.vf AB03_re, AB03_re, ALPHA_re
    vfmul.vf AB03_im, AB03_im, ALPHA_re

    vfmul.vf AB10_re, AB10_re, ALPHA_re
    vfmul.vf AB10_im, AB10_im, ALPHA_re
    vfmul.vf AB11_re, AB11_re, ALPHA_re
    vfmul.vf AB11_im, AB11_im, ALPHA_re
    vfmul.vf AB12_re, AB12_re, ALPHA_re
    vfmul.vf AB12_im, AB12_im, ALPHA_re
    vfmul.vf AB13_re, AB13_re, ALPHA_re
    vfmul.vf AB13_im, AB13_im, ALPHA_re

MULTIPLYBETA:
    feq.s tmp, BETA_im, fa4  // fa4 is zero
    bne tmp, zero, BETAREAL

    // Load and deinterleave C(0:VLEN-1, 0:1)
    vlseg2e32.v C0_re, (C00_ptr)
    vlseg2e32.v C1_re, (C01_ptr)

    // Load and deinterleave C(0:VLEN-1, 2:3)
    vlseg2e32.v C2_re, (C02_ptr)
    vlseg2e32.v C3_re, (C03_ptr)

    // C(0:VLEN-1,0:1) * beta + AB(0:VLEN-1,0:1)
    vfmacc.vf AB00_re, BETA_re, C0_re
    vfnmsac.vf AB00_re, BETA_im, C0_im
    vfmacc.vf AB00_im, BETA_re, C0_im
    vfmacc.vf AB00_im, BETA_im, C0_re
    vsseg2e32.v AB00_re, (C00_ptr)
    vfmacc.vf AB01_re, BETA_re, C1_re
    vfnmsac.vf AB01_re, BETA_im, C1_im
    vfmacc.vf AB01_im, BETA_re, C1_im
    vfmacc.vf AB01_im, BETA_im, C1_re
    vsseg2e32.v AB01_re, (C01_ptr)

    // C(0:VLEN-1,2:3) * beta + AB(0:VLEN-1,2:3)
    vfmacc.vf AB02_re, BETA_re, C2_re
    vfnmsac.vf AB02_re, BETA_im, C2_im
    vfmacc.vf AB02_im, BETA_re, C2_im
    vfmacc.vf AB02_im, BETA_im, C2_re
    vsseg2e32.v AB02_re, (C02_ptr)

    vfmacc.vf AB03_re, BETA_re, C3_re
    vfnmsac.vf AB03_re, BETA_im, C3_im
    vfmacc.vf AB03_im, BETA_re, C3_im
    vfmacc.vf AB03_im, BETA_im, C3_re
    vsseg2e32.v AB03_re, (C03_ptr)


    // Load and deinterleave C(VLEN:2*VLEN-1, 0:1)
    vlseg2e32.v C0_re, (C10_ptr)
    vlseg2e32.v C1_re, (C11_ptr)

    // Load and deinterleave C(VLEN:2*VLEN-1, 2:3)
    vlseg2e32.v C2_re, (C12_ptr)
    vlseg2e32.v C3_re, (C13_ptr)

    // C(VLEN:2*VLEN-1,0:1) * beta + AB(VLEN:2*VLEN-1,0:1)
    vfmacc.vf AB10_re, BETA_re, C0_re
    vfnmsac.vf AB10_re, BETA_im, C0_im
    vfmacc.vf AB10_im, BETA_re, C0_im
    vfmacc.vf AB10_im, BETA_im, C0_re
    vsseg2e32.v AB10_re, (C10_ptr)
    vfmacc.vf AB11_re, BETA_re, C1_re
    vfnmsac.vf AB11_re, BETA_im, C1_im
    vfmacc.vf AB11_im, BETA_re, C1_im
    vfmacc.vf AB11_im, BETA_im, C1_re
    vsseg2e32.v AB11_re, (C11_ptr)

    // C(VLEN:2*VLEN-1,2:3) * beta + AB(VLEN:2*VLEN-1,2:3)
    vfmacc.vf AB12_re, BETA_re, C2_re
    vfnmsac.vf AB12_re, BETA_im, C2_im
    vfmacc.vf AB12_im, BETA_re, C2_im
    vfmacc.vf AB12_im, BETA_im, C2_re
    vsseg2e32.v AB12_re, (C12_ptr)
    vfmacc.vf AB13_re, BETA_re, C3_re
    vfnmsac.vf AB13_re, BETA_im, C3_im
    vfmacc.vf AB13_im, BETA_re, C3_im
    vfmacc.vf AB13_im, BETA_im, C3_re
    vsseg2e32.v AB13_re, (C13_ptr)

    j END

BETAREAL:
    feq.s tmp, BETA_re, fa4 // fa4 is zero
    bne tmp, zero, BETAZERO

    // Load and deinterleave C(0:VLEN-1, 0:3)
    vlseg2e32.v C0_re, (C00_ptr)
    vlseg2e32.v C1_re, (C01_ptr)
    vlseg2e32.v C2_re, (C02_ptr)
    vlseg2e32.v C3_re, (C03_ptr)

    // C(0:VLEN-1,0:3) * beta + AB(0:VLEN-1,0:3)
    vfmacc.vf AB00_re, BETA_re, C0_re
    vfmacc.vf AB00_im, BETA_re, C0_im
    vfmacc.vf AB01_re, BETA_re, C1_re
    vfmacc.vf AB01_im, BETA_re, C1_im
    vfmacc.vf AB02_re, BETA_re, C2_re
    vfmacc.vf AB02_im, BETA_re, C2_im
    vfmacc.vf AB03_re, BETA_re, C3_re
    vfmacc.vf AB03_im, BETA_re, C3_im
    vsseg2e32.v AB00_re, (C00_ptr)
    vsseg2e32.v AB01_re, (C01_ptr)
    vsseg2e32.v AB02_re, (C02_ptr)
    vsseg2e32.v AB03_re, (C03_ptr)

    // Load and deinterleave C(VLEN:2*VLEN-1, 0:3)
    vlseg2e32.v C0_re, (C10_ptr)
    vlseg2e32.v C1_re, (C11_ptr)
    vlseg2e32.v C2_re, (C12_ptr)
    vlseg2e32.v C3_re, (C13_ptr)

    // C(VLEN:2*VLEN-1,0:3) * beta + AB(VLEN:2*VLEN-1,0:3)
    vfmacc.vf AB10_re, BETA_re, C0_re
    vfmacc.vf AB10_im, BETA_re, C0_im
    vfmacc.vf AB11_re, BETA_re, C1_re
    vfmacc.vf AB11_im, BETA_re, C1_im
    vfmacc.vf AB12_re, BETA_re, C2_re
    vfmacc.vf AB12_im, BETA_re, C2_im
    vfmacc.vf AB13_re, BETA_re, C3_re
    vfmacc.vf AB13_im, BETA_re, C3_im
    vsseg2e32.v AB10_re, (C10_ptr)
    vsseg2e32.v AB11_re, (C11_ptr)
    vsseg2e32.v AB12_re, (C12_ptr)
    vsseg2e32.v AB13_re, (C13_ptr)

    j END

BETAZERO:
    vsseg2e32.v AB00_re, (C00_ptr)
    vsseg2e32.v AB01_re, (C01_ptr)
    vsseg2e32.v AB02_re, (C02_ptr)
    vsseg2e32.v AB03_re, (C03_ptr)

    vsseg2e32.v AB10_re, (C10_ptr)
    vsseg2e32.v AB11_re, (C11_ptr)
    vsseg2e32.v AB12_re, (C12_ptr)
    vsseg2e32.v AB13_re, (C13_ptr)

END:
    ld s7,56(sp)
    ld s6,48(sp)
    ld s5,40(sp)
    ld s4,32(sp)
    ld s3,24(sp)
    ld s2,16(sp)
    ld s1,8(sp)
    ld s0,0(sp)
    addi sp,sp,64
    ret
