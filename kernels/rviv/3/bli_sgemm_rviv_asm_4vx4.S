#define REALNAME bli_sgemm_rviv_asm_4vx4
    .text
    .global     REALNAME

// void REALNAME(uint64_t k, float* alpha, float* a, float* b,
//               float* beta, float* c, uint64_t rs_c, uint64_t cs_c)
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

#define A0_ptr a2
#define A1_ptr t0
#define A2_ptr t1
#define A3_ptr t2


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

#define ALPHA fa1
#define BETA  fa2

#define B00    fa4
#define B01    fa5
#define B02    fa6
#define B03    fa7

#define B10    fa0
#define B11    fa1
#define B12    fa2
#define B13    fa3

#define A00   v24
#define A10   v25
#define A20   v26
#define A30   v27

#define A01   v28
#define A11   v29
#define A21   v30
#define A31   v31

#define AB00   v0
#define AB01   v1
#define AB02   v2
#define AB03   v3
#define AB10   v4
#define AB11   v5
#define AB12   v6
#define AB13   v7
#define AB20   v8
#define AB21   v9
#define AB22   v10
#define AB23   v11
#define AB30   v12
#define AB31   v13
#define AB32   v14
#define AB33   v15

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
    slli s0, s0, 2      // vector length in bytes
    // Workaround for `mul rs_c, rs_c, s0` because rv64iv does not support `mul`
    mv rs_c, s0 // Assumes unit row skip
    slli cs_c, cs_c, 2  // count in FP32 steps

    // Set up pointers
    add C01_ptr, C00_ptr, cs_c
    add C02_ptr, C01_ptr, cs_c
    add C03_ptr, C02_ptr, cs_c
    add C10_ptr, C00_ptr, rs_c
    add C11_ptr, C01_ptr, rs_c
    add C12_ptr, C02_ptr, rs_c
    add C13_ptr, C03_ptr, rs_c

    add A1_ptr, A0_ptr, s0
    add A2_ptr, A1_ptr, s0
    add A3_ptr, A2_ptr, s0

    slli s0, s0, 2 // length of a column of A in bytes

    // Zero-initialize accumulators
    fmv.w.x fa3, zero
    vfmv.v.f AB00, fa3
    vfmv.v.f AB01, fa3
    vfmv.v.f AB02, fa3
    vfmv.v.f AB03, fa3
    vfmv.v.f AB10, fa3
    vfmv.v.f AB11, fa3
    vfmv.v.f AB12, fa3
    vfmv.v.f AB13, fa3
    vfmv.v.f AB20, fa3
    vfmv.v.f AB21, fa3
    vfmv.v.f AB22, fa3
    vfmv.v.f AB23, fa3
    vfmv.v.f AB30, fa3
    vfmv.v.f AB31, fa3
    vfmv.v.f AB32, fa3
    vfmv.v.f AB33, fa3

    li tmp, 1
    ble loop_counter, tmp, TAIL

    // Load B(l,0:3)
    flw B00, (B_row_ptr)
    flw B01, 4(B_row_ptr)
    flw B02, 8(B_row_ptr)
    flw B03, 12(B_row_ptr)

    // Load A(:,l)
    vle32.v A00, (A0_ptr)
    vle32.v A10, (A1_ptr)
    vle32.v A20, (A2_ptr)
    vle32.v A30, (A3_ptr)

LOOP_K:
    addi loop_counter, loop_counter, -2

    // Point to A(:,l+1)
    add A0_ptr, A0_ptr, s0
    add A1_ptr, A1_ptr, s0
    add A2_ptr, A2_ptr, s0
    add A3_ptr, A3_ptr, s0

    vfmacc.vf AB00, B00, A00
    vfmacc.vf AB01, B01, A00
    vfmacc.vf AB02, B02, A00
    vfmacc.vf AB03, B03, A00

    vfmacc.vf AB10, B00, A10
    vfmacc.vf AB11, B01, A10
    vfmacc.vf AB12, B02, A10
    vfmacc.vf AB13, B03, A10

    // Load B(l+1,0:3)
    flw B10, 16(B_row_ptr)
    flw B11, 20(B_row_ptr)
    flw B12, 24(B_row_ptr)
    flw B13, 28(B_row_ptr)
    addi B_row_ptr, B_row_ptr, 32

    // Load A(:,l+1)
    vle32.v A01, (A0_ptr)
    vle32.v A11, (A1_ptr)
    vle32.v A21, (A2_ptr)
    vle32.v A31, (A3_ptr)

    vfmacc.vf AB20, B00, A20
    vfmacc.vf AB21, B01, A20
    vfmacc.vf AB22, B02, A20
    vfmacc.vf AB23, B03, A20

    vfmacc.vf AB30, B00, A30
    vfmacc.vf AB31, B01, A30
    vfmacc.vf AB32, B02, A30
    vfmacc.vf AB33, B03, A30

    // Point to A(:,l+2)
    add A0_ptr, A0_ptr, s0
    add A1_ptr, A1_ptr, s0
    add A2_ptr, A2_ptr, s0
    add A3_ptr, A3_ptr, s0

    vfmacc.vf AB00, B10, A01
    vfmacc.vf AB01, B11, A01
    vfmacc.vf AB02, B12, A01
    vfmacc.vf AB03, B13, A01

    vfmacc.vf AB10, B10, A11
    vfmacc.vf AB11, B11, A11
    vfmacc.vf AB12, B12, A11
    vfmacc.vf AB13, B13, A11

    vfmacc.vf AB20, B10, A21
    vfmacc.vf AB21, B11, A21
    vfmacc.vf AB22, B12, A21
    vfmacc.vf AB23, B13, A21

    vfmacc.vf AB30, B10, A31
    vfmacc.vf AB31, B11, A31
    vfmacc.vf AB32, B12, A31
    vfmacc.vf AB33, B13, A31

    li tmp, 1
    ble loop_counter, tmp, TAIL

    // Load next row of B and next column of A for the next iteration
    flw B00, (B_row_ptr)
    flw B01, 4(B_row_ptr)
    flw B02, 8(B_row_ptr)
    flw B03, 12(B_row_ptr)

    vle32.v A00, (A0_ptr)
    vle32.v A10, (A1_ptr)
    vle32.v A20, (A2_ptr)
    vle32.v A30, (A3_ptr)

    j LOOP_K

TAIL:
    beqz loop_counter, MULTIPLYALPHA

    // Load row of B
    flw B00, (B_row_ptr)
    flw B01, 4(B_row_ptr)
    flw B02, 8(B_row_ptr)
    flw B03, 12(B_row_ptr)

    // Load A(:,l)
    vle32.v A00, (A0_ptr)
    vle32.v A10, (A1_ptr)
    vle32.v A20, (A2_ptr)
    vle32.v A30, (A3_ptr)

    vfmacc.vf AB00, B00, A00
    vfmacc.vf AB01, B01, A00
    vfmacc.vf AB02, B02, A00
    vfmacc.vf AB03, B03, A00

    vfmacc.vf AB10, B00, A10
    vfmacc.vf AB11, B01, A10
    vfmacc.vf AB12, B02, A10
    vfmacc.vf AB13, B03, A10

    vfmacc.vf AB20, B00, A20
    vfmacc.vf AB21, B01, A20
    vfmacc.vf AB22, B02, A20
    vfmacc.vf AB23, B03, A20

    vfmacc.vf AB30, B00, A30
    vfmacc.vf AB31, B01, A30
    vfmacc.vf AB32, B02, A30
    vfmacc.vf AB33, B03, A30


MULTIPLYALPHA:
    flw ALPHA,(a1)
    flw BETA, (a4)

    // Multiply with alpha
    vfmul.vf AB00, AB00, ALPHA
    vfmul.vf AB01, AB01, ALPHA
    vfmul.vf AB02, AB02, ALPHA
    vfmul.vf AB03, AB03, ALPHA

    vfmul.vf AB10, AB10, ALPHA
    vfmul.vf AB11, AB11, ALPHA
    vfmul.vf AB12, AB12, ALPHA
    vfmul.vf AB13, AB13, ALPHA

    vfmul.vf AB20, AB20, ALPHA
    vfmul.vf AB21, AB21, ALPHA
    vfmul.vf AB22, AB22, ALPHA
    vfmul.vf AB23, AB23, ALPHA

    vfmul.vf AB30, AB30, ALPHA
    vfmul.vf AB31, AB31, ALPHA
    vfmul.vf AB32, AB32, ALPHA
    vfmul.vf AB33, AB33, ALPHA

    fmv.w.x fa4, zero
    feq.s tmp, BETA, fa4
    beq tmp, zero, BETANOTZERO

BETAZERO:
    vse32.v AB00, (C00_ptr)
    vse32.v AB01, (C01_ptr)
    vse32.v AB02, (C02_ptr)
    vse32.v AB03, (C03_ptr)

    add C00_ptr, C10_ptr, rs_c  // advance pointers to row 2*VLEN
    add C01_ptr, C11_ptr, rs_c
    add C02_ptr, C12_ptr, rs_c
    add C03_ptr, C13_ptr, rs_c

    vse32.v AB10, (C10_ptr)
    vse32.v AB11, (C11_ptr)
    vse32.v AB12, (C12_ptr)
    vse32.v AB13, (C13_ptr)

    add C10_ptr, C00_ptr, rs_c  // advance pointers to row 3*VLEN
    add C11_ptr, C01_ptr, rs_c
    add C12_ptr, C02_ptr, rs_c
    add C13_ptr, C03_ptr, rs_c

    vse32.v AB20, (C00_ptr)
    vse32.v AB21, (C01_ptr)
    vse32.v AB22, (C02_ptr)
    vse32.v AB23, (C03_ptr)

    vse32.v AB30, (C10_ptr)
    vse32.v AB31, (C11_ptr)
    vse32.v AB32, (C12_ptr)
    vse32.v AB33, (C13_ptr)

    j END

BETANOTZERO:
    vle32.v v16, (C00_ptr)  // Load C(0:VLEN-1, 0:3)
    vle32.v v17, (C01_ptr)
    vle32.v v18, (C02_ptr)
    vle32.v v19, (C03_ptr)

    vfmacc.vf AB00, BETA, v16
    vfmacc.vf AB01, BETA, v17
    vfmacc.vf AB02, BETA, v18
    vfmacc.vf AB03, BETA, v19

    vse32.v AB00, (C00_ptr)  // Store C(0:VLEN-1, 0:3)
    vse32.v AB01, (C01_ptr)
    vse32.v AB02, (C02_ptr)
    vse32.v AB03, (C03_ptr)

    vle32.v v20, (C10_ptr)  // Load C(VLEN:2*VLEN-1, 0:3)
    vle32.v v21, (C11_ptr)
    vle32.v v22, (C12_ptr)
    vle32.v v23, (C13_ptr)

    vfmacc.vf AB10, BETA, v20
    vfmacc.vf AB11, BETA, v21
    vfmacc.vf AB12, BETA, v22
    vfmacc.vf AB13, BETA, v23

    vse32.v AB10, (C10_ptr)  // Store C(VLEN:2*VLEN-1, 0:3)
    vse32.v AB11, (C11_ptr)
    vse32.v AB12, (C12_ptr)
    vse32.v AB13, (C13_ptr)

    add C00_ptr, C10_ptr, rs_c  // advance pointers to row 2*VLEN
    add C01_ptr, C11_ptr, rs_c
    add C02_ptr, C12_ptr, rs_c
    add C03_ptr, C13_ptr, rs_c

    add C10_ptr, C00_ptr, rs_c  // advance pointers to row 3*VLEN
    add C11_ptr, C01_ptr, rs_c
    add C12_ptr, C02_ptr, rs_c
    add C13_ptr, C03_ptr, rs_c

    vle32.v v16, (C00_ptr)  // Load C(2*VLEN:3*VLEN-1, 0:3)
    vle32.v v17, (C01_ptr)
    vle32.v v18, (C02_ptr)
    vle32.v v19, (C03_ptr)

    vfmacc.vf AB20, BETA, v16
    vfmacc.vf AB21, BETA, v17
    vfmacc.vf AB22, BETA, v18
    vfmacc.vf AB23, BETA, v19

    vse32.v AB20, (C00_ptr)  // Store C(2*VLEN:3*VLEN-1, 0:3)
    vse32.v AB21, (C01_ptr)
    vse32.v AB22, (C02_ptr)
    vse32.v AB23, (C03_ptr)

    vle32.v v20, (C10_ptr)  // Load C(3*VLEN:4*VLEN-1, 0:3)
    vle32.v v21, (C11_ptr)
    vle32.v v22, (C12_ptr)
    vle32.v v23, (C13_ptr)

    vfmacc.vf AB30, BETA, v20
    vfmacc.vf AB31, BETA, v21
    vfmacc.vf AB32, BETA, v22
    vfmacc.vf AB33, BETA, v23

    vse32.v AB30, (C10_ptr)  // Store C(3*VLEN:4*VLEN-1, 0:3)
    vse32.v AB31, (C11_ptr)
    vse32.v AB32, (C12_ptr)
    vse32.v AB33, (C13_ptr)

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
