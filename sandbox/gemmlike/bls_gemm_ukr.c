
#include "bls_gemm_ukr.h"

void c_hello()
{
    printf("H!\n");
    return;
}

void c_print(_Float16* c,
             dim_t m,
             dim_t n,
             inc_t rs_c,
             inc_t cs_c
             )
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%f, ", (float)c[i*rs_c+j*cs_c]);
        }
        printf("\n");
    }
    return;
}


void bli_hgemm_armv8a_asm_h24x8r
    (
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const void*   alpha_,
        const void*       a,
        const void*       b,
        const void*    beta_,
              void*       c,
              inc_t   rs_c0,
              inc_t   cs_c0,
        const auxinfo_t* data,
        const cntx_t* cntx
    )
{
    
    uint64_t  k_mker = k / 4;
    uint64_t  k_left = k % 4;

    uint64_t  rs_c   = rs_c0;
    uint64_t  cs_c   = cs_c0;

    _Float16 alpha_cast = 1.0;
    _Float16 beta_cast = 1.0;
    _Float16* alpha = &alpha_cast;
    _Float16* beta = &beta_cast;


    //_Float16* c_cast_use     = ( _Float16* )c;
    //_Float16* a_cast_use     = ( _Float16* )a;
    //_Float16* b_cast_use     = ( _Float16* )b;

    //_Float16 alpha_cast     = *( _Float16* )alpha;
    //_Float16 beta_cast     = *( _Float16* )beta;
    //
    //printf("********B Panle in every UKR function call*****\n\n");
    //for(int i_ch =0; i_ch < 5; i_ch++)
    //{
    //    for(int j_ch = 0; j_ch < 8; j_ch++)
    //    {
    //        printf("%f, ", (float)b_cast_use[i_ch * 8 + j_ch]);
    //    }
    //    printf("\n");
    //}

    //printf("UKR is running!\n");

    //printf("alpha_cast = %f, beta_cast = %f\n", (float)alpha_cast, (float)beta_cast);

    //printf("C_first, A_first, B_first, C_last = %f, %f, %f, %f\n", (float)c_cast_use[0], (float)a_cast_use[0],  (float)b_cast_use[0], (float)c_cast_use[(m-1)*rs_c + (n-1)*cs_c]);

    //printf("k_mker, k_left = %d, %d\n", k_mker, k_left);
    //printf("C[0], A[0], B[0] = %f, %f, %f", c[0], a[0], b[0]);
    const uint64_t ldc = (rs_c == 1)? cs_c : rs_c;
    //printf("C_Addr = %p, A_Addr = %p, B_Addr = %p,  m = %d, n = %d, k = %d, rs_c0 = %d , cs_c0 = %d, ldc = %d\n", c, a, b, m, n, k, rs_c0, cs_c0, ldc);

    //printf("****************print C before the kernel *******************\n");
    //c_print(c_cast_use, m, n, rs_c, cs_c);
    //printf("****************print A before the kernel *******************\n");
    //c_print(a_cast_use, m, k, 1, 24);
    //printf("****************print B before the kernel *******************\n");
    //c_print(b_cast_use, k, n, 8, 1);
    __asm__ volatile
    (
    "ldr   x0,  %[a]               \n\t"  // Address of matrix A
    "ldr   x1,  %[b]               \n\t"  // Address of matrix B
    "                             \n\t"
    "mov   x2,  #24                \n\t"  // Column skip for A
    "mov   x3,  #8                 \n\t"  // Row_skip of B 

    "ldr   x5,  %[c]               \n\t"  // Address of matrix C
    "ldr   x6,  %[rs_c]            \n\t"  // Row_skip of C, (Column skip is 1)
    "                             \n\t"
    "                             \n\t"  //Multiply some address ships by the sizeof(half) 
    "lsl   x2, x2, #1             \n\t"  // cs_a
    "lsl   x3, x3, #1             \n\t"  // rs_b
    "lsl   x6, x6, #1             \n\t"  // rs_c
    "                             \n\t"
    //" cmp   %w[ct], wzr            \n\t"  
    "mov   x9,     x5             \n\t"      
    "ldr   x4, %[k_mker]          \n\t"
    "ldr   x8, %[k_left]          \n\t"
    
    "                             \n\t"
    "cmp   x4,    #0              \n\t"  // No-microkernel early return is a must to avoid out-of boundry read
    "b.eq  LDCLEAR_CCOLS%=        \n\t"
    "                             \n\t"  
    "ldr   q0,   [x0, #16*0]     \n\t" // load A
    "ldr   q1,   [x0, #16*1]     \n\t" 
    "ldr   q2,   [x0, #16*2]     \n\t"
    "add   x0,    x0,   x2        \n\t"
    "ldr   q3,   [x1, #16*0]     \n\t" // load B
    "add   x1,    x1,   x3        \n\t"
    "LDCLEAR_CCOLS%=:             \n\t"
    "                             \n\t" // clean vector
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 1/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 2/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v8.8h,  wzr            \n\t"
    "dup   v9.8h,  wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 3/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 4/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v10.8h, wzr            \n\t"
    "dup   v11.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 5/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 6/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v12.8h, wzr            \n\t"
    "dup   v13.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 7/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 8/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v14.8h, wzr            \n\t"
    "dup   v15.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 9/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 10/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v16.8h, wzr            \n\t"
    "dup   v17.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 11/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 12/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v18.8h, wzr            \n\t"
    "dup   v19.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 13/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 14/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v20.8h, wzr            \n\t"
    "dup   v21.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 15/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 16/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v22.8h, wzr            \n\t"
    "dup   v23.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 17/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 18/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v24.8h, wzr            \n\t"
    "dup   v25.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 19/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 20/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v26.8h, wzr            \n\t"
    "dup   v27.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 21/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 22/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v28.8h, wzr            \n\t"
    "dup   v29.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 23/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 24/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v30.8h, wzr            \n\t"
    "dup   v31.8h, wzr            \n\t"
    "                             \n\t"
    //"cmp   x4,    #0              \n\t"  // No-microkernel early return is a must to avoid out-of boundry read
    "b.eq  LHK_LEFT_LOOP%=         \n\t" // Need to be noted here, C is row-major, A is col-major, B is row-major
    "                             \n\t" //  if x4 (the loop_meker) is 0, then we will skip the loop 
    "LHK_MKER_LOOP%=:              \n\t" //"                              \n\t"  // First Loop Unrolling
    "fmla  v8.8h,   v3.8h, v0.h[0]   \n\t" 
    "fmla  v9.8h,   v3.8h, v0.h[1]   \n\t" 
    "fmla  v10.8h,  v3.8h, v0.h[2]   \n\t" 
    "fmla  v11.8h,  v3.8h, v0.h[3]   \n\t" 
    "fmla  v12.8h,  v3.8h, v0.h[4]   \n\t" 
    "fmla  v13.8h,  v3.8h, v0.h[5]   \n\t" 
    "fmla  v14.8h,  v3.8h, v0.h[6]   \n\t" 
    "fmla  v15.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]     \n\t"  // Load A1 for the next kernel excute
    "fmla  v16.8h,  v3.8h, v1.h[0]   \n\t" 
    "fmla  v17.8h,  v3.8h, v1.h[1]   \n\t" 
    "fmla  v18.8h,  v3.8h, v1.h[2]   \n\t" 
    "fmla  v19.8h,  v3.8h, v1.h[3]   \n\t" 
    "fmla  v20.8h,  v3.8h, v1.h[4]   \n\t" 
    "fmla  v21.8h,  v3.8h, v1.h[5]   \n\t" 
    "fmla  v22.8h,  v3.8h, v1.h[6]   \n\t" 
    "fmla  v23.8h,  v3.8h, v1.h[7]   \n\t" 
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "fmla  v24.8h,  v3.8h, v2.h[0]   \n\t" 
    "fmla  v25.8h,  v3.8h, v2.h[1]   \n\t" 
    "fmla  v26.8h,  v3.8h, v2.h[2]   \n\t" 
    "fmla  v27.8h,  v3.8h, v2.h[3]   \n\t" 
    "fmla  v28.8h,  v3.8h, v2.h[4]   \n\t" 
    "fmla  v29.8h,  v3.8h, v2.h[5]   \n\t" 
    "fmla  v30.8h,  v3.8h, v2.h[6]   \n\t" 
    "fmla  v31.8h,  v3.8h, v2.h[7]   \n\t" 
    "ldr   q2,     [x0, #16 * 2]     \n\t"  // Load A1 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "                                \n\t"  // Second Loop Unrolling
    "fmla  v8.8h,   v3.8h, v0.h[0]   \n\t" 
    "fmla  v9.8h,   v3.8h, v0.h[1]   \n\t" 
    "fmla  v10.8h,  v3.8h, v0.h[2]   \n\t" 
    "fmla  v11.8h,  v3.8h, v0.h[3]   \n\t" 
    "fmla  v12.8h,  v3.8h, v0.h[4]   \n\t" 
    "fmla  v13.8h,  v3.8h, v0.h[5]   \n\t" 
    "fmla  v14.8h,  v3.8h, v0.h[6]   \n\t" 
    "fmla  v15.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]     \n\t"  // Load A1 for the next kernel excute
    "fmla  v16.8h,  v3.8h, v1.h[0]   \n\t" 
    "fmla  v17.8h,  v3.8h, v1.h[1]   \n\t" 
    "fmla  v18.8h,  v3.8h, v1.h[2]   \n\t" 
    "fmla  v19.8h,  v3.8h, v1.h[3]   \n\t" 
    "fmla  v20.8h,  v3.8h, v1.h[4]   \n\t" 
    "fmla  v21.8h,  v3.8h, v1.h[5]   \n\t" 
    "fmla  v22.8h,  v3.8h, v1.h[6]   \n\t" 
    "fmla  v23.8h,  v3.8h, v1.h[7]   \n\t" 
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "fmla  v24.8h,  v3.8h, v2.h[0]   \n\t" 
    "fmla  v25.8h,  v3.8h, v2.h[1]   \n\t" 
    "fmla  v26.8h,  v3.8h, v2.h[2]   \n\t" 
    "fmla  v27.8h,  v3.8h, v2.h[3]   \n\t" 
    "fmla  v28.8h,  v3.8h, v2.h[4]   \n\t" 
    "fmla  v29.8h,  v3.8h, v2.h[5]   \n\t" 
    "fmla  v30.8h,  v3.8h, v2.h[6]   \n\t" 
    "fmla  v31.8h,  v3.8h, v2.h[7]   \n\t" 
    "ldr   q2,     [x0, #16 * 2]     \n\t"  // Load A1 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load A1 for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "                                \n\t"  // Third Loop Unrolling
    "fmla  v8.8h,   v3.8h, v0.h[0]   \n\t" 
    "fmla  v9.8h,   v3.8h, v0.h[1]   \n\t" 
    "fmla  v10.8h,  v3.8h, v0.h[2]   \n\t" 
    "fmla  v11.8h,  v3.8h, v0.h[3]   \n\t" 
    "fmla  v12.8h,  v3.8h, v0.h[4]   \n\t" 
    "fmla  v13.8h,  v3.8h, v0.h[5]   \n\t" 
    "fmla  v14.8h,  v3.8h, v0.h[6]   \n\t" 
    "fmla  v15.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]     \n\t"  // Load A0 for the next kernel excute
    "fmla  v16.8h,  v3.8h, v1.h[0]   \n\t" 
    "fmla  v17.8h,  v3.8h, v1.h[1]   \n\t" 
    "fmla  v18.8h,  v3.8h, v1.h[2]   \n\t" 
    "fmla  v19.8h,  v3.8h, v1.h[3]   \n\t" 
    "fmla  v20.8h,  v3.8h, v1.h[4]   \n\t" 
    "fmla  v21.8h,  v3.8h, v1.h[5]   \n\t" 
    "fmla  v22.8h,  v3.8h, v1.h[6]   \n\t" 
    "fmla  v23.8h,  v3.8h, v1.h[7]   \n\t" 
    "ldr   q1,     [x0, #16 * 1]    \n\t"  // Load A1 for the next kernel excute
    "fmla  v24.8h,  v3.8h, v2.h[0]   \n\t" 
    "fmla  v25.8h,  v3.8h, v2.h[1]   \n\t" 
    "fmla  v26.8h,  v3.8h, v2.h[2]   \n\t" 
    "fmla  v27.8h,  v3.8h, v2.h[3]   \n\t" 
    "fmla  v28.8h,  v3.8h, v2.h[4]   \n\t" 
    "fmla  v29.8h,  v3.8h, v2.h[5]   \n\t" 
    "fmla  v30.8h,  v3.8h, v2.h[6]   \n\t" 
    "fmla  v31.8h,  v3.8h, v2.h[7]   \n\t" 
    "ldr   q2,     [x0, #16 * 2]     \n\t"  // Load A2 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "                                \n\t"  // Fourth Loop Unrolling
    "cmp    x4,     #1               \n\t"
    "b.eq   LH_LAST_UNROLL_LOOP%=    \n\t"
    "fmla  v8.8h,   v3.8h, v0.h[0]   \n\t" 
    "fmla  v9.8h,   v3.8h, v0.h[1]   \n\t" 
    "fmla  v10.8h,  v3.8h, v0.h[2]   \n\t" 
    "fmla  v11.8h,  v3.8h, v0.h[3]   \n\t" 
    "fmla  v12.8h,  v3.8h, v0.h[4]   \n\t" 
    "fmla  v13.8h,  v3.8h, v0.h[5]   \n\t" 
    "fmla  v14.8h,  v3.8h, v0.h[6]   \n\t" 
    "fmla  v15.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]     \n\t"  // Load A0 for the next kernel excute
    "fmla  v16.8h,  v3.8h, v1.h[0]   \n\t" 
    "fmla  v17.8h,  v3.8h, v1.h[1]   \n\t" 
    "fmla  v18.8h,  v3.8h, v1.h[2]   \n\t" 
    "fmla  v19.8h,  v3.8h, v1.h[3]   \n\t" 
    "fmla  v20.8h,  v3.8h, v1.h[4]   \n\t" 
    "fmla  v21.8h,  v3.8h, v1.h[5]   \n\t" 
    "fmla  v22.8h,  v3.8h, v1.h[6]   \n\t" 
    "fmla  v23.8h,  v3.8h, v1.h[7]   \n\t" 
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "fmla  v24.8h,  v3.8h, v2.h[0]   \n\t" 
    "fmla  v25.8h,  v3.8h, v2.h[1]   \n\t" 
    "fmla  v26.8h,  v3.8h, v2.h[2]   \n\t" 
    "fmla  v27.8h,  v3.8h, v2.h[3]   \n\t" 
    "fmla  v28.8h,  v3.8h, v2.h[4]   \n\t" 
    "fmla  v29.8h,  v3.8h, v2.h[5]   \n\t" 
    "fmla  v30.8h,  v3.8h, v2.h[6]   \n\t" 
    "fmla  v31.8h,  v3.8h, v2.h[7]   \n\t" 
    "ldr   q2,     [x0, #16 * 2]     \n\t"  // Load A2 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "sub   x4,      x4,       #1     \n\t"
    "cmp   x4,      #0               \n\t"
    "b.ne  LHK_MKER_LOOP%=           \n\t"
    "                                \n\t"  // end of unroll part
    "LH_LAST_UNROLL_LOOP%=:          \n\t"
    "fmla  v8.8h,   v3.8h, v0.h[0]   \n\t" 
    "fmla  v9.8h,   v3.8h, v0.h[1]   \n\t" 
    "fmla  v10.8h,  v3.8h, v0.h[2]   \n\t" 
    "fmla  v11.8h,  v3.8h, v0.h[3]   \n\t" 
    "fmla  v12.8h,  v3.8h, v0.h[4]   \n\t" 
    "fmla  v13.8h,  v3.8h, v0.h[5]   \n\t" 
    "fmla  v14.8h,  v3.8h, v0.h[6]   \n\t" 
    "fmla  v15.8h,  v3.8h, v0.h[7]   \n\t"
    "fmla  v16.8h,  v3.8h, v1.h[0]   \n\t" 
    "fmla  v17.8h,  v3.8h, v1.h[1]   \n\t" 
    "fmla  v18.8h,  v3.8h, v1.h[2]   \n\t" 
    "fmla  v19.8h,  v3.8h, v1.h[3]   \n\t" 
    "fmla  v20.8h,  v3.8h, v1.h[4]   \n\t" 
    "fmla  v21.8h,  v3.8h, v1.h[5]   \n\t" 
    "fmla  v22.8h,  v3.8h, v1.h[6]   \n\t" 
    "fmla  v23.8h,  v3.8h, v1.h[7]   \n\t" 
    "fmla  v24.8h,  v3.8h, v2.h[0]   \n\t" 
    "fmla  v25.8h,  v3.8h, v2.h[1]   \n\t" 
    "fmla  v26.8h,  v3.8h, v2.h[2]   \n\t" 
    "fmla  v27.8h,  v3.8h, v2.h[3]   \n\t" 
    "fmla  v28.8h,  v3.8h, v2.h[4]   \n\t" 
    "fmla  v29.8h,  v3.8h, v2.h[5]   \n\t" 
    "fmla  v30.8h,  v3.8h, v2.h[6]   \n\t" 
    "fmla  v31.8h,  v3.8h, v2.h[7]   \n\t" 
    "LHK_LEFT_LOOP%=:                \n\t"  // Label for if MK_MKER to ZREO 
    "                                \n\t"
    "cmp   x8,              #0       \n\t" 
    "b.eq  LHWRITE_MEM_PREP%=        \n\t"
    "ldr   q0,   [x0, #16*0]         \n\t" // load A0, A1, A2
    "ldr   q1,   [x0, #16*1]         \n\t" 
    "ldr   q2,   [x0, #16*2]         \n\t"
    "add   x0,    x0,   x2           \n\t"
    "ldr   q3,   [x1, #16*0]         \n\t" // load B
    "add   x1,    x1,   x3           \n\t"
    "sub   x8,    x8,   #1           \n\t"
    "                                \n\t"
    "fmla  v8.8h,   v3.8h, v0.h[0]   \n\t" 
    "fmla  v9.8h,   v3.8h, v0.h[1]   \n\t" 
    "fmla  v10.8h,  v3.8h, v0.h[2]   \n\t" 
    "fmla  v11.8h,  v3.8h, v0.h[3]   \n\t" 
    "fmla  v12.8h,  v3.8h, v0.h[4]   \n\t" 
    "fmla  v13.8h,  v3.8h, v0.h[5]   \n\t" 
    "fmla  v14.8h,  v3.8h, v0.h[6]   \n\t" 
    "fmla  v15.8h,  v3.8h, v0.h[7]   \n\t"
    "fmla  v16.8h,  v3.8h, v1.h[0]   \n\t" 
    "fmla  v17.8h,  v3.8h, v1.h[1]   \n\t" 
    "fmla  v18.8h,  v3.8h, v1.h[2]   \n\t" 
    "fmla  v19.8h,  v3.8h, v1.h[3]   \n\t" 
    "fmla  v20.8h,  v3.8h, v1.h[4]   \n\t" 
    "fmla  v21.8h,  v3.8h, v1.h[5]   \n\t" 
    "fmla  v22.8h,  v3.8h, v1.h[6]   \n\t" 
    "fmla  v23.8h,  v3.8h, v1.h[7]   \n\t" 
    "fmla  v24.8h,  v3.8h, v2.h[0]   \n\t" 
    "fmla  v25.8h,  v3.8h, v2.h[1]   \n\t" 
    "fmla  v26.8h,  v3.8h, v2.h[2]   \n\t" 
    "fmla  v27.8h,  v3.8h, v2.h[3]   \n\t" 
    "fmla  v28.8h,  v3.8h, v2.h[4]   \n\t" 
    "fmla  v29.8h,  v3.8h, v2.h[5]   \n\t" 
    "fmla  v30.8h,  v3.8h, v2.h[6]   \n\t" 
    "fmla  v31.8h,  v3.8h, v2.h[7]   \n\t" 
    "b    LHK_LEFT_LOOP%=            \n\t"
    "LHWRITE_MEM_PREP%=:             \n\t"
    "                                \n\t"
    "                                \n\t" // // Scale and write to memory.
    "ldr   x4,       %[alpha]        \n\t" // load alpha & beta address
    "ldr   x8,       %[beta]         \n\t"
    "ld1r  {v0.8h}, [x4]             \n\t" // load alpha & beta
    "ld1r  {v1.8h}, [x8]             \n\t" 
    "fmov  h2,      #1.0             \n\t"
    "fcmp  h0,      h2               \n\t" // alpha is 1.0 or not
    "b.eq  LH_UNIT_ALPHA%=            \n\t"
    "fmul  v8.8h,   v8.8h,  v0.8h    \n\t"
    "fmul  v9.8h,   v9.8h,  v0.8h    \n\t"
    "fmul  v10.8h,  v10.8h, v0.8h    \n\t"
    "fmul  v11.8h,  v11.8h, v0.8h    \n\t"
    "fmul  v12.8h,  v12.8h, v0.8h    \n\t"
    "fmul  v13.8h,  v13.8h, v0.8h    \n\t"
    "fmul  v14.8h,  v14.8h, v0.8h    \n\t"
    "fmul  v15.8h,  v15.8h, v0.8h    \n\t"
    "fmul  v16.8h,  v16.8h, v0.8h    \n\t"
    "fmul  v17.8h,  v17.8h, v0.8h    \n\t"
    "fmul  v18.8h,  v18.8h, v0.8h    \n\t"
    "fmul  v19.8h,  v19.8h, v0.8h    \n\t"
    "fmul  v20.8h,  v20.8h, v0.8h    \n\t"
    "fmul  v21.8h,  v21.8h, v0.8h    \n\t"
    "fmul  v22.8h,  v22.8h, v0.8h    \n\t"
    "fmul  v23.8h,  v23.8h, v0.8h    \n\t"
    "fmul  v24.8h,  v24.8h, v0.8h    \n\t"
    "fmul  v25.8h,  v25.8h, v0.8h    \n\t"
    "fmul  v26.8h,  v26.8h, v0.8h    \n\t"
    "fmul  v27.8h,  v27.8h, v0.8h    \n\t"
    "fmul  v28.8h,  v28.8h, v0.8h    \n\t"
    "fmul  v29.8h,  v29.8h, v0.8h    \n\t"
    "fmul  v30.8h,  v30.8h, v0.8h    \n\t"
    "fmul  v31.8h,  v31.8h, v0.8h    \n\t"
    "LH_UNIT_ALPHA%=:                \n\t"
    "mov   x9,      x5               \n\t" // loading C address
    "fcmp  h1,    #0.0               \n\t" // test beta is 0.0 or not
    "b.eq  LH_BETA_ZERO_R_8_13%=            \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q5,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q7,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v8.8h,   v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v9.8h,   v3.8h,  v1.h[0]  \n\t"
    "fmla  v10.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v11.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v12.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v13.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_8_13%=:           \n\t"
    "str   q8,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q9,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q10,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q11,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q12,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q13,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LH_BETA_ZERO_R_14_19%=     \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q5,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q7,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v14.8h,  v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v15.8h,  v3.8h,  v1.h[0]  \n\t"
    "fmla  v16.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v17.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v18.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v19.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_14_19%=:          \n\t"
    "str   q14,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q15,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q16,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q17,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q18,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q19,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LH_BETA_ZERO_R_20_25%=    \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q5,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q7,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v20.8h,  v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v21.8h,  v3.8h,  v1.h[0]  \n\t"
    "fmla  v22.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v23.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v24.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v25.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_20_25%=:         \n\t"
    "str   q20,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q21,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q22,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q23,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q24,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q25,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LH_BETA_ZERO_R_26_31%=    \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q5,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q7,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v26.8h,  v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v27.8h,  v3.8h,  v1.h[0]  \n\t"
    "fmla  v28.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v29.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v30.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v31.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_26_31%=:         \n\t"
    "str   q26,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q27,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q28,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q29,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q30,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q31,     [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    :// output operands(none)
    :[a]        "m"      (a), 
     [b]        "m"      (b),
     [c]        "m"      (c),
     [rs_c]     "m"   (rs_c),
     [k_mker]   "m" (k_mker),
     [k_left]   "m" (k_left),
     [alpha]    "m"  (alpha),
     [beta]     "m"   (beta) //[ct]       "m"     (ct)
    :"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
     "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
     "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
     "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );
    
    //printf("C_first, A_first, B_first = %f, %f, %f\n", (float)c_cast_use[0], (float)a_cast_use[0],  (float)b_cast_use[0]);
    //printf("****************print C after the kernel *******************\n");
    //c_print(c_cast_use, m, n, rs_c, cs_c);
    return;
}



void bli_hgemm_armv8a_asm_h12x16r(
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const void*   alpha_,
        const void*       a,
        const void*       b,
        const void*    beta_,
              void*       c,
              inc_t   rs_c0,
              inc_t   cs_c0,
        const auxinfo_t* data,
        const cntx_t* cntx
        )
{

    uint64_t  k_mker = k / 4;
    uint64_t  k_left = k % 4;

    uint64_t  rs_c   = rs_c0;
    uint64_t  cs_c   = cs_c0;

    _Float16 alpha_cast = 1.0;
    _Float16 beta_cast = 1.0;
    _Float16* alpha = &alpha_cast;
    _Float16* beta = &beta_cast;

    const uint64_t ldc = (rs_c == 1)? cs_c : rs_c;
        __asm__ volatile
    (
    "ldr   x0,  %[a]               \n\t"  // Address of matrix A
    "ldr   x1,  %[b]               \n\t"  // Address of matrix B
    "                              \n\t"
    "mov   x2,  #12                \n\t"  // Column skip for A
    "mov   x3,  #16                \n\t"  // Row_skip of B

    "ldr   x5,  %[c]               \n\t"  // Address of matrix C
    "ldr   x6,  %[rs_c]            \n\t"  // Row_skip of C, (Column skip is 1)
    "                              \n\t"
    "                              \n\t"  //Multiply some address ships by the sizeof(half)
    "lsl   x2, x2, #1              \n\t"  // cs_a
    "lsl   x3, x3, #1              \n\t"  // rs_b
    "lsl   x6, x6, #1              \n\t"  // rs_c
    "                              \n\t"
    //" cmp   %w[ct], wzr            \n\t"
    "mov   x9,     x5             \n\t"
    "ldr   x4, %[k_mker]          \n\t"
    "ldr   x8, %[k_left]          \n\t"

    "                             \n\t"
    "cmp   x4,    #0              \n\t"  // No-microkernel early return is a must to avoid out-of boundry read
    "b.eq  LDCLEAR_CCOLS%=        \n\t"
    "                             \n\t"
    "ldr   q0,   [x0, #16*0]     \n\t" // load A
    "ldr   q1,   [x0, #16*1]     \n\t"
    "add   x0,    x0,   x2        \n\t"
    "ldr   q2,   [x1, #16*0]     \n\t"
    "ldr   q3,   [x1, #16*1]     \n\t" // load B
    "add   x1,    x1,   x3        \n\t"
    "LDCLEAR_CCOLS%=:             \n\t"
    "                             \n\t" // clean vector
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 1,2/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v8.8h,  wzr            \n\t"
    "dup   v9.8h,  wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 3,4/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v10.8h, wzr            \n\t"
    "dup   v11.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 5,6/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v12.8h, wzr            \n\t"
    "dup   v13.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 7,8/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v14.8h, wzr            \n\t"
    "dup   v15.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 9,10/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v16.8h, wzr            \n\t"
    "dup   v17.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 11,12/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v18.8h, wzr            \n\t"
    "dup   v19.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 13,14/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v20.8h, wzr            \n\t"
    "dup   v21.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 15,16/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v22.8h, wzr            \n\t"
    "dup   v23.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 17,18/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v24.8h, wzr            \n\t"
    "dup   v25.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 19,20/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v26.8h, wzr            \n\t"
    "dup   v27.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 21,22/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v28.8h, wzr            \n\t"
    "dup   v29.8h, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 23,24/24
    "prfm  PLDL1KEEP, [x9, #32]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v30.8h, wzr            \n\t"
    "dup   v31.8h, wzr            \n\t"
    "                             \n\t"
    //"cmp   x4,    #0              \n\t"  // No-microkernel early return is a must to avoid out-of boundry read
    "b.eq  LHK_LEFT_LOOP%=        \n\t" // Need to be noted here, C is row-major, A is col-major, B is row-major
    "                             \n\t" //  if x4 (the loop_meker) is 0, then we will skip the loop
    "LHK_MKER_LOOP%=:             \n\t" //"                              \n\t"  // First Loop Unrolling
    "fmla  v8.8h,   v2.8h, v0.h[0]   \n\t"
    "fmla  v9.8h,   v3.8h, v0.h[0]   \n\t"
    "fmla  v10.8h,  v2.8h, v0.h[1]   \n\t"
    "fmla  v11.8h,  v3.8h, v0.h[1]   \n\t"
    "fmla  v12.8h,  v2.8h, v0.h[2]   \n\t"
    "fmla  v13.8h,  v3.8h, v0.h[2]   \n\t"
    "fmla  v14.8h,  v2.8h, v0.h[3]   \n\t"
    "fmla  v15.8h,  v3.8h, v0.h[3]   \n\t"
    "fmla  v16.8h,  v2.8h, v0.h[4]   \n\t"
    "fmla  v17.8h,  v3.8h, v0.h[4]   \n\t"
    "fmla  v18.8h,  v2.8h, v0.h[5]   \n\t"
    "fmla  v19.8h,  v3.8h, v0.h[5]   \n\t"
    "fmla  v20.8h,  v2.8h, v0.h[6]   \n\t"
    "fmla  v21.8h,  v3.8h, v0.h[6]   \n\t"
    "fmla  v22.8h,  v2.8h, v0.h[7]   \n\t"
    "fmla  v23.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,      [x0, #16 *  0]   \n\t"  // Load A0 for the next kernel excute
    "fmla  v24.8h,  v2.8h, v1.h[0]   \n\t"
    "fmla  v25.8h,  v3.8h, v1.h[0]   \n\t"
    "fmla  v26.8h,  v2.8h, v1.h[1]   \n\t"
    "fmla  v27.8h,  v3.8h, v1.h[1]   \n\t"
    "fmla  v28.8h,  v2.8h, v1.h[2]   \n\t"
    "fmla  v29.8h,  v3.8h, v1.h[2]   \n\t"
    "fmla  v30.8h,  v2.8h, v1.h[3]   \n\t"
    "fmla  v31.8h,  v3.8h, v1.h[3]   \n\t"
    "ldr   q1,      [x0, #16 *  1]   \n\t"  // Load A1 for the next kernel excute
    "ldr   q2,      [x1,       #0]   \n\t"  // Load B0 for the next kernel excute
    "ldr   q3,      [x1, #16 *  1]   \n\t"  // Load B1 for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    //"                                \n\t"
    //"                                \n\t"  // Second Loop Unrolling
    "fmla  v8.8h,   v2.8h, v0.h[0]   \n\t"
    "fmla  v9.8h,   v3.8h, v0.h[0]   \n\t"
    "fmla  v10.8h,  v2.8h, v0.h[1]   \n\t"
    "fmla  v11.8h,  v3.8h, v0.h[1]   \n\t"
    "fmla  v12.8h,  v2.8h, v0.h[2]   \n\t"
    "fmla  v13.8h,  v3.8h, v0.h[2]   \n\t"
    "fmla  v14.8h,  v2.8h, v0.h[3]   \n\t"
    "fmla  v15.8h,  v3.8h, v0.h[3]   \n\t"
    "fmla  v16.8h,  v2.8h, v0.h[4]   \n\t"
    "fmla  v17.8h,  v3.8h, v0.h[4]   \n\t"
    "fmla  v18.8h,  v2.8h, v0.h[5]   \n\t"
    "fmla  v19.8h,  v3.8h, v0.h[5]   \n\t"
    "fmla  v20.8h,  v2.8h, v0.h[6]   \n\t"
    "fmla  v21.8h,  v3.8h, v0.h[6]   \n\t"
    "fmla  v22.8h,  v2.8h, v0.h[7]   \n\t"
    "fmla  v23.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,      [x0, #16 *  0]   \n\t"  // Load A0 for the next kernel excute
    "fmla  v24.8h,  v2.8h, v1.h[0]   \n\t"
    "fmla  v25.8h,  v3.8h, v1.h[0]   \n\t"
    "fmla  v26.8h,  v2.8h, v1.h[1]   \n\t"
    "fmla  v27.8h,  v3.8h, v1.h[1]   \n\t"
    "fmla  v28.8h,  v2.8h, v1.h[2]   \n\t"
    "fmla  v29.8h,  v3.8h, v1.h[2]   \n\t"
    "fmla  v30.8h,  v2.8h, v1.h[3]   \n\t"
    "fmla  v31.8h,  v3.8h, v1.h[3]   \n\t"
    "ldr   q1,      [x0, #16 *  1]   \n\t"  // Load A1 for the next kernel excute
    "ldr   q2,      [x1,       #0]   \n\t"  // Load B0 for the next kernel excute
    "ldr   q3,      [x1, #16 *  1]   \n\t"  // Load B1 for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    //"                                \n\t"
    //"                                \n\t"  // Third Loop Unrolling
    "fmla  v8.8h,   v2.8h, v0.h[0]   \n\t"
    "fmla  v9.8h,   v3.8h, v0.h[0]   \n\t"
    "fmla  v10.8h,  v2.8h, v0.h[1]   \n\t"
    "fmla  v11.8h,  v3.8h, v0.h[1]   \n\t"
    "fmla  v12.8h,  v2.8h, v0.h[2]   \n\t"
    "fmla  v13.8h,  v3.8h, v0.h[2]   \n\t"
    "fmla  v14.8h,  v2.8h, v0.h[3]   \n\t"
    "fmla  v15.8h,  v3.8h, v0.h[3]   \n\t"
    "fmla  v16.8h,  v2.8h, v0.h[4]   \n\t"
    "fmla  v17.8h,  v3.8h, v0.h[4]   \n\t"
    "fmla  v18.8h,  v2.8h, v0.h[5]   \n\t"
    "fmla  v19.8h,  v3.8h, v0.h[5]   \n\t"
    "fmla  v20.8h,  v2.8h, v0.h[6]   \n\t"
    "fmla  v21.8h,  v3.8h, v0.h[6]   \n\t"
    "fmla  v22.8h,  v2.8h, v0.h[7]   \n\t"
    "fmla  v23.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,      [x0, #16 *  0]   \n\t"  // Load A0 for the next kernel excute
    "fmla  v24.8h,  v2.8h, v1.h[0]   \n\t"
    "fmla  v25.8h,  v3.8h, v1.h[0]   \n\t"
    "fmla  v26.8h,  v2.8h, v1.h[1]   \n\t"
    "fmla  v27.8h,  v3.8h, v1.h[1]   \n\t"
    "fmla  v28.8h,  v2.8h, v1.h[2]   \n\t"
    "fmla  v29.8h,  v3.8h, v1.h[2]   \n\t"
    "fmla  v30.8h,  v2.8h, v1.h[3]   \n\t"
    "fmla  v31.8h,  v3.8h, v1.h[3]   \n\t"
    "ldr   q1,      [x0, #16 *  1]   \n\t"  // Load A1 for the next kernel excute
    "ldr   q2,      [x1,       #0]   \n\t"  // Load B0 for the next kernel excute
    "ldr   q3,      [x1, #16 *  1]   \n\t"  // Load B1 for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    //"                                \n\t"
    //"                                \n\t"  // Fourth Loop Unrolling
    "cmp   x4,      #1               \n\t"
    "b.eq  LH_LAST_UNROLL_LOOP%=     \n\t"
    "fmla  v8.8h,   v2.8h, v0.h[0]   \n\t"
    "fmla  v9.8h,   v3.8h, v0.h[0]   \n\t"
    "fmla  v10.8h,  v2.8h, v0.h[1]   \n\t"
    "fmla  v11.8h,  v3.8h, v0.h[1]   \n\t"
    "fmla  v12.8h,  v2.8h, v0.h[2]   \n\t"
    "fmla  v13.8h,  v3.8h, v0.h[2]   \n\t"
    "fmla  v14.8h,  v2.8h, v0.h[3]   \n\t"
    "fmla  v15.8h,  v3.8h, v0.h[3]   \n\t"
    "fmla  v16.8h,  v2.8h, v0.h[4]   \n\t"
    "fmla  v17.8h,  v3.8h, v0.h[4]   \n\t"
    "fmla  v18.8h,  v2.8h, v0.h[5]   \n\t"
    "fmla  v19.8h,  v3.8h, v0.h[5]   \n\t"
    "fmla  v20.8h,  v2.8h, v0.h[6]   \n\t"
    "fmla  v21.8h,  v3.8h, v0.h[6]   \n\t"
    "fmla  v22.8h,  v2.8h, v0.h[7]   \n\t"
    "fmla  v23.8h,  v3.8h, v0.h[7]   \n\t"
    "ldr   q0,      [x0, #16 *  0]   \n\t"  // Load A0 for the next kernel excute
    "fmla  v24.8h,  v2.8h, v1.h[0]   \n\t"
    "fmla  v25.8h,  v3.8h, v1.h[0]   \n\t"
    "fmla  v26.8h,  v2.8h, v1.h[1]   \n\t"
    "fmla  v27.8h,  v3.8h, v1.h[1]   \n\t"
    "fmla  v28.8h,  v2.8h, v1.h[2]   \n\t"
    "fmla  v29.8h,  v3.8h, v1.h[2]   \n\t"
    "fmla  v30.8h,  v2.8h, v1.h[3]   \n\t"
    "fmla  v31.8h,  v3.8h, v1.h[3]   \n\t"
    "ldr   q1,      [x0, #16 *  1]   \n\t"  // Load A1 for the next kernel excute
    "ldr   q2,      [x1,       #0]   \n\t"  // Load B0 for the next kernel excute
    "ldr   q3,      [x1, #16 *  1]   \n\t"  // Load B1 for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "sub   x4,      x4,       #1     \n\t"
    "cmp   x4,      #0               \n\t"
    "b.ne  LHK_MKER_LOOP%=           \n\t"
    "                                \n\t"  // end of unroll part
    "LH_LAST_UNROLL_LOOP%=:          \n\t"
    "fmla  v8.8h,   v2.8h, v0.h[0]   \n\t"
    "fmla  v9.8h,   v3.8h, v0.h[0]   \n\t"
    "fmla  v10.8h,  v2.8h, v0.h[1]   \n\t"
    "fmla  v11.8h,  v3.8h, v0.h[1]   \n\t"
    "fmla  v12.8h,  v2.8h, v0.h[2]   \n\t"
    "fmla  v13.8h,  v3.8h, v0.h[2]   \n\t"
    "fmla  v14.8h,  v2.8h, v0.h[3]   \n\t"
    "fmla  v15.8h,  v3.8h, v0.h[3]   \n\t"
    "fmla  v16.8h,  v2.8h, v0.h[4]   \n\t"
    "fmla  v17.8h,  v3.8h, v0.h[4]   \n\t"
    "fmla  v18.8h,  v2.8h, v0.h[5]   \n\t"
    "fmla  v19.8h,  v3.8h, v0.h[5]   \n\t"
    "fmla  v20.8h,  v2.8h, v0.h[6]   \n\t"
    "fmla  v21.8h,  v3.8h, v0.h[6]   \n\t"
    "fmla  v22.8h,  v2.8h, v0.h[7]   \n\t"
    "fmla  v23.8h,  v3.8h, v0.h[7]   \n\t"
    "fmla  v24.8h,  v2.8h, v1.h[0]   \n\t"
    "fmla  v25.8h,  v3.8h, v1.h[0]   \n\t"
    "fmla  v26.8h,  v2.8h, v1.h[1]   \n\t"
    "fmla  v27.8h,  v3.8h, v1.h[1]   \n\t"
    "fmla  v28.8h,  v2.8h, v1.h[2]   \n\t"
    "fmla  v29.8h,  v3.8h, v1.h[2]   \n\t"
    "fmla  v30.8h,  v2.8h, v1.h[3]   \n\t"
    "fmla  v31.8h,  v3.8h, v1.h[3]   \n\t"
    "                                \n\t"
    "LHK_LEFT_LOOP%=:                \n\t"  // Label for if MK_MKER to ZREO
    "                                \n\t"
    "cmp   x8,              #0       \n\t"
    "b.eq  LHWRITE_MEM_PREP%=        \n\t"
    "ldr   q0,   [x0, #16*0]         \n\t" // load A0, A1
    "ldr   q1,   [x0, #16*1]         \n\t"
    "add   x0,    x0,   x2           \n\t"
    "ldr   q2,   [x1, #16*0]         \n\t"
    "ldr   q3,   [x1, #16*1]         \n\t" // load B0, B1
    "add   x1,    x1,   x3           \n\t"
    "sub   x8,    x8,   #1           \n\t"
    "                                \n\t"
    "fmla  v8.8h,   v2.8h, v0.h[0]   \n\t"
    "fmla  v9.8h,   v3.8h, v0.h[0]   \n\t"
    "fmla  v10.8h,  v2.8h, v0.h[1]   \n\t"
    "fmla  v11.8h,  v3.8h, v0.h[1]   \n\t"
    "fmla  v12.8h,  v2.8h, v0.h[2]   \n\t"
    "fmla  v13.8h,  v3.8h, v0.h[2]   \n\t"
    "fmla  v14.8h,  v2.8h, v0.h[3]   \n\t"
    "fmla  v15.8h,  v3.8h, v0.h[3]   \n\t"
    "fmla  v16.8h,  v2.8h, v0.h[4]   \n\t"
    "fmla  v17.8h,  v3.8h, v0.h[4]   \n\t"
    "fmla  v18.8h,  v2.8h, v0.h[5]   \n\t"
    "fmla  v19.8h,  v3.8h, v0.h[5]   \n\t"
    "fmla  v20.8h,  v2.8h, v0.h[6]   \n\t"
    "fmla  v21.8h,  v3.8h, v0.h[6]   \n\t"
    "fmla  v22.8h,  v2.8h, v0.h[7]   \n\t"
    "fmla  v23.8h,  v3.8h, v0.h[7]   \n\t"
    "fmla  v24.8h,  v2.8h, v1.h[0]   \n\t"
    "fmla  v25.8h,  v3.8h, v1.h[0]   \n\t"
    "fmla  v26.8h,  v2.8h, v1.h[1]   \n\t"
    "fmla  v27.8h,  v3.8h, v1.h[1]   \n\t"
    "fmla  v28.8h,  v2.8h, v1.h[2]   \n\t"
    "fmla  v29.8h,  v3.8h, v1.h[2]   \n\t"
    "fmla  v30.8h,  v2.8h, v1.h[3]   \n\t"
    "fmla  v31.8h,  v3.8h, v1.h[3]   \n\t"
    "b    LHK_LEFT_LOOP%=            \n\t"
    "LHWRITE_MEM_PREP%=:             \n\t"
    "                                \n\t"
    "                                \n\t" // // Scale and write to memory.
    "ldr   x4,       %[alpha]        \n\t" // load alpha & beta address
    "ldr   x8,       %[beta]         \n\t"
    "ld1r  {v0.8h}, [x4]             \n\t" // load alpha & beta
    "ld1r  {v1.8h}, [x8]             \n\t"
    "fmov  h2,      #1.0             \n\t"
    "fcmp  h0,      h2               \n\t" // alpha is 1.0 or not
    "b.eq  LH_UNIT_ALPHA%=            \n\t"
    "fmul  v8.8h,   v8.8h,  v0.8h    \n\t"
    "fmul  v9.8h,   v9.8h,  v0.8h    \n\t"
    "fmul  v10.8h,  v10.8h, v0.8h    \n\t"
    "fmul  v11.8h,  v11.8h, v0.8h    \n\t"
    "fmul  v12.8h,  v12.8h, v0.8h    \n\t"
    "fmul  v13.8h,  v13.8h, v0.8h    \n\t"
    "fmul  v14.8h,  v14.8h, v0.8h    \n\t"
    "fmul  v15.8h,  v15.8h, v0.8h    \n\t"
    "fmul  v16.8h,  v16.8h, v0.8h    \n\t"
    "fmul  v17.8h,  v17.8h, v0.8h    \n\t"
    "fmul  v18.8h,  v18.8h, v0.8h    \n\t"
    "fmul  v19.8h,  v19.8h, v0.8h    \n\t"
    "fmul  v20.8h,  v20.8h, v0.8h    \n\t"
    "fmul  v21.8h,  v21.8h, v0.8h    \n\t"
    "fmul  v22.8h,  v22.8h, v0.8h    \n\t"
    "fmul  v23.8h,  v23.8h, v0.8h    \n\t"
    "fmul  v24.8h,  v24.8h, v0.8h    \n\t"
    "fmul  v25.8h,  v25.8h, v0.8h    \n\t"
    "fmul  v26.8h,  v26.8h, v0.8h    \n\t"
    "fmul  v27.8h,  v27.8h, v0.8h    \n\t"
    "fmul  v28.8h,  v28.8h, v0.8h    \n\t"
    "fmul  v29.8h,  v29.8h, v0.8h    \n\t"
    "fmul  v30.8h,  v30.8h, v0.8h    \n\t"
    "fmul  v31.8h,  v31.8h, v0.8h    \n\t"
    "LH_UNIT_ALPHA%=:                \n\t"
    "mov   x9,      x5               \n\t" // loading C address
    "fcmp  h1,    #0.0               \n\t" // test beta is 0.0 or not
    "b.eq  LH_BETA_ZERO_R_8_13%=            \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "ldr   q3,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "ldr   q5,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "ldr   q7,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v8.8h,   v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v9.8h,   v3.8h,  v1.h[0]  \n\t"
    "fmla  v10.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v11.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v12.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v13.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_8_13%=:          \n\t"
    "str   q8,      [x5]             \n\t"
    "str   q9,      [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q10,     [x5]             \n\t"
    "str   q11,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q12,     [x5]             \n\t"
    "str   q13,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LH_BETA_ZERO_R_14_19%=     \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "ldr   q3,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "ldr   q5,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "ldr   q7,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v14.8h,  v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v15.8h,  v3.8h,  v1.h[0]  \n\t"
    "fmla  v16.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v17.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v18.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v19.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_14_19%=:         \n\t"
    "str   q14,     [x5]             \n\t"
    "str   q15,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q16,     [x5]             \n\t"
    "str   q17,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q18,     [x5]             \n\t"
    "str   q19,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LH_BETA_ZERO_R_20_25%=    \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "ldr   q3,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "ldr   q5,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "ldr   q7,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v20.8h,  v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v21.8h,  v3.8h,  v1.h[0]  \n\t"
    "fmla  v22.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v23.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v24.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v25.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_20_25%=:         \n\t"
    "str   q20,     [x5]             \n\t"
    "str   q21,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q22,     [x5]             \n\t"
    "str   q23,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q24,     [x5]             \n\t"
    "str   q25,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LH_BETA_ZERO_R_26_31%=    \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining six FP register
    "ldr   q3,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q4,      [x9]             \n\t"
    "ldr   q5,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "ldr   q6,      [x9]             \n\t"
    "ldr   q7,      [x9, #16*1]      \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fmla  v26.8h,  v2.8h,  v1.h[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v27.8h,  v3.8h,  v1.h[0]  \n\t"
    "fmla  v28.8h,  v4.8h,  v1.h[0]  \n\t"
    "fmla  v29.8h,  v5.8h,  v1.h[0]  \n\t"
    "fmla  v30.8h,  v6.8h,  v1.h[0]  \n\t"
    "fmla  v31.8h,  v7.8h,  v1.h[0]  \n\t"
    "LH_BETA_ZERO_R_26_31%=:         \n\t"
    "str   q26,     [x5]             \n\t"
    "str   q27,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q28,     [x5]             \n\t"
    "str   q29,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q30,     [x5]             \n\t"
    "str   q31,     [x5, #16*1]      \n\t"
    "add   x5,      x5,     x6       \n\t"
    :// output operands(none)
    :[a]        "m"      (a),
     [b]        "m"      (b),
     [c]        "m"      (c),
     [rs_c]     "m"   (rs_c),
     [k_mker]   "m" (k_mker),
     [k_left]   "m" (k_left),
     [alpha]    "m"  (alpha),
     [beta]     "m"   (beta) //[ct]       "m"     (ct)
    :"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
     "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
     "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
     "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );

    return;
}

 

void bli_hgemm_armv8a_asm_sh12x8r(
        dim_t  m,
        dim_t  n,
        dim_t  k,
        const void*   alpha_,
        const void*       a,
        const void*       b,
        const void*    beta_,
              void*       c,
              inc_t   rs_c0,
              inc_t   cs_c0,
        const auxinfo_t* data,
        const cntx_t* cntx
        )
{

    uint64_t  k_mker = k / 4;
    uint64_t  k_left = k % 4;

    uint64_t  rs_c   = rs_c0;
    uint64_t  cs_c   = cs_c0;

    _Float16 alpha_cast = 1.0;
    _Float16 beta_cast = 1.0;
    _Float16* alpha = &alpha_cast;
    _Float16* beta = &beta_cast;

    //printf("we are using sh12x8 kernel\n");
    const uint64_t ldc = (rs_c == 1)? cs_c : rs_c;
    __asm__ volatile
    (
    "ldr   x0,  %[a]               \n\t"  // Address of matrix A
    "ldr   x1,  %[b]               \n\t"  // Address of matrix B
    "                              \n\t"
    "mov   x2,  #12                \n\t"  // Column skip for A
    "mov   x3,  #8                 \n\t"  // Row_skip of B

    "ldr   x5,  %[c]              \n\t"  // Address of matrix C
    "ldr   x6,  %[rs_c]           \n\t"  // Row_skip of C, (Column skip is 1)
    "                             \n\t"
    "                             \n\t"  //Multiply some address ships by the sizeof(half)
    "lsl   x2, x2, #1             \n\t"  // cs_a
    "lsl   x3, x3, #1             \n\t"  // rs_b
    "lsl   x6, x6, #1             \n\t"  // rs_c
    "                             \n\t"
    //" cmp   %w[ct], wzr            \n\t"
    "mov   x9,     x5             \n\t"
    "ldr   x4, %[k_mker]          \n\t"
    "ldr   x8, %[k_left]          \n\t"

    "                             \n\t"
    "cmp   x4,    #0              \n\t"  // No-microkernel early return is a must to avoid out-of boundry read
    "b.eq  LDCLEAR_CCOLS%=        \n\t"
    "                             \n\t"
    "ldr   q0,   [x0, #16*0]     \n\t" // load A
    "ldr   q1,   [x0, #16*1]     \n\t"
    "add   x0,    x0,   x2        \n\t"
    "ldr   q3,   [x1, #16*0]     \n\t" // load B
    "add   x1,    x1,   x3        \n\t"
    "LDCLEAR_CCOLS%=:             \n\t"
    "                             \n\t" // clean vector
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 1,2/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v8.4s,  wzr            \n\t"
    "dup   v9.4s,  wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 3,4/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v10.4s, wzr            \n\t"
    "dup   v11.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 5,6/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v12.4s, wzr            \n\t"
    "dup   v13.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 7,8/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v14.4s, wzr            \n\t"
    "dup   v15.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 9,10/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v16.4s, wzr            \n\t"
    "dup   v17.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 11,12/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v18.4s, wzr            \n\t"
    "dup   v19.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 13,14/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v20.4s, wzr            \n\t"
    "dup   v21.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 15,16/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v22.4s, wzr            \n\t"
    "dup   v23.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 17,18/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v24.4s, wzr            \n\t"
    "dup   v25.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 19,20/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v26.4s, wzr            \n\t"
    "dup   v27.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 21,22/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v28.4s, wzr            \n\t"
    "dup   v29.4s, wzr            \n\t"
    "prfm  PLDL1KEEP, [x9]        \n\t" // prefetch C 23,24/24
    "prfm  PLDL1KEEP, [x9, #16]   \n\t"
    "add   x9,     x9,     x6     \n\t"
    "dup   v30.4s, wzr            \n\t"
    "dup   v31.4s, wzr            \n\t"
    "                             \n\t"
    //"cmp   x4,    #0            \n\t"  // No-microkernel early return is a must to avoid out-of boundry read
    "b.eq  LSHK_LEFT_LOOP%=       \n\t" // Need to be noted here, C is row-major, A is col-major, B is row-major
    "                             \n\t" //  if x4 (the loop_meker) is 0, then we will skip the loop
    "LSHK_MKER_LOOP%=:            \n\t" //"                              \n\t"  // First Loop Unrolling
    "fmlal   v8.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal2  v9.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal   v10.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal2  v11.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal   v12.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal2  v13.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal   v14.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal2  v15.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal   v16.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal2  v17.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal   v18.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal2  v19.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal   v20.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal2  v21.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal   v22.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal2  v23.4s,  v3.4h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]      \n\t"  // Load A1 for the next kernel excute
    "fmlal   v24.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal2  v25.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal   v26.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal2  v27.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal   v28.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal2  v29.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal   v30.4s,  v3.4h, v1.h[3]   \n\t"
    "fmlal2  v31.4s,  v3.4h, v1.h[3]   \n\t"
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "                                \n\t"  // Second Loop Unrolling
    "fmlal   v8.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal2  v9.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal   v10.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal2  v11.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal   v12.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal2  v13.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal   v14.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal2  v15.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal   v16.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal2  v17.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal   v18.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal2  v19.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal   v20.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal2  v21.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal   v22.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal2  v23.4s,  v3.4h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]      \n\t"  // Load A1 for the next kernel excute
    "fmlal   v24.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal2  v25.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal   v26.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal2  v27.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal   v28.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal2  v29.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal   v30.4s,  v3.4h, v1.h[3]   \n\t"
    "fmlal2  v31.4s,  v3.4h, v1.h[3]   \n\t"
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"  // Third Loop Unrolling
    "fmlal   v8.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal2  v9.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal   v10.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal2  v11.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal   v12.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal2  v13.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal   v14.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal2  v15.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal   v16.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal2  v17.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal   v18.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal2  v19.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal   v20.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal2  v21.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal   v22.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal2  v23.4s,  v3.4h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]      \n\t"  // Load A1 for the next kernel excute
    "fmlal   v24.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal2  v25.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal   v26.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal2  v27.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal   v28.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal2  v29.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal   v30.4s,  v3.4h, v1.h[3]   \n\t"
    "fmlal2  v31.4s,  v3.4h, v1.h[3]   \n\t"
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "cmp   x4,      #1               \n\t"
    "b.eq  LH_LAST_UNROLL_LOOP%=     \n\t"
    "                                \n\t"  // Fourth Loop Unrolling
    "fmlal   v8.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal2  v9.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal   v10.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal2  v11.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal   v12.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal2  v13.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal   v14.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal2  v15.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal   v16.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal2  v17.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal   v18.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal2  v19.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal   v20.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal2  v21.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal   v22.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal2  v23.4s,  v3.4h, v0.h[7]   \n\t"
    "ldr   q0,     [x0, #16 * 0]      \n\t"  // Load A1 for the next kernel excute
    "fmlal   v24.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal2  v25.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal   v26.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal2  v27.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal   v28.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal2  v29.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal   v30.4s,  v3.4h, v1.h[3]   \n\t"
    "fmlal2  v31.4s,  v3.4h, v1.h[3]   \n\t"
    "ldr   q1,     [x0, #16 * 1]     \n\t"  // Load A1 for the next kernel excute
    "ldr   q3,     [x1,      #0]     \n\t"  // Load B for the next kernel excute
    "add   x0,      x0,       x2     \n\t"
    "add   x1,      x1,       x3     \n\t"
    "                                \n\t"
    "sub   x4,      x4,       #1     \n\t"
    "cmp   x4,      #0               \n\t"
    "b.ne  LSHK_MKER_LOOP%=          \n\t"
    "                                \n\t"
    "LH_LAST_UNROLL_LOOP%=:          \n\t"
    "fmlal   v8.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal2  v9.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal   v10.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal2  v11.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal   v12.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal2  v13.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal   v14.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal2  v15.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal   v16.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal2  v17.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal   v18.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal2  v19.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal   v20.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal2  v21.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal   v22.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal2  v23.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal   v24.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal2  v25.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal   v26.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal2  v27.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal   v28.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal2  v29.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal   v30.4s,  v3.4h, v1.h[3]   \n\t"
    "fmlal2  v31.4s,  v3.4h, v1.h[3]   \n\t"
    "                                  \n\t"  // end of unroll part
    "LSHK_LEFT_LOOP%=:                 \n\t"  // Label for if MK_MKER to ZREO
    "                                \n\t"
    "cmp   x8,              #0       \n\t"
    "b.eq  LSHWRITE_MEM_PREP%=       \n\t"
    "ldr   q0,   [x0, #16*0]         \n\t" // load A0, A1, A2
    "ldr   q1,   [x0, #16*1]         \n\t"
    "add   x0,    x0,   x2           \n\t"
    "ldr   q3,   [x1, #16*0]         \n\t" // load B
    "add   x1,    x1,   x3           \n\t"
    "sub   x8,    x8,   #1           \n\t"
    "                                \n\t"
    "fmlal   v8.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal2  v9.4s,   v3.4h, v0.h[0]   \n\t"
    "fmlal   v10.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal2  v11.4s,  v3.4h, v0.h[1]   \n\t"
    "fmlal   v12.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal2  v13.4s,  v3.4h, v0.h[2]   \n\t"
    "fmlal   v14.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal2  v15.4s,  v3.4h, v0.h[3]   \n\t"
    "fmlal   v16.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal2  v17.4s,  v3.4h, v0.h[4]   \n\t"
    "fmlal   v18.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal2  v19.4s,  v3.4h, v0.h[5]   \n\t"
    "fmlal   v20.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal2  v21.4s,  v3.4h, v0.h[6]   \n\t"
    "fmlal   v22.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal2  v23.4s,  v3.4h, v0.h[7]   \n\t"
    "fmlal   v24.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal2  v25.4s,  v3.4h, v1.h[0]   \n\t"
    "fmlal   v26.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal2  v27.4s,  v3.4h, v1.h[1]   \n\t"
    "fmlal   v28.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal2  v29.4s,  v3.4h, v1.h[2]   \n\t"
    "fmlal   v30.4s,  v3.4h, v1.h[3]   \n\t"
    "fmlal2  v31.4s,  v3.4h, v1.h[3]   \n\t"
    "b    LSHK_LEFT_LOOP%=           \n\t"
    "LSHWRITE_MEM_PREP%=:            \n\t"
    "                                \n\t"
    "                                \n\t" // // Scale and write to memory.
    "ldr   x4,       %[alpha]        \n\t" // load alpha & beta address
    "ldr   x8,       %[beta]         \n\t"
    "ld1r  {v2.8h}, [x4]             \n\t" // load alpha & beta
    "ld1r  {v3.8h}, [x8]             \n\t"
    "fcvtl v0.4s,   v2.4h            \n\t"
    "fcvtl v1.4s,   v3.4h            \n\t"
    "fmov  h2,      #1.0             \n\t"
    "fcmp  h0,      h2               \n\t" // alpha is 1.0 or not
    "b.eq  LSH_UNIT_ALPHA%=          \n\t"
    "fmul  v8.4s,   v8.4s,  v0.4s    \n\t"
    "fmul  v9.4s,   v9.4s,  v0.4s    \n\t"
    "fmul  v10.4s,  v10.4s, v0.4s    \n\t"
    "fmul  v11.4s,  v11.4s, v0.4s    \n\t"
    "fmul  v12.4s,  v12.4s, v0.4s    \n\t"
    "fmul  v13.4s,  v13.4s, v0.4s    \n\t"
    "fmul  v14.4s,  v14.4s, v0.4s    \n\t"
    "fmul  v15.4s,  v15.4s, v0.4s    \n\t"
    "fmul  v16.4s,  v16.4s, v0.4s    \n\t"
    "fmul  v17.4s,  v17.4s, v0.4s    \n\t"
    "fmul  v18.4s,  v18.4s, v0.4s    \n\t"
    "fmul  v19.4s,  v19.4s, v0.4s    \n\t"
    "fmul  v20.4s,  v20.4s, v0.4s    \n\t"
    "fmul  v21.4s,  v21.4s, v0.4s    \n\t"
    "fmul  v22.4s,  v22.4s, v0.4s    \n\t"
    "fmul  v23.4s,  v23.4s, v0.4s    \n\t"
    "fmul  v24.4s,  v24.4s, v0.4s    \n\t"
    "fmul  v25.4s,  v25.4s, v0.4s    \n\t"
    "fmul  v26.4s,  v26.4s, v0.4s    \n\t"
    "fmul  v27.4s,  v27.4s, v0.4s    \n\t"
    "fmul  v28.4s,  v28.4s, v0.4s    \n\t"
    "fmul  v29.4s,  v29.4s, v0.4s    \n\t"
    "fmul  v30.4s,  v30.4s, v0.4s    \n\t"
    "fmul  v31.4s,  v31.4s, v0.4s    \n\t"
    "LSH_UNIT_ALPHA%=:               \n\t"
    "mov   x9,      x5               \n\t" // loading C address
    "fcmp  s1,    #0.0               \n\t" // test beta is 0.0 or not
    "b.eq  LSH_BETA_ZERO_R_8_11%=    \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining four FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fcvtl  v4.4s,   v2.4h           \n\t"
    "fcvtl2 v5.4s,   v2.8h           \n\t"
    "fcvtl  v6.4s,   v3.4h           \n\t"
    "fcvtl2 v7.4s,   v3.8h           \n\t"
    "fmla  v8.4s,   v4.4s,  v1.s[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v9.4s,   v5.4s,  v1.s[0]  \n\t"
    "fmla  v10.4s,  v6.4s,  v1.s[0]  \n\t"
    "fmla  v11.4s,  v7.4s,  v1.s[0]  \n\t"
    "LSH_BETA_ZERO_R_8_11%=:         \n\t"
    "fcvtn  v4.4h,  v8.4s            \n\t"
    "fcvtn2 v4.8h,  v9.4s            \n\t"
    "fcvtn  v5.4h,  v10.4s           \n\t"
    "fcvtn2 v5.8h,  v11.4s           \n\t"
    "str   q4,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q5,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LSH_BETA_ZERO_R_12_15%=   \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining four FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fcvtl  v4.4s,   v2.4h           \n\t"
    "fcvtl2 v5.4s,   v2.8h           \n\t"
    "fcvtl  v6.4s,   v3.4h           \n\t"
    "fcvtl2 v7.4s,   v3.8h           \n\t"
    "fmla  v12.4s,  v4.4s,  v1.s[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v13.4s,  v5.4s,  v1.s[0]  \n\t"
    "fmla  v14.4s,  v6.4s,  v1.s[0]  \n\t"
    "fmla  v15.4s,  v7.4s,  v1.s[0]  \n\t"
    "LSH_BETA_ZERO_R_12_15%=:        \n\t"
    "fcvtn  v2.4h,  v12.4s           \n\t"
    "fcvtn2 v2.8h,  v13.4s           \n\t"
    "fcvtn  v3.4h,  v14.4s           \n\t"
    "fcvtn2 v3.8h,  v15.4s           \n\t"
    "str   q2,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q3,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LSH_BETA_ZERO_R_16_19%=   \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining four FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fcvtl  v4.4s,   v2.4h           \n\t"
    "fcvtl2 v5.4s,   v2.8h           \n\t"
    "fcvtl  v6.4s,   v3.4h           \n\t"
    "fcvtl2 v7.4s,   v3.8h           \n\t"
    "fmla  v16.4s,  v4.4s,  v1.s[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v17.4s,  v5.4s,  v1.s[0]  \n\t"
    "fmla  v18.4s,  v6.4s,  v1.s[0]  \n\t"
    "fmla  v19.4s,  v7.4s,  v1.s[0]  \n\t"
    "LSH_BETA_ZERO_R_16_19%=:        \n\t"
    "fcvtn  v2.4h,  v16.4s           \n\t"
    "fcvtn2 v2.8h,  v17.4s           \n\t"
    "fcvtn  v3.4h,  v18.4s           \n\t"
    "fcvtn2 v3.8h,  v19.4s           \n\t"
    "str   q2,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q3,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LSH_BETA_ZERO_R_20_23%=   \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining four FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fcvtl  v4.4s,   v2.4h           \n\t"
    "fcvtl2 v5.4s,   v2.8h           \n\t"
    "fcvtl  v6.4s,   v3.4h           \n\t"
    "fcvtl2 v7.4s,   v3.8h           \n\t"
    "fmla  v20.4s,  v4.4s,  v1.s[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v21.4s,  v5.4s,  v1.s[0]  \n\t"
    "fmla  v22.4s,  v6.4s,  v1.s[0]  \n\t"
    "fmla  v23.4s,  v7.4s,  v1.s[0]  \n\t"
    "LSH_BETA_ZERO_R_20_23%=:        \n\t"
    "fcvtn  v2.4h,  v20.4s           \n\t"
    "fcvtn2 v2.8h,  v21.4s           \n\t"
    "fcvtn  v3.4h,  v22.4s           \n\t"
    "fcvtn2 v3.8h,  v23.4s           \n\t"
    "str   q2,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q3,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LSH_BETA_ZERO_R_24_27%=   \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining four FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fcvtl  v4.4s,   v2.4h           \n\t"
    "fcvtl2 v5.4s,   v2.8h           \n\t"
    "fcvtl  v6.4s,   v3.4h           \n\t"
    "fcvtl2 v7.4s,   v3.8h           \n\t"
    "fmla  v24.4s,  v4.4s,  v1.s[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v25.4s,  v5.4s,  v1.s[0]  \n\t"
    "fmla  v26.4s,  v6.4s,  v1.s[0]  \n\t"
    "fmla  v27.4s,  v7.4s,  v1.s[0]  \n\t"
    "LSH_BETA_ZERO_R_24_27%=:        \n\t"
    "fcvtn  v2.4h,  v24.4s           \n\t"
    "fcvtn2 v2.8h,  v25.4s           \n\t"
    "fcvtn  v3.4h,  v26.4s           \n\t"
    "fcvtn2 v3.8h,  v27.4s           \n\t"
    "str   q2,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q3,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "b.eq  LSH_BETA_ZERO_R_28_31%=   \n\t"
    "ldr   q2,      [x9]             \n\t" // load C to the remaining four FP register
    "add   x9,      x9,     x6       \n\t"
    "ldr   q3,      [x9]             \n\t"
    "add   x9,      x9,     x6       \n\t"
    "fcvtl  v4.4s,   v2.4h           \n\t"
    "fcvtl2 v5.4s,   v2.8h           \n\t"
    "fcvtl  v6.4s,   v3.4h           \n\t"
    "fcvtl2 v7.4s,   v3.8h           \n\t"
    "fmla  v28.4s,  v4.4s,  v1.s[0]  \n\t" // SCALE C WITH BETA AND ADD IT TO A*B
    "fmla  v29.4s,  v5.4s,  v1.s[0]  \n\t"
    "fmla  v30.4s,  v6.4s,  v1.s[0]  \n\t"
    "fmla  v31.4s,  v7.4s,  v1.s[0]  \n\t"
    "LSH_BETA_ZERO_R_28_31%=:        \n\t"
    "fcvtn  v2.4h,  v28.4s           \n\t"
    "fcvtn2 v2.8h,  v29.4s           \n\t"
    "fcvtn  v3.4h,  v30.4s           \n\t"
    "fcvtn2 v3.8h,  v31.4s           \n\t"
    "str   q2,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    "str   q3,      [x5]             \n\t"
    "add   x5,      x5,     x6       \n\t"
    :
    :[a]        "m"      (a),
     [b]        "m"      (b),
     [c]        "m"      (c),
     [rs_c]     "m"   (rs_c),
     [k_mker]   "m" (k_mker),
     [k_left]   "m" (k_left),
     [alpha]    "m"  (alpha),
     [beta]     "m"   (beta) //[ct]       "m"     (ct)
    :"x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
     "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
     "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
     "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
    );
    return;
}


