/*

   BLIS    
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas at Austin nor the names
      of its contributors may be used to endorse or promote products
      derived derived derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


*/

#include "blis.h"

/*
   o 4x4 Single precision micro-kernel fully functional.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 7.3 GFLOPS @ 1.1 GHz. 

   December 2014.
*/
void bli_sgemm_opt_4x4(
                        dim_t              k,
                        float*    restrict alpha,
                        float*    restrict a,
                        float*    restrict b,
                        float*    restrict beta,
                        float*    restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	void* a_next = bli_auxinfo_next_a( data );
	void* b_next = bli_auxinfo_next_b( data );

	dim_t k_iter = k / 4;
	dim_t k_left = k % 4;

__asm__ volatile
(
"                                            \n\t"
"                                            \n\t"
" ldr x0,%[aaddr]                            \n\t" // Load address of A.
" ldr x1,%[baddr]                            \n\t" // Load address of B.
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
" mov x4,#1                                  \n\t" // Init loop counter (i=0).
"                                            \n\t"
" ldr x16,%[a_next]                          \n\t" // Pointer to next block of A.
" ldr x17,%[b_next]                          \n\t" // Pointer to next pointer of B.
"                                            \n\t"
" ldr x5,%[k_iter]                           \n\t" // Number of unrolled iterations (k_iter).
" ldr x6,%[k_left]                           \n\t" // Number of remaining iterations (k_left).
"                                            \n\t" 
" ldr x7,%[alpha]                            \n\t" // Alpha address.      
" ldr x8,%[beta]                             \n\t" // Beta address.     
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c.
" lsl x10,x9,#2                              \n\t" // cs_c * sizeof(float) -- AUX.
" lsl x11,x9,#3                              \n\t" // 2 * cs_c * sizeof(float) -- AUX.
" lsl x12,x9,#4                              \n\t" // 3 * cs_c * sizeof(float) -- AUX.
"                                            \n\t" 
" ldr x13,%[rs_c]                            \n\t" // Load rs_c.
" lsl x14,x13,#2                             \n\t" // rs_c * sizeof(float).
"                                            \n\t" 
" ldp q0,q1,[x0,0]                           \n\t" // Preload columns a,a+1 into two quads.
" ldp q4,q5,[x1,0]                           \n\t" // Preload rows    b,b+1 into two quads.
"                                            \n\t"
" prfm pldl1keep,[x2,0]                      \n\t" // Prefetch c.
" prfm pldl1keep,[x2,x10]                    \n\t" // Prefetch c.
" prfm pldl1keep,[x2,x11]                    \n\t" // Prefetch c.
" prfm pldl1keep,[x2,x12]                    \n\t" // Prefetch c.
"                                            \n\t"
"                                            \n\t" // Vectors for result columns.
" movi v8.4s,#0                              \n\t" // Vector for result column 0.
" movi v9.4s,#0                              \n\t" // Vector for result column 1.
" movi v10.4s,#0                             \n\t" // Vector for result column 2.
" movi v11.4s,#0                             \n\t" // Vector for result column 3.
"                                            \n\t"
"                                            \n\t" // Replicating accum. vectors for unrolling.
" movi v12.4s,#0                             \n\t" // Vector 1 for accummulating column 0.
" movi v13.4s,#0                             \n\t" // Vector 1 for accummulating column 1.
" movi v14.4s,#0                             \n\t" // Vector 1 for accummulating column 2.
" movi v15.4s,#0                             \n\t" // Vector 1 for accummulating column 3.
"                                            \n\t"
" movi v16.4s,#0                             \n\t" // Vector 2 for accummulating column 0.
" movi v17.4s,#0                             \n\t" // Vector 2 for accummulating column 1.
" movi v18.4s,#0                             \n\t" // Vector 2 for accummulating column 2.
" movi v19.4s,#0                             \n\t" // Vector 2 for accummulating column 3.
"                                            \n\t"
" movi v20.4s,#0                             \n\t" // Vector 3 for accummulating column 0.
" movi v21.4s,#0                             \n\t" // Vector 3 for accummulating column 1.
" movi v22.4s,#0                             \n\t" // Vector 3 for accummulating column 2.
" movi v23.4s,#0                             \n\t" // Vector 3 for accummulating column 3.
"                                            \n\t"
" movi v24.4s,#0                             \n\t" // Vector 4 for accummulating column 0.
" movi v25.4s,#0                             \n\t" // Vector 4 for accummulating column 1.
" movi v26.4s,#0                             \n\t" // Vector 4 for accummulating column 2.
" movi v27.4s,#0                             \n\t" // Vector 4 for accummulating column 3.
"                                            \n\t"
" ld1r {v31.4s},[x8]                         \n\t" // Load beta into quad.
"                                            \n\t"
" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
" beq .SCONSIDERKLEFT                        \n\t"
"                                            \n\t"
" cmp x5,1                                   \n\t" // If there is just one k_iter, jump to that one. 
" beq .SLASTITER                             \n\t" // (as loop is do-while-like).
"                                            \n\t"
" .SLOOPKITER:                               \n\t" // Body of the k_iter loop.
"                                            \n\t"
" prfm pldl1keep,[x0,#1024]                  \n\t" // Prefetch.
" prfm pldl1keep,[x1,#1024]                  \n\t" // Prefetch.
"                                            \n\t"
" fmla v12.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v13.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" ldp q6,q7,[x1,32]                          \n\t" // Load rows b+2,b+3 into quads.
"                                            \n\t"
" fmla v14.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v15.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" ldp q2,q3,[x0,32]                          \n\t" // Load columns a+2,a+3 into quads.
"                                            \n\t"
" fmla v16.4s,v1.4s,v5.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v5.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v18.4s,v1.4s,v5.s[2]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v5.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" add x0,x0,64                               \n\t" // Update a_ptr.
" add x1,x1,64                               \n\t" // Update b_ptr.
"                                            \n\t"
" fmla v20.4s,v2.4s,v6.s[0]                  \n\t" // Accummulate.
" fmla v21.4s,v2.4s,v6.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" ldp q0,q1,[x0]                             \n\t" // Load columns a,a+1 into quads (next iteration).
"                                            \n\t"
" fmla v22.4s,v2.4s,v6.s[2]                  \n\t" // Accummulate.
" fmla v23.4s,v2.4s,v6.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" ldp q4,q5,[x1]                             \n\t" // Load rows b,b+1 into quads (next iteration).
"                                            \n\t"
" fmla v24.4s,v3.4s,v7.s[0]                  \n\t" // Accummulate.
" fmla v25.4s,v3.4s,v7.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" prfm pldl1keep,[x0,#64]                    \n\t" // Prefetch.
" prfm pldl1keep,[x1,#64]                    \n\t" // Prefetch.
"                                            \n\t"
" fmla v26.4s,v3.4s,v7.s[2]                  \n\t" // Accummulate.
" fmla v27.4s,v3.4s,v7.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" sub x5,x5,1                                \n\t" // i-=1.
" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
" bne .SLOOPKITER                            \n\t"
"                                            \n\t" 
//" prfm pldl1keep,[x0,#1024]                \n\t"
//" prfm pldl1keep,[x1,#1024]                \n\t"
"                                            \n\t" 
" .SLASTITER:                                \n\t" // Last iteration of k_iter loop.
"                                            \n\t" 
" fmla v12.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v13.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" ldp q6,q7,[x1,32]                          \n\t" // Load rows b+2,b+3 into quads.
"                                            \n\t"
" fmla v14.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v15.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" ldp q2,q3,[x0,32]                          \n\t" // Load columns a+2,a+3 into quads.
"                                            \n\t"
" fmla v16.4s,v1.4s,v5.s[0]                  \n\t" // Accummulate.
" fmla v17.4s,v1.4s,v5.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" ld1r {v30.4s},[x7]                         \n\t" // Load alpha.
"                                            \n\t"
" fmla v18.4s,v1.4s,v5.s[2]                  \n\t" // Accummulate.
" fmla v19.4s,v1.4s,v5.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v20.4s,v2.4s,v6.s[0]                  \n\t" // Accummulate.
" fmla v21.4s,v2.4s,v6.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v22.4s,v2.4s,v6.s[2]                  \n\t" // Accummulate.
" fmla v23.4s,v2.4s,v6.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v24.4s,v3.4s,v7.s[0]                  \n\t" // Accummulate.
" fmla v25.4s,v3.4s,v7.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v26.4s,v3.4s,v7.s[2]                  \n\t" // Accummulate.
" fmla v27.4s,v3.4s,v7.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
//" ld1 {v8.4s},[x2],x10                       \n\t" // Load c    into quad and increment by cs_c
//" ld1 {v9.4s},[x2],x10                       \n\t" // Load c+4  into quad and increment by cs_c
//" ld1 {v10.4s},[x2],x10                      \n\t" // Load c+8  into quad and increment by cs_c
//" ld1 {v11.4s},[x2],x10                      \n\t" // Load c+16 into quad and increment by cs_c
"                                            \n\t"
" fadd v12.4s,v12.4s,v16.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v13.4s,v13.4s,v17.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v14.4s,v14.4s,v18.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v15.4s,v15.4s,v19.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v12.4s,v12.4s,v20.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v13.4s,v13.4s,v21.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v14.4s,v14.4s,v22.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v15.4s,v15.4s,v23.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v12.4s,v12.4s,v24.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v13.4s,v13.4s,v25.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v14.4s,v14.4s,v26.4s                  \n\t" // Final accummulate of temporal accum. vectors.
" fadd v15.4s,v15.4s,v27.4s                  \n\t" // Final accummulate of temporal accum. vectors.
"                                            \n\t"
" add x0,x0,64                               \n\t" // Update a_ptr.
" add x1,x1,64                               \n\t" // Update b_ptr.
"                                            \n\t"
" .SCONSIDERKLEFT:                           \n\t" 
" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
" beq .SPOSTACCUM                            \n\t" // else, we enter the k_left loop.
"                                            \n\t"
" .SLOOPKLEFT:                               \n\t" // Body of the left iterations
"                                            \n\t"
" prfm pldl1keep,[x0,#1024]                  \n\t" // Prefetch.
" prfm pldl1keep,[x1,#1024]                  \n\t" // Prefetch.
"                                            \n\t"
" ldr q0,[x0]                                \n\t" // Load a into quad (next iteration).
" ldr q4,[x1]                                \n\t" // Load b into quad (next iteration).
"                                            \n\t"
" add x0,x0,16                               \n\t" // Update a_ptr.
" add x1,x1,16                               \n\t" // Update b_ptr.
"                                            \n\t"
" sub x6,x6,1                                \n\t" // i = i-1.
"                                            \n\t"
" fmla v12.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
" fmla v13.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
"                                            \n\t"
" fmla v14.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
" fmla v15.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
"                                            \n\t"
" cmp x6,0                                   \n\t" // Iterate again.
" bne .SLOOPKLEFT                            \n\t" // if i!=0.
"                                            \n\t"
" ld1r {v30.4s},[x7]                         \n\t" // Load alpha.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
" .SPOSTACCUM:                               \n\t"
" cmp x13,#1                                 \n\t" // If rs_c != 1 (column-major)
" bne .SGENSTORED                            \n\t"
"                                            \n\t"
"                                            \n\t"
" .SCOLSTORED:                               \n\t" // C is column-major.
"                                            \n\t"
" fcmp s31,#0.0                              \n\t"
" beq .BETAZEROCOLSTORED                     \n\t" // Taking care of the beta==0 case.
"                                            \n\t"
"                                            \n\t" // If beta!=0, then we can read from C.
" ld1 {v8.4s},[x2],x10                       \n\t" // Load c    into quad and increment by cs_c.
" ld1 {v9.4s},[x2],x10                       \n\t" // Load c+4  into quad and increment by cs_c.
" ld1 {v10.4s},[x2],x10                      \n\t" // Load c+8  into quad and increment by cs_c.
" ld1 {v11.4s},[x2],x10                      \n\t" // Load c+16 into quad and increment by cs_c.
"                                            \n\t"
" prfm pldl1keep,[x16,0]                     \n\t" // Prefetch.
" prfm pldl1keep,[x17,0]                     \n\t" // Prefetch.
"                                            \n\t"
"                                            \n\t"
" fmul v8.4s,v8.4s,v31.s[0]                  \n\t" // Scale by beta.
" fmul v9.4s,v9.4s,v31.s[0]                  \n\t" // Scale by beta.
" fmul v10.4s,v10.4s,v31.s[0]                \n\t" // Scale by beta.
" fmul v11.4s,v11.4s,v31.s[0]                \n\t" // Scale by beta.
"                                            \n\t"
" .BETAZEROCOLSTORED:                        \n\t" // If beta==0, we won't read from C (nor scale).
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
" fmla v8.4s,v12.4s,v30.s[0]                 \n\t" // Scale by alpha
" fmla v9.4s,v13.4s,v30.s[0]                 \n\t" // Scale by alpha
" fmla v10.4s,v14.4s,v30.s[0]                \n\t" // Scale by alpha
" fmla v11.4s,v15.4s,v30.s[0]                \n\t" // Scale by alpha
"                                            \n\t"
" st1 {v8.4s},[x2],x10                       \n\t" // Store quad into c    and increment by cs_c
" st1 {v9.4s},[x2],x10                       \n\t" // Store quad into c+4  and increment by cs_c
" st1 {v10.4s},[x2],x10                      \n\t" // Store quad into c+8  and increment by cs_c
" st1 {v11.4s},[x2],x10                      \n\t" // Store quad into c+16 and increment by cs_c
"                                            \n\t"
" b .SEND                                    \n\t" // Done (TODO: this obviously needs to be moved down to remove jump).
"                                            \n\t"
"                                            \n\t"
" .SGENSTORED:                               \n\t" // C is general-stride stored.
"                                            \n\t"
" fcmp s31,#0.0                              \n\t"
" beq .BETAZEROGENSTORED                     \n\t"
"                                            \n\t"
"                                            \n\t" // If beta!=0, then we can read from C.
"                                            \n\t" // TODO: this was done fast. Rearrange to remove so many address reloads.
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
" ld1 {v8.s}[0],[x2],x14                     \n\t" // Load c00  into quad and increment by rs_c.
" ld1 {v8.s}[1],[x2],x14                     \n\t" // Load c01  into quad and increment by rs_c.
" ld1 {v8.s}[2],[x2],x14                     \n\t" // Load c02  into quad and increment by rs_c.
" ld1 {v8.s}[3],[x2],x14                     \n\t" // Load c03  into quad and increment by rs_c.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
" add x2,x2,x10                              \n\t" // c += cs_c.
"                                            \n\t"
" ld1 {v9.s}[0],[x2],x14                     \n\t" // Load c10  into quad and increment by rs_c.
" ld1 {v9.s}[1],[x2],x14                     \n\t" // Load c11  into quad and increment by rs_c.
" ld1 {v9.s}[2],[x2],x14                     \n\t" // Load c12  into quad and increment by rs_c.
" ld1 {v9.s}[3],[x2],x14                     \n\t" // Load c13  into quad and increment by rs_c.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
" add x2,x2,x10                              \n\t" // c += cs_c.
" add x2,x2,x10                              \n\t" // c += cs_c.
"                                            \n\t"
" ld1 {v10.s}[0],[x2],x14                    \n\t" // Load c10  into quad and increment by rs_c.
" ld1 {v10.s}[1],[x2],x14                    \n\t" // Load c11  into quad and increment by rs_c.
" ld1 {v10.s}[2],[x2],x14                    \n\t" // Load c12  into quad and increment by rs_c.
" ld1 {v10.s}[3],[x2],x14                    \n\t" // Load c13  into quad and increment by rs_c.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
" add x2,x2,x10                              \n\t" // c += cs_c.
" add x2,x2,x10                              \n\t" // c += cs_c.
" add x2,x2,x10                              \n\t" // c += cs_c.
"                                            \n\t"
" ld1 {v11.s}[0],[x2],x14                    \n\t" // Load c10  into quad and increment by rs_c.
" ld1 {v11.s}[1],[x2],x14                    \n\t" // Load c11  into quad and increment by rs_c.
" ld1 {v11.s}[2],[x2],x14                    \n\t" // Load c12  into quad and increment by rs_c.
" ld1 {v11.s}[3],[x2],x14                    \n\t" // Load c13  into quad and increment by rs_c.
"                                            \n\t"
"                                            \n\t"
" prfm pldl1keep,[x16,0]                     \n\t" // Prefetch.
" prfm pldl1keep,[x17,0]                     \n\t" // Prefetch.
"                                            \n\t"
" fmul v8.4s,v8.4s,v31.s[0]                  \n\t" // Scale by beta.
" fmul v9.4s,v9.4s,v31.s[0]                  \n\t" // Scale by beta.
" fmul v10.4s,v10.4s,v31.s[0]                \n\t" // Scale by beta.
" fmul v11.4s,v11.4s,v31.s[0]                \n\t" // Scale by beta.
"                                            \n\t"
" .BETAZEROGENSTORED:                        \n\t" // If beta==0, we cannot read from C (nor scale).
"                                            \n\t"
" fmla v8.4s,v12.4s,v30.s[0]                 \n\t" // Scale by alpha.
" fmla v9.4s,v13.4s,v30.s[0]                 \n\t" // Scale by alpha.
" fmla v10.4s,v14.4s,v30.s[0]                \n\t" // Scale by alpha.
" fmla v11.4s,v15.4s,v30.s[0]                \n\t" // Scale by alpha.
"                                            \n\t"
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
"                                            \n\t"
" st1 {v8.s}[0],[x2],x14                     \n\t" // Store c00  into quad and increment by rs_c.
" st1 {v8.s}[1],[x2],x14                     \n\t" // Store c01  into quad and increment by rs_c.
" st1 {v8.s}[2],[x2],x14                     \n\t" // Store c02  into quad and increment by rs_c.
" st1 {v8.s}[3],[x2],x14                     \n\t" // Store c03  into quad and increment by rs_c.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
" add x2,x2,x10                              \n\t" // c += cs_c.
"                                            \n\t"
" st1 {v9.s}[0],[x2],x14                     \n\t" // Store c10  into quad and increment by rs_c.
" st1 {v9.s}[1],[x2],x14                     \n\t" // Store c11  into quad and increment by rs_c.
" st1 {v9.s}[2],[x2],x14                     \n\t" // Store c12  into quad and increment by rs_c.
" st1 {v9.s}[3],[x2],x14                     \n\t" // Store c13  into quad and increment by rs_c.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
" add x2,x2,x10                              \n\t" // c += cs_c.
" add x2,x2,x10                              \n\t" // c += cs_c.
"                                            \n\t"
" st1 {v10.s}[0],[x2],x14                    \n\t" // Store c10  into quad and increment by rs_c.
" st1 {v10.s}[1],[x2],x14                    \n\t" // Store c11  into quad and increment by rs_c.
" st1 {v10.s}[2],[x2],x14                    \n\t" // Store c12  into quad and increment by rs_c.
" st1 {v10.s}[3],[x2],x14                    \n\t" // Store c13  into quad and increment by rs_c.
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C.
" add x2,x2,x10                              \n\t" // c += cs_c.
" add x2,x2,x10                              \n\t" // c += cs_c.
" add x2,x2,x10                              \n\t" // c += cs_c.
"                                            \n\t"
" st1 {v11.s}[0],[x2],x14                    \n\t" // Store c10  into quad and increment by rs_c.
" st1 {v11.s}[1],[x2],x14                    \n\t" // Store c11  into quad and increment by rs_c.
" st1 {v11.s}[2],[x2],x14                    \n\t" // Store c12  into quad and increment by rs_c.
" st1 {v11.s}[3],[x2],x14                    \n\t" // Store c13  into quad and increment by rs_c.
"                                            \n\t"
"                                            \n\t"
"                                            \n\t"
" .SEND:                                     \n\t" // Done!
"                                            \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_iter] "m" (k_iter), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 7
 [cs_c]   "m" (cs_c),   // 8
 [a_next] "m" (a_next), // 9
 [b_next] "m" (b_next), // 10
 [k]      "m" (k)       // 11
:// Register clobber list
 "x0", "x1", "x2", "x4",
 "x5", "x6", "x7", "x8",
 "x9", "x10","x11","x12",
 "x13","x14","x20",
 "v0", "v1", "v2", "v3",
 "v4", "v5", "v6", "v7",
 "v8", "v9", "v10","v11",
 "v12","v13","v14","v15",
 "v16","v17","v18","v19",
 "v20","v21","v22","v23",
 "v24","v25","v26","v27",
 "v30","v31"
);

}


/*
   o 4x4 Double precision micro-kernel NOT fully functional yet.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 3 GFLOPS @ 1.1 GHz. 

   December 2014.
*/
void bli_dgemm_opt_4x4(
                        dim_t              k,
                        double*   restrict alpha,
                        double*   restrict a,
                        double*   restrict b,
                        double*   restrict beta,
                        double*   restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	void* a_next = bli_auxinfo_next_a( data );
	void* b_next = bli_auxinfo_next_b( data );

	dim_t k_iter = k / 2;
	dim_t k_left = k % 2;

__asm__ volatile
(
"                                            \n\t"
" ldr x0,%[aaddr]                            \n\t" // Load address of A
" ldr x1,%[baddr]                            \n\t" // Load address of B
" ldr x2,%[caddr]                            \n\t" // Load address of C
"                                            \n\t"
" mov x4,#0                                  \n\t" // Init loop counter (i=0)
"                                            \n\t"
" ldr x16,%[a_next]                          \n\t" // Move pointer
" ldr x17,%[b_next]                          \n\t" // Move pointer
"                                            \n\t"
" ldr x5,%[k_iter]                           \n\t" // Init guard (k_iter)
" ldr x6,%[k_left]                           \n\t" // Init guard (k_iter)
"                                            \n\t" 
" ldr x7,%[alpha]                            \n\t" // Alpha address      
" ldr x8,%[beta]                             \n\t" // Beta address      
"                                            \n\t" 
" ldr x9,%[cs_c]                             \n\t" // Load cs_c
" lsl x10,x9,#3                              \n\t" // cs_c * sizeof(double)
"                                            \n\t" 
" ldp q0,q1,[x0],32                          \n\t" // Load a    into quad
" ldp q4,q5,[x1],32                          \n\t" // Load b    into quad
"                                            \n\t"
" movi v12.2d,#0                             \n\t" // Vector for accummulating column 0 
" movi v13.2d,#0                             \n\t" // Vector for accummulating column 0
" movi v14.2d,#0                             \n\t" // Vector for accummulating column 1
" movi v15.2d,#0                             \n\t" // Vector for accummulating column 1
" movi v16.2d,#0                             \n\t" // Vector for accummulating column 2 
" movi v17.2d,#0                             \n\t" // Vector for accummulating column 2
" movi v18.2d,#0                             \n\t" // Vector for accummulating column 3
" movi v19.2d,#0                             \n\t" // Vector for accummulating column 3
"                                            \n\t"
" ld1r {v31.2d},[x8]                         \n\t" // Load beta
"                                            \n\t"
" DLOOP:                                     \n\t" // Body
"                                            \n\t"
" ldp q6,q7,[x1],32                          \n\t" // Load b+4  into quad
"                                            \n\t"
" fmla v12.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v14.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v13.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" ldp q2,q3,[x0],32                          \n\t" // Load a+4  into quad
"                                            \n\t"
" fmla v16.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v18.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v17.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
" fmla v19.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" ldp q0,q1,[x0],32                          \n\t" // Load a    into quad
"                                            \n\t"
" fmla v12.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v14.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v13.2d,v3.2d,v6.d[0]                  \n\t" // Accummulate
" fmla v15.2d,v3.2d,v6.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" ldp q4,q5,[x1],32                          \n\t" // Load b    into quad
"                                            \n\t"
" fmla v16.2d,v2.2d,v7.d[0]                  \n\t" // Accummulate
" fmla v18.2d,v2.2d,v7.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" fmla v17.2d,v3.2d,v7.d[0]                  \n\t" // Accummulate
" fmla v19.2d,v3.2d,v7.d[1]                  \n\t" // Accummulate
"                                            \n\t"
" add x4,x4,1                                \n\t" // i = i+1
" cmp x4,x5                                  \n\t" // Continue
" blt DLOOP                                  \n\t" // if i < N 
"                                            \n\t"
" ldp q0,q1,[x2]                             \n\t" // Load c    into quad and increment by cs_c
" add x2,x2,x10                              \n\t"
" ldp q2,q3,[x2]                             \n\t" // Load c    into quad and increment by cs_c
" add x2,x2,x10                              \n\t"
" ldp q4,q5,[x2]                             \n\t" // Load c    into quad and increment by cs_c
" add x2,x2,x10                              \n\t"
" ldp q6,q7,[x2]                             \n\t" // Load c    into quad and increment by cs_c
"                                            \n\t"
" ld1r {v30.2d},[x7]                         \n\t" // Load alpha
"                                            \n\t"
" fmul v0.2d,v0.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v1.2d,v1.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v2.2d,v2.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v3.2d,v3.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v4.2d,v4.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v5.2d,v5.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v6.2d,v6.2d,v31.d[0]                  \n\t" // Scale by beta
" fmul v7.2d,v7.2d,v31.d[0]                  \n\t" // Scale by beta
"                                            \n\t"
" prfm pldl2keep,[x16]                       \n\t"
" prfm pldl2keep,[x17]                       \n\t"
"                                            \n\t"
" fmla v0.2d,v12.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v1.2d,v13.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v2.2d,v14.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v3.2d,v15.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v4.2d,v16.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v5.2d,v17.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v6.2d,v18.2d,v30.d[0]                 \n\t" // Scale by alpha
" fmla v7.2d,v19.2d,v30.d[0]                 \n\t" // Scale by alpha
"                                            \n\t"
" ldr x2,%[caddr]                            \n\t" // Load address of C
"                                            \n\t"
" stp q0,q1,[x2]                             \n\t" // Store quad into c    and increment by cs_c
" add x2,x2,x10                              \n\t"
" stp q2,q3,[x2]                             \n\t" // Store quad into c+4  and increment by cs_c
" add x2,x2,x10                              \n\t"
" stp q4,q5,[x2]                             \n\t" // Store quad into c+8  and increment by cs_c
" add x2,x2,x10                              \n\t"
" stp q6,q7,[x2]                             \n\t" // Store quad into c+16 and increment by cs_c
"                                            \n\t"
:// output operands (none)
:// input operands
 [aaddr]  "m" (a),      // 0
 [baddr]  "m" (b),      // 1
 [caddr]  "m" (c),      // 2
 [k_iter] "m" (k_iter), // 3
 [k_left] "m" (k_left), // 4
 [alpha]  "m" (alpha),  // 5
 [beta]   "m" (beta),   // 6
 [rs_c]   "m" (rs_c),   // 6
 [cs_c]   "m" (cs_c),   // 7
 [a_next] "m" (a_next), // 8
 [b_next] "m" (b_next)  // 9
:// Register clobber list
 "x0","x1","x2",
 "x4","x5","x6",
 "x7","x8","x9",
 "x10","x11","x12",
 "v0","v1","v2",
 "v3","v4","v5",
 "v6","v7","v8",
 "v9","v10","v11",
 "v12","v13","v14",
 "v15","v16","v17","v18","v19",
 "v30","v31"
);



}

void bli_cgemm_opt_4x4(
                        dim_t              k,
                        scomplex* restrict alpha,
                        scomplex* restrict a,
                        scomplex* restrict b,
                        scomplex* restrict beta,
                        scomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	/* Just call the reference implementation. */
	BLIS_CGEMM_UKERNEL_REF( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}

void bli_zgemm_opt_4x4(
                        dim_t              k,
                        dcomplex* restrict alpha,
                        dcomplex* restrict a,
                        dcomplex* restrict b,
                        dcomplex* restrict beta,
                        dcomplex* restrict c, inc_t rs_c, inc_t cs_c,
                        auxinfo_t*         data
                      )
{
	/* Just call the reference implementation. */
	BLIS_ZGEMM_UKERNEL_REF( k,
	                   alpha,
	                   a,
	                   b,
	                   beta,
	                   c, rs_c, cs_c,
	                   data );
}

