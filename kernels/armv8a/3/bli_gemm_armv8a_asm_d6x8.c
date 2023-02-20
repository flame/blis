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
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "blis.h"
#include "armv8a_asm_utils.h"

// #define DISPLAY_DEBUG_INFO

// Added prefetch fix for non-cacheline aligned C columns
// (with the prefetches interleaved with other instructions)
// to both sgemm and dgemm versions.

// Added sgemm prefetch fix for non-cacheline aligned C columns
// (with the prefetches interleaved with other instructions)

/*
   o 4x4 Single precision micro-kernel fully functional.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 7.3 GFLOPS @ 1.1 GHz.

   December 2014.

 * UPDATE NOVEMBER 2015
 * Micro-kernel changed to 8x12
 * Tested on Juno Board. Around  8.1 GFLOPS, 1 x A57 core  @ 1.1 GHz.
 * Tested on Juno Board. Around 15.9 GFLOPS, 2 x A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around  3.1 GFLOPS, 1 x A53 core  @ 850 MHz.
 * Tested on Juno board. Around 12   GFLOPS, 4 x A53 cores @ 850 MHz.

 * UPDATE JULY 2021 - Leick Robinson
 * Both Microkernels changed to fix two prefetching performance bugs
 * Tested on 2s Altra.   Around 6,900 GFLOPS, 160 x N2 cores @ 3.0 GHz
 * Tested on 1s Altra Max. Arnd 5,800 GFLOPS. 128 x N2 cores @ 3.0 GHz
*/

void bli_sgemm_armv8a_asm_8x12
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
             auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( s, 8, 12, false );


	__asm__ volatile
	(
	"                                            \n\t"
	"                                            \n\t"
	" ldr x0,%[aaddr]                            \n\t" // Load address of A.
	" ldr x1,%[baddr]                            \n\t" // Load address of B.
	" ldr x2,%[caddr]                            \n\t" // Load address of C.
	"                                            \n\t"
	" ldr x10,%[cs_c]                            \n\t" // Load cs_c.
	" lsl x10,x10,#2                             \n\t" // cs_c * sizeof(float) -- AUX.
	"                                            \n\t"
	" ldr x5,%[k_iter]                           \n\t" // Number of unrolled iterations (k_iter).
	" ldr x6,%[k_left]                           \n\t" // Number of remaining iterations (k_left).
	" add x16,x2,x10                             \n\t" // Load address Column 1 of C
	"                                            \n\t"
	// " ldr x14,%[rs_c]                            \n\t" // Load rs_c.
	// " lsl x14,x14,#2                             \n\t" // rs_c * sizeof(float).
	"                                            \n\t"
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t" // Vector for accummulating column 0
	" prfm    PLDL1KEEP, [x1, #192]              \n\t"
	" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
	" add x17,x16,x10                            \n\t" // Load address Column 2 of C

	" dup  v9.4s, wzr                            \n\t" // Vector for accummulating column 0
	" prfm    PLDL1KEEP, [x1, #256]              \n\t"
	"                                            \n\t" // Since columns of C can cross a cache
	                                                   // line boundary, we also need to prefetch
	                                                   // the "ends."
	" prfm pldl1keep,[x2, #16]                   \n\t" // Prefetch c.
	" add x19,x17,x10                            \n\t" // Load address Column 3 of C

	" dup  v10.4s, wzr                           \n\t" // Vector for accummulating column 1
	" prfm    PLDL1KEEP, [x1, #320]              \n\t"
	" prfm pldl1keep,[x16]                       \n\t" // Prefetch c.
	" add x20,x19,x10                            \n\t" // Load address Column 4 of C

	" dup  v11.4s, wzr                           \n\t" // Vector for accummulating column 1
	" prfm pldl1keep,[x16]                       \n\t" // Prefetch c.
	" prfm pldl1keep,[x16, #16]                  \n\t" // Prefetch c.

	" dup  v12.4s, wzr                           \n\t" // Vector for accummulating column 2
	" prfm pldl1keep,[x17]                       \n\t" // Prefetch c.
	" add x21,x20,x10                            \n\t" // Load address Column 5 of C

	" dup  v13.4s, wzr                           \n\t" // Vector for accummulating column 2
	" prfm pldl1keep,[x17, #16]                  \n\t" // Prefetch c.
	"                                            \n\t"
	" dup  v14.4s, wzr                           \n\t" // Vector for accummulating column 3
	" prfm    PLDL1KEEP, [x0, #128]              \n\t"
	" prfm pldl1keep,[x19]                       \n\t" // Prefetch c.
	" add x22,x21,x10                            \n\t" // Load address Column 6 of C

	" dup  v15.4s, wzr                           \n\t" // Vector for accummulating column 3
	" prfm    PLDL1KEEP, [x0, #192]              \n\t"
	" prfm pldl1keep,[x19, #16]                  \n\t" // Prefetch c.

	" dup  v16.4s, wzr                           \n\t" // Vector for accummulating column 4
	" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.
	" add x23,x22,x10                            \n\t" // Load address Column 7 of C

	" dup  v17.4s, wzr                           \n\t" // Vector for accummulating column 4
	" prfm pldl1keep,[x20, #16]                  \n\t" // Prefetch c.

	" dup  v18.4s, wzr                           \n\t" // Vector for accummulating column 5
	" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
	" add x24,x23,x10                            \n\t" // Load address Column 8 of C

	" dup  v19.4s, wzr                           \n\t" // Vector for accummulating column 5
	" prfm pldl1keep,[x21, #16]                  \n\t" // Prefetch c.

	"                                            \n\t"
	" dup  v20.4s, wzr                           \n\t" // Vector for accummulating column 6
	" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.
	" add x25,x24,x10                            \n\t" // Load address Column 9 of C

	" dup  v21.4s, wzr                           \n\t" // Vector for accummulating column 6
	" prfm pldl1keep,[x22, #16]                  \n\t" // Prefetch c.

	" dup  v22.4s, wzr                           \n\t" // Vector for accummulating column 7
	" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.
	" add x26,x25,x10                            \n\t" // Load address Column 10 of C

	" dup  v23.4s, wzr                           \n\t" // Vector for accummulating column 7
	" prfm pldl1keep,[x23, #16]                  \n\t" // Prefetch c.

	" dup  v24.4s, wzr                           \n\t" // Vector for accummulating column 8
	" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
	" add x27,x26,x10                            \n\t" // Load address Column 11 of C

	" dup  v25.4s, wzr                           \n\t" // Vector for accummulating column 8
	" prfm pldl1keep,[x24, #16]                  \n\t" // Prefetch c.
	"                                            \n\t"
	" dup  v26.4s, wzr                           \n\t" // Vector for accummulating column 9
	" prfm pldl1keep,[x25]                       \n\t" // Prefetch c.

	" dup  v27.4s, wzr                           \n\t" // Vector for accummulating column 9
	" prfm pldl1keep,[x25, #16]                  \n\t" // Prefetch c.

	" dup  v28.4s, wzr                           \n\t" // Vector for accummulating column 10
	" prfm pldl1keep,[x26]                       \n\t" // Prefetch c.

	" dup  v29.4s, wzr                           \n\t" // Vector for accummulating column 10
	" prfm pldl1keep,[x26, #16]                  \n\t" // Prefetch c.

	" dup  v30.4s, wzr                           \n\t" // Vector for accummulating column 11
	" prfm pldl1keep,[x27]                       \n\t" // Prefetch c.

	" dup  v31.4s, wzr                           \n\t" // Vector for accummulating column 11
	" prfm pldl1keep,[x27, #16]                  \n\t" // Prefetch c.
	"                                            \n\t"
	"                                            \n\t"

	" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
	BEQ(SCONSIDERKLEFT)
	"                                            \n\t"
	" ldr q0, [x0]                               \n\t"
	" ldr q1, [x0, #16]                          \n\t" // Load a
	"                                            \n\t"
	" ldr q2, [x1]                               \n\t" // Load b
	" ldr q3, [x1, #16]                          \n\t"
	" ldr q4, [x1, #32]                          \n\t"
	"                                            \n\t"
	" add x0, x0, #32                            \n\t" // Update address of A
	" add x1, x1, #48                            \n\t" // Update address of B
	"                                            \n\t"
	" cmp x5,1                                   \n\t" // If there's only one k_iter, jump to it
	BEQ(SLASTITER)                                     // (as loop is do-while-like).
	"                                            \n\t"
	LABEL(SLOOPKITER)                                  // Body of the k_iter loop.
	"                                            \n\t"
	" ldr q5, [x0]                               \n\t"
	" fmla v8.4s, v0.4s,v2.s[0]                  \n\t" // Accummulate.
	" fmla v9.4s, v1.4s,v2.s[0]                  \n\t" // Accummulate.
	" ldr q6, [x0, #16]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" prfm    PLDL1KEEP, [x1, #336]              \n\t"
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" prfm    PLDL1KEEP, [x1, #400]              \n\t"
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" prfm    PLDL1KEEP, [x1, #464]              \n\t"
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #32]                          \n\t"
	"                                            \n\t"                  // End It 1
	"                                            \n\t"
	" ldr q0, [x0, #32]                          \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q1, [x0, #48]                          \n\t"
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #48]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" prfm    PLDL1KEEP, [x0, #224]              \n\t"
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" prfm    PLDL1KEEP, [x0, #288]              \n\t"
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t"                  // End It 2
	"                                            \n\t"
	" ldr q5, [x0, #64]                          \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q6, [x0, #80]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #112]                         \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #128]                         \n\t"
	"                                            \n\t"                  // End It 3
	"                                            \n\t"
	" ldr q0, [x0, #96]                          \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q1, [x0, #112]                         \n\t"
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #144]                         \n\t"
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #160]                         \n\t"
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #176]                         \n\t"
	" add x1, x1, #192                           \n\t"
	" add x0, x0, #128                           \n\t"
	"                                            \n\t"                  // End It 4
	" sub x5,x5,1                                \n\t" // i-=1.
	" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
	BNE(SLOOPKITER)
	"                                            \n\t"
	LABEL(SLASTITER)                                   // Last iteration of k_iter loop.
	"                                            \n\t"
	"                                            \n\t"
	" ldr q5, [x0]                               \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q6, [x0, #16]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #32]                          \n\t"
	"                                            \n\t"                  // End It 1
	"                                            \n\t"
	" ldr q0, [x0, #32]                          \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q1, [x0, #48]                          \n\t"
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #48]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t"                  // End It 2
	"                                            \n\t"
	" ldr q5, [x0, #64]                          \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" ldr q6, [x0, #80]                          \n\t"
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	" ldr q2, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q3, [x1, #112]                         \n\t"
	"                                            \n\t"
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	" ldr q4, [x1, #128]                         \n\t"
	"                                            \n\t"                  // End It 3
	"                                            \n\t"
	" fmla v8.4s,v5.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v6.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v10.4s,v5.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v6.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v5.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v6.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v5.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v6.4s,v2.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v16.4s,v5.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v6.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v5.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v6.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v5.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v6.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v5.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v6.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v5.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v5.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v5.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v5.4s,v4.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v25.4s,v6.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v6.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v6.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v6.4s,v4.s[3]                  \n\t" // Accummulate.
	" add x1, x1, #144                           \n\t"
	" add x0, x0, #96                            \n\t"
	"                                            \n\t"                  // End It 4
	"                                            \n\t"
	LABEL(SCONSIDERKLEFT)
	" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
	BEQ(SPOSTACCUM)                                    // else, we enter the k_left loop.
	"                                            \n\t"
	LABEL(SLOOPKLEFT)                                  // Body of the left iterations
	"                                            \n\t"
	" ldr q0, [x0],#16                           \n\t"
	" ldr q1, [x0],#16                           \n\t" // Load a
	"                                            \n\t"
	" ldr q2, [x1],#16                           \n\t" // Load b
	" ldr q3, [x1],#16                           \n\t"
	" ldr q4, [x1],#16                           \n\t"
	"                                            \n\t"
	" sub x6,x6,1                                \n\t" // i = i-1.
	"                                            \n\t"
	" fmla v8.4s,v0.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v9.4s,v1.4s,v2.s[0]                   \n\t" // Accummulate.
	" fmla v10.4s,v0.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v11.4s,v1.4s,v2.s[1]                  \n\t" // Accummulate.
	" fmla v12.4s,v0.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v13.4s,v1.4s,v2.s[2]                  \n\t" // Accummulate.
	" fmla v14.4s,v0.4s,v2.s[3]                  \n\t" // Accummulate.
	" fmla v15.4s,v1.4s,v2.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v16.4s,v0.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v17.4s,v1.4s,v3.s[0]                  \n\t" // Accummulate.
	" fmla v18.4s,v0.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v19.4s,v1.4s,v3.s[1]                  \n\t" // Accummulate.
	" fmla v20.4s,v0.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v21.4s,v1.4s,v3.s[2]                  \n\t" // Accummulate.
	" fmla v22.4s,v0.4s,v3.s[3]                  \n\t" // Accummulate.
	" fmla v23.4s,v1.4s,v3.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" fmla v24.4s,v0.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v26.4s,v0.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v28.4s,v0.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v30.4s,v0.4s,v4.s[3]                  \n\t" // Accummulate.
	" fmla v25.4s,v1.4s,v4.s[0]                  \n\t" // Accummulate.
	" fmla v27.4s,v1.4s,v4.s[1]                  \n\t" // Accummulate.
	" fmla v29.4s,v1.4s,v4.s[2]                  \n\t" // Accummulate.
	" fmla v31.4s,v1.4s,v4.s[3]                  \n\t" // Accummulate.
	"                                            \n\t"
	" cmp x6,0                                   \n\t" // Iterate again.
	BNE(SLOOPKLEFT)                                    // if i!=0.
	"                                            \n\t"
	LABEL(SPOSTACCUM)
	"                                            \n\t"
	" ldr x0,%[alpha]                            \n\t" // Alpha address.
	" ldr x1,%[beta]                             \n\t" // Beta address.
	"                                            \n\t"
	" ld1r {v6.4s},[x0]                          \n\t" // Load alpha.
	" ld1r {v7.4s},[x1]                          \n\t" // Load beta
	"                                            \n\t"
	" ldr x0,%[a_next]                           \n\t" // Pointer to next block of A.
	" ldr x1,%[b_next]                           \n\t" // Pointer to next pointer of B.
	"                                            \n\t"
	LABEL(SCOLSTORED)                                  // C is column-major.
	"                                            \n\t"
	" dup  v0.4s, wzr                            \n\t"
	" dup  v1.4s, wzr                            \n\t"
	" dup  v2.4s, wzr                            \n\t"
	" dup  v3.4s, wzr                            \n\t"
	" dup  v4.4s, wzr                            \n\t"
	" dup  v5.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ(SBETAZEROCOLSTOREDS1)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x2]                               \n\t" // Load column 0 of C
	" ldr q1, [x2, #16]                          \n\t"
	" ldr q2, [x16]                              \n\t" // Load column 1 of C
	" ldr q3, [x16, #16]                         \n\t"
	" ldr q4, [x17]                              \n\t" // Load column 2 of C
	" ldr q5, [x17, #16]                         \n\t"
	"                                            \n\t"
	" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(SBETAZEROCOLSTOREDS1)
	"                                            \n\t"
	" fmla v0.4s,v8.4s,v6.s[0]                   \n\t" // Scale by alpha
	" fmla v1.4s,v9.4s,v6.s[0]                   \n\t" // Scale by alpha
	" fmla v2.4s,v10.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v3.4s,v11.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v4.4s,v12.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v5.4s,v13.4s,v6.s[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x2]                               \n\t" // Store column 0 of C
	" str q1, [x2, #16]                          \n\t"
	" str q2, [x16]                              \n\t" // Store column 1 of C
	" str q3, [x16, #16]                         \n\t"
	" str q4, [x17]                              \n\t" // Store column 2 of C
	" str q5, [x17, #16]                         \n\t"
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t"
	" dup  v9.4s, wzr                            \n\t"
	" dup  v10.4s, wzr                           \n\t"
	" dup  v11.4s, wzr                           \n\t"
	" dup  v12.4s, wzr                           \n\t"
	" dup  v13.4s, wzr                           \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ(SBETAZEROCOLSTOREDS2)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x19]                              \n\t" // Load column 3 of C
	" ldr q9, [x19, #16]                         \n\t"
	" ldr q10, [x20]                             \n\t" // Load column 4 of C
	" ldr q11, [x20, #16]                        \n\t"
	" ldr q12, [x21]                             \n\t" // Load column 5 of C
	" ldr q13, [x21, #16]                        \n\t"
	"                                            \n\t"
	" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(SBETAZEROCOLSTOREDS2)
	"                                            \n\t"
	" fmla v8.4s, v14.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v9.4s, v15.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v10.4s,v16.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v11.4s,v17.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v12.4s,v18.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v13.4s,v19.4s,v6.s[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x19]                              \n\t" // Store column 3 of C
	" str q9, [x19, #16]                         \n\t"
	" str q10, [x20]                             \n\t" // Store column 4 of C
	" str q11, [x20, #16]                        \n\t"
	" str q12, [x21]                             \n\t" // Store column 5 of C
	" str q13, [x21, #16]                        \n\t"
	"                                            \n\t"
	" dup  v0.4s, wzr                            \n\t"
	" dup  v1.4s, wzr                            \n\t"
	" dup  v2.4s, wzr                            \n\t"
	" dup  v3.4s, wzr                            \n\t"
	" dup  v4.4s, wzr                            \n\t"
	" dup  v5.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ(SBETAZEROCOLSTOREDS3)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x22]                              \n\t" // Load column 6 of C
	" ldr q1, [x22, #16]                         \n\t"
	" ldr q2, [x23]                              \n\t" // Load column 7 of C
	" ldr q3, [x23, #16]                         \n\t"
	" ldr q4, [x24]                              \n\t" // Load column 8 of C
	" ldr q5, [x24, #16]                         \n\t"
	"                                            \n\t"
	" fmul v0.4s,v0.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v1.4s,v1.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v2.4s,v2.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v3.4s,v3.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v4.4s,v4.4s,v7.s[0]                   \n\t" // Scale by beta
	" fmul v5.4s,v5.4s,v7.s[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(SBETAZEROCOLSTOREDS3)
	"                                            \n\t"
	" fmla v0.4s,v20.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v1.4s,v21.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v2.4s,v22.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v3.4s,v23.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v4.4s,v24.4s,v6.s[0]                  \n\t" // Scale by alpha
	" fmla v5.4s,v25.4s,v6.s[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x22]                              \n\t" // Store column 6 of C
	" str q1, [x22, #16]                         \n\t"
	" str q2, [x23]                              \n\t" // Store column 7 of C
	" str q3, [x23, #16]                         \n\t"
	" str q4, [x24]                              \n\t" // Store column 8 of C
	" str q5, [x24, #16]                         \n\t"
	"                                            \n\t"
	" dup  v8.4s, wzr                            \n\t"
	" dup  v9.4s, wzr                            \n\t"
	" dup  v10.4s, wzr                            \n\t"
	" dup  v11.4s, wzr                            \n\t"
	" dup  v12.4s, wzr                            \n\t"
	" dup  v13.4s, wzr                            \n\t"
	"                                            \n\t"
	" fcmp s7,#0.0                               \n\t"
	BEQ(SBETAZEROCOLSTOREDS4)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x25]                              \n\t" // Load column 9 of C
	" ldr q9, [x25, #16]                         \n\t"
	" ldr q10, [x26]                             \n\t" // Load column 10 of C
	" ldr q11, [x26, #16]                        \n\t"
	" ldr q12, [x27]                             \n\t" // Load column 11 of C
	" ldr q13, [x27, #16]                        \n\t"
	"                                            \n\t"
	" fmul v8.4s, v8.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v9.4s, v9.4s, v7.s[0]                 \n\t" // Scale by beta
	" fmul v10.4s,v10.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v11.4s,v11.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v12.4s,v12.4s,v7.s[0]                 \n\t" // Scale by beta
	" fmul v13.4s,v13.4s,v7.s[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(SBETAZEROCOLSTOREDS4)
	"                                            \n\t"
	" prfm pldl2keep,[x0]                        \n\t"
	" prfm pldl2keep,[x1]                        \n\t"
	"                                            \n\t"
	" fmla v8.4s, v26.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v9.4s, v27.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v10.4s,v28.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v11.4s,v29.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v12.4s,v30.4s,v6.s[0]                 \n\t" // Scale by alpha
	" fmla v13.4s,v31.4s,v6.s[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x25]                              \n\t" // Store column 9 of C
	" str q9, [x25, #16]                         \n\t"
	" str q10, [x26]                             \n\t" // Store column 10 of C
	" str q11, [x26, #16]                        \n\t"
	" str q12, [x27]                             \n\t" // Store column 11 of C
	" str q13, [x27, #16]                        \n\t"
	"                                            \n\t"
	"                                            \n\t"
	// BRANCH(SEND)                                       // Done.
	// LABEL(SEND)                                        // Done!
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
	 [b_next] "m" (b_next) // 10
	:// Register clobber list
	 "x0", "x1", "x2",
	 "x5", "x6", "x10",
	 "x16","x17","x19","x20",
	 "x21","x22","x23","x24",
	 "x25","x26","x27",
	 "v0", "v1", "v2", "v3",
	 "v4", "v5", "v6", "v7",
	 "v8", "v9", "v10","v11",
	 "v12","v13","v14","v15",
	 "v16","v17","v18","v19",
	 "v20","v21","v22","v23",
	 "v24","v25","v26","v27",
	 "v28","v29","v30","v31"
	);

	GEMM_UKR_FLUSH_CT( s );
}


/*
   o 4x4 Double precision micro-kernel NOT fully functional yet.
   o Runnable on ARMv8, compiled with aarch64 GCC.
   o Use it together with the armv8 BLIS configuration.
   o Tested on Juno board. Around 3 GFLOPS @ 1.1 GHz.

   December 2014.

 * UPDATE OCTOBER 2015: Now is fully functional.
 * Tested on Juno board. Around 5.6 GFLOPS, 2 A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around 4 GFLOPS, 4 A53 cores @ 850 MHz.

 * UPDATE NOVEMBER 2015
 * Micro-kernel changed to 6x8
 * Tested on Juno Board. Around 4   GFLOPS, 1 x A57 core  @ 1.1 GHz.
 * Tested on Juno Board. Around 7.6 GFLOPS, 2 x A57 cores @ 1.1 GHz.
 * Tested on Juno board. Around 1.5 GFLOPS, 1 x A53 core  @ 850 MHz.
 * Tested on Juno board. Around 5.5 GFLOPS, 4 x A53 cores @ 850 MHz.

 * UPDATE JULY 2021 - Leick Robinson
 * Both Microkernels changed to fix two prefetching performance bugs
 * Tested on 2s Altra. Around 3,200 GFLOPS, 160 x N2 cores @ 3.0 GHz
 * Tested on 1s Altra, Around 1,700 GFLOPS,  80 x N2 cores @ 3.0 GHz
 * Tested on 1s Altra Max,  ~ 2,600 GFLOPS, 128 x N2 cores @ 3.0 GHz
 */
void bli_dgemm_armv8a_asm_6x8
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c0, inc_t cs_c0,
             auxinfo_t* data,
       const cntx_t*    cntx
     )
{
#ifdef DISPLAY_DEBUG_INFO

	static bool bFirstTime = true;

	if ( bFirstTime )
	{
		printf( "In bli_dgemm_armv8a_asm_6x8: rs_c0=%d, cs_c0=%d \n",
		        (int) rs_c0, (int) cs_c0 );
		fflush( stdout );
		bFirstTime = false;
	}

#endif

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

	GEMM_UKR_SETUP_CT( d, 6, 8, false );

	__asm__ volatile
	(
	"                                            \n\t"
	" ldr x0,%[aaddr]                            \n\t" // Load address of A
	" ldr x1,%[baddr]                            \n\t" // Load address of B
	" ldr x2,%[caddr]                            \n\t" // Load address of C
	"                                            \n\t"
	" ldr x10,%[cs_c]                            \n\t" // Load cs_c
	" lsl x10,x10,#3                             \n\t" // cs_c * sizeof(double)
	"                                            \n\t"
	" ldr x5,%[k_iter]                           \n\t" // Init guard (k_iter)
	" ldr x6,%[k_left]                           \n\t" // Init guard (k_iter)
	" add x20,x2,x10                             \n\t" // Load address Column 1 of C
	"                                            \n\t"
	// " ldr x14,%[rs_c]                            \n\t" // Load rs_c.
	// " lsl x14,x14,#3                             \n\t" // rs_c * sizeof(double).
	"                                            \n\t"
	"                                            \n\t"
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t" // Vector for accummulating column 0
	" prfm    PLDL1KEEP, [x1, #256]              \n\t"
	" dup  v9.2d, xzr                            \n\t" // Vector for accummulating column 0
	" prfm    PLDL1KEEP, [x1, #320]              \n\t"
	" dup  v10.2d, xzr                           \n\t" // Vector for accummulating column 0
	" prfm    PLDL1KEEP, [x1, #384]              \n\t"
	" dup  v11.2d, xzr                           \n\t" // Vector for accummulating column 1
	" prfm    PLDL1KEEP, [x1, #448]              \n\t"
	" dup  v12.2d, xzr                           \n\t" // Vector for accummulating column 1
	" prfm    PLDL1KEEP, [x0, #192]              \n\t"
	" add x21,x20,x10                            \n\t" // Load address Column 2 of C

	" dup  v13.2d, xzr                           \n\t" // Vector for accummulating column 1
	" prfm    PLDL1KEEP, [x0, #256]              \n\t"
	"                                            \n\t"
	" dup  v14.2d, xzr                           \n\t" // Vector for accummulating column 2
	" prfm    PLDL1KEEP, [x0, #320]              \n\t"
	" add x22,x21,x10                            \n\t" // Load address Column 3 of C

	" dup  v15.2d, xzr                           \n\t" // Vector for accummulating column 2
	" prfm pldl1keep,[x2]                        \n\t" // Prefetch c.
	" dup  v16.2d, xzr                           \n\t" // Vector for accummulating column 2
	"                                            \n\t" // Since columns of C can cross a cache
	                                                   // line boundary, we also need to prefetch
	                                                   // the "ends."
	" prfm pldl1keep,[x2, #32]                   \n\t" // Prefetch c.
	" add x23,x22,x10                            \n\t" // Load address Column 4 of C

	" dup  v17.2d, xzr                           \n\t" // Vector for accummulating column 3
	" prfm pldl1keep,[x20]                       \n\t" // Prefetch c.

	" dup  v18.2d, xzr                           \n\t" // Vector for accummulating column 3
	" prfm pldl1keep,[x20, #32]                  \n\t" // Prefetch c.
	" add x24,x23,x10                            \n\t" // Load address Column 5 of C

	" dup  v19.2d, xzr                           \n\t" // Vector for accummulating column 3
	" prfm pldl1keep,[x21]                       \n\t" // Prefetch c.
	"                                            \n\t"

	" dup  v20.2d, xzr                           \n\t" // Vector for accummulating column 4
	" prfm pldl1keep,[x21, #32]                  \n\t" // Prefetch c.
	" add x25,x24,x10                            \n\t" // Load address Column 6 of C

	" dup  v21.2d, xzr                           \n\t" // Vector for accummulating column 4
	" prfm pldl1keep,[x22]                       \n\t" // Prefetch c.

	" dup  v22.2d, xzr                           \n\t" // Vector for accummulating column 4
	" prfm pldl1keep,[x22, #32]                  \n\t" // Prefetch c.
	" add x26,x25,x10                            \n\t" // Load address Column 7 of C

	" dup  v23.2d, xzr                           \n\t" // Vector for accummulating column 5
	" prfm pldl1keep,[x23]                       \n\t" // Prefetch c.

	" dup  v24.2d, xzr                           \n\t" // Vector for accummulating column 5
	" prfm pldl1keep,[x23, #32]                  \n\t" // Prefetch c.

	" dup  v25.2d, xzr                           \n\t" // Vector for accummulating column 5
	" prfm pldl1keep,[x24]                       \n\t" // Prefetch c.
	"                                            \n\t"
	" dup  v26.2d, xzr                           \n\t" // Vector for accummulating column 6
	" prfm pldl1keep,[x24, #32]                  \n\t" // Prefetch c.

	" dup  v27.2d, xzr                           \n\t" // Vector for accummulating column 6
	" prfm pldl1keep,[x25]                       \n\t" // Prefetch c.

	" dup  v28.2d, xzr                           \n\t" // Vector for accummulating column 6
	" prfm pldl1keep,[x25, #32]                  \n\t" // Prefetch c.

	" dup  v29.2d, xzr                           \n\t" // Vector for accummulating column 7
	" prfm pldl1keep,[x26]                       \n\t" // Prefetch c.

	" dup  v30.2d, xzr                           \n\t" // Vector for accummulating column 7
	" prfm pldl1keep,[x26, #32]                  \n\t" // Prefetch c.

	" dup  v31.2d, xzr                           \n\t" // Vector for accummulating column 7
	"                                            \n\t"
	"                                            \n\t"

	" cmp x5,#0                                  \n\t" // If k_iter == 0, jump to k_left.
	BEQ(DCONSIDERKLEFT)
	"                                            \n\t"
	" ldr q0, [x0]                               \n\t" // Load a
	" ldr q1, [x0, #16]                          \n\t"
	" ldr q2, [x0, #32]                          \n\t"
	"                                            \n\t"
	" ldr q3, [x1]                               \n\t" // Load b
	" ldr q4, [x1, #16]                          \n\t"
	" ldr q5, [x1, #32]                          \n\t"
	" ldr q6, [x1, #48]                          \n\t"
	"                                            \n\t"
	" add x0, x0, #48                            \n\t" // Update address of A
	" add x1, x1, #64                            \n\t" // Update address of B
	"                                            \n\t"
	" cmp x5,1                                   \n\t" // If there's only one k_iter, jump to it
	BEQ(DLASTITER)                                     // (as loop is do-while-like).
	"                                            \n\t"
	LABEL(DLOOP)                                       // Body
	"                                            \n\t"
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x1, #448]              \n\t" //512-64=448
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x1, #512]              \n\t"
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x1, #576]              \n\t"
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q3, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q7, [x0, #32]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	" ldr q4, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #32]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0]                               \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #16]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #48]                          \n\t"
	"                                            \n\t"                  // End it 1
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x1, #640]              \n\t"
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x0, #336]              \n\t"
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x0, #400]              \n\t"
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q2, [x0, #80]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #48]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #64]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #112]                         \n\t"
	"                                            \n\t"                  // End it 2
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" prfm    PLDL1KEEP, [x0, #464]              \n\t"
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q3, [x1, #128]                         \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q7, [x0, #128]                         \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	" ldr q4, [x1, #144]                         \n\t"
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #160]                         \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #96]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #112]                         \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #176]                         \n\t"
	"                                            \n\t"                  // End it 3
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1, #192]                         \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q2, [x0, #176]                         \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #208]                         \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #224]                         \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #144]                         \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #160]                         \n\t"
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #240]                         \n\t"
	"                                            \n\t"                  // End it 4
	" add x0, x0, #192                           \n\t"
	" add x1, x1, #256                           \n\t"
	"                                            \n\t"
	" sub x5,x5,1                                \n\t" // i-=1
	" cmp x5,1                                   \n\t" // Iterate again if we are not in k_iter == 1.
	BNE(DLOOP)
	"                                            \n\t"
	LABEL(DLASTITER)
	"                                            \n\t"
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1]                               \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q7, [x0, #32]                          \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #16]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #32]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0]                               \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #16]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #48]                          \n\t"
	"                                            \n\t"                  // End it 1
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1, #64]                          \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q2, [x0, #80]                          \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #80]                          \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #96]                          \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #48]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #64]                          \n\t"
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #112]                         \n\t"
	"                                            \n\t"                  // End it 2
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	" ldr q3, [x1, #128]                         \n\t"
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	" ldr q7, [x0, #128]                         \n\t"
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	" ldr q4, [x1, #144]                         \n\t"
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	" ldr q5, [x1, #160]                         \n\t"
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q0, [x0, #96]                          \n\t"
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q1, [x0, #112]                         \n\t"
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	" ldr q6, [x1, #176]                         \n\t"
	"                                            \n\t"                  // End it 3
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v7.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v7.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v7.2d,v4.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v7.2d,v4.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v7.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v7.2d,v5.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" add x1, x1, #192                           \n\t"
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v28.2d,v7.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v7.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"                  // End it 4
	" add x0, x0, #144                           \n\t"
	"                                            \n\t"
	LABEL(DCONSIDERKLEFT)
	" cmp x6,0                                   \n\t" // If k_left == 0, we are done.
	BEQ(DPOSTACCUM)                                    // else, we enter the k_left loop.
	"                                            \n\t"
	LABEL(DLOOPKLEFT)
	"                                            \n\t"
	" ldr q0, [x0],#16                           \n\t"
	" ldr q1, [x0],#16                           \n\t" // Load a
	" ldr q2, [x0],#16                           \n\t"
	"                                            \n\t"
	" ldr q3, [x1],#16                           \n\t" // Load b
	" ldr q4, [x1],#16                           \n\t"
	" ldr q5, [x1],#16                           \n\t"
	" ldr q6, [x1],#16                           \n\t"
	"                                            \n\t"
	" sub x6,x6,1                                \n\t"
	"                                            \n\t"
	" fmla v8.2d ,v0.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v9.2d ,v1.2d,v3.d[0]                  \n\t" // Accummulate
	" fmla v10.2d,v2.2d,v3.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v11.2d,v0.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v12.2d,v1.2d,v3.d[1]                  \n\t" // Accummulate
	" fmla v13.2d,v2.2d,v3.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v14.2d,v0.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v15.2d,v1.2d,v4.d[0]                  \n\t" // Accummulate
	" fmla v16.2d,v2.2d,v4.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v17.2d,v0.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v18.2d,v1.2d,v4.d[1]                  \n\t" // Accummulate
	" fmla v19.2d,v2.2d,v4.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v20.2d,v0.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v21.2d,v1.2d,v5.d[0]                  \n\t" // Accummulate
	" fmla v22.2d,v2.2d,v5.d[0]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v23.2d,v0.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v24.2d,v1.2d,v5.d[1]                  \n\t" // Accummulate
	" fmla v25.2d,v2.2d,v5.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v26.2d,v0.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v29.2d,v0.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v27.2d,v1.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v30.2d,v1.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" fmla v28.2d,v2.2d,v6.d[0]                  \n\t" // Accummulate
	" fmla v31.2d,v2.2d,v6.d[1]                  \n\t" // Accummulate
	"                                            \n\t"
	" cmp x6,0                                   \n\t" // Iterate again.
	BNE(DLOOPKLEFT)                                    // if i!=0.
	"                                            \n\t"
	LABEL(DPOSTACCUM)
	"                                            \n\t"
	" ldr x0,%[alpha]                            \n\t" // Alpha address
	" ldr x1,%[beta]                             \n\t" // Beta address
	"                                            \n\t"
	" ld1r {v6.2d},[x0]                          \n\t" // Load alpha.
	" ld1r {v7.2d},[x1]                          \n\t" // Load beta
	"                                            \n\t"
	" ldr x0,%[a_next]                           \n\t" // Next A address for later use.
	" ldr x1,%[b_next]                           \n\t" // Next B address for later use.
	"                                            \n\t"
	LABEL(DCOLSTORED)                                  // C is column-major.
	"                                            \n\t"
	" dup  v0.2d, xzr                            \n\t"
	" dup  v1.2d, xzr                            \n\t"
	" dup  v2.2d, xzr                            \n\t"
	" dup  v3.2d, xzr                            \n\t"
	" dup  v4.2d, xzr                            \n\t"
	" dup  v5.2d, xzr                            \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ(DBETAZEROCOLSTOREDS1)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x2]                               \n\t" // Load column 0 of C
	" ldr q1, [x2, #16]                          \n\t"
	" ldr q2, [x2, #32]                          \n\t"
	"                                            \n\t"
	" ldr q3, [x20]                              \n\t" // Load column 1 of C
	" ldr q4, [x20, #16]                         \n\t"
	" ldr q5, [x20, #32]                         \n\t"
	"                                            \n\t"
	" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(DBETAZEROCOLSTOREDS1)
	"                                            \n\t"
	" fmla v0.2d,v8.2d,v6.d[0]                   \n\t" // Scale by alpha
	" fmla v1.2d,v9.2d,v6.d[0]                   \n\t" // Scale by alpha
	" fmla v2.2d,v10.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v3.2d,v11.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v4.2d,v12.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v5.2d,v13.2d,v6.d[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x2]                               \n\t" // Store column 0 of C
	" str q1, [x2, #16]                          \n\t"
	" str q2, [x2, #32]                          \n\t"
	"                                            \n\t"
	" str q3, [x20]                              \n\t" // Store column 1 of C
	" str q4, [x20, #16]                         \n\t"
	" str q5, [x20, #32]                         \n\t"
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t"
	" dup  v9.2d, xzr                            \n\t"
	" dup  v10.2d, xzr                           \n\t"
	" dup  v11.2d, xzr                           \n\t"
	" dup  v12.2d, xzr                           \n\t"
	" dup  v13.2d, xzr                           \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ(DBETAZEROCOLSTOREDS2)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x21]                              \n\t" // Load column 2 of C
	" ldr q9, [x21, #16]                         \n\t"
	" ldr q10, [x21, #32]                        \n\t"
	"                                            \n\t"
	" ldr q11, [x22]                             \n\t" // Load column 3 of C
	" ldr q12, [x22, #16]                        \n\t"
	" ldr q13, [x22, #32]                        \n\t"
	"                                            \n\t"
	" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(DBETAZEROCOLSTOREDS2)
	"                                            \n\t"
	" fmla v8.2d, v14.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v9.2d, v15.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v10.2d,v16.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v11.2d,v17.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v12.2d,v18.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v13.2d,v19.2d,v6.d[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x21]                              \n\t" // Store column 2 of C
	" str q9, [x21, #16]                         \n\t"
	" str q10, [x21, #32]                        \n\t"
	"                                            \n\t"
	" str q11, [x22]                             \n\t" // Store column 3 of C
	" str q12, [x22, #16]                        \n\t"
	" str q13, [x22, #32]                        \n\t"
	"                                            \n\t"
	" dup  v0.2d, xzr                            \n\t"
	" dup  v1.2d, xzr                            \n\t"
	" dup  v2.2d, xzr                            \n\t"
	" dup  v3.2d, xzr                            \n\t"
	" dup  v4.2d, xzr                            \n\t"
	" dup  v5.2d, xzr                            \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ(DBETAZEROCOLSTOREDS3)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q0, [x23]                              \n\t" // Load column 4 of C
	" ldr q1, [x23, #16]                         \n\t"
	" ldr q2, [x23, #32]                         \n\t"
	"                                            \n\t"
	" ldr q3, [x24]                              \n\t" // Load column 5 of C
	" ldr q4, [x24, #16]                         \n\t"
	" ldr q5, [x24, #32]                         \n\t"
	"                                            \n\t"
	" fmul v0.2d,v0.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v1.2d,v1.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v2.2d,v2.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v3.2d,v3.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v4.2d,v4.2d,v7.d[0]                   \n\t" // Scale by beta
	" fmul v5.2d,v5.2d,v7.d[0]                   \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(DBETAZEROCOLSTOREDS3)
	"                                            \n\t"
	" fmla v0.2d,v20.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v1.2d,v21.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v2.2d,v22.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v3.2d,v23.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v4.2d,v24.2d,v6.d[0]                  \n\t" // Scale by alpha
	" fmla v5.2d,v25.2d,v6.d[0]                  \n\t" // Scale by alpha
	"                                            \n\t"
	" str q0, [x23]                              \n\t" // Store column 4 of C
	" str q1, [x23, #16]                         \n\t"
	" str q2, [x23, #32]                         \n\t"
	"                                            \n\t"
	" str q3, [x24]                              \n\t" // Store column 5 of C
	" str q4, [x24, #16]                         \n\t"
	" str q5, [x24, #32]                         \n\t"
	"                                            \n\t"
	" dup  v8.2d, xzr                            \n\t"
	" dup  v9.2d, xzr                            \n\t"
	" dup  v10.2d, xzr                           \n\t"
	" dup  v11.2d, xzr                           \n\t"
	" dup  v12.2d, xzr                           \n\t"
	" dup  v13.2d, xzr                           \n\t"
	"                                            \n\t"
	" fcmp d7,#0.0                               \n\t"
	BEQ(DBETAZEROCOLSTOREDS4)                          // Taking care of the beta==0 case.
	"                                            \n\t"
	" ldr q8, [x25]                              \n\t" // Load column 6 of C
	" ldr q9, [x25, #16]                         \n\t"
	" ldr q10, [x25, #32]                        \n\t"
	"                                            \n\t"
	" ldr q11, [x26]                             \n\t" // Load column 7 of C
	" ldr q12, [x26, #16]                        \n\t"
	" ldr q13, [x26, #32]                        \n\t"
	"                                            \n\t"
	" fmul v8.2d, v8.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v9.2d, v9.2d, v7.d[0]                 \n\t" // Scale by beta
	" fmul v10.2d,v10.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v11.2d,v11.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v12.2d,v12.2d,v7.d[0]                 \n\t" // Scale by beta
	" fmul v13.2d,v13.2d,v7.d[0]                 \n\t" // Scale by beta
	"                                            \n\t"
	LABEL(DBETAZEROCOLSTOREDS4)
	"                                            \n\t"
	" prfm pldl2keep,[x0]                        \n\t"
	" prfm pldl2keep,[x1]                        \n\t"
	"                                            \n\t"
	" fmla v8.2d, v26.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v9.2d, v27.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v10.2d,v28.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v11.2d,v29.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v12.2d,v30.2d,v6.d[0]                 \n\t" // Scale by alpha
	" fmla v13.2d,v31.2d,v6.d[0]                 \n\t" // Scale by alpha
	"                                            \n\t"
	" str q8, [x25]                              \n\t" // Store column 6 of C
	" str q9, [x25, #16]                         \n\t"
	" str q10, [x25, #32]                        \n\t"
	"                                            \n\t"
	" str q11, [x26]                             \n\t" // Store column 7 of C
	" str q12, [x26, #16]                        \n\t"
	" str q13, [x26, #32]                        \n\t"
	"                                            \n\t"
	// BRANCH(DEND)
	// LABEL(DEND)                                        // Done!
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
	 "x5","x6","x10",
	 "x16","x17","x20",
	 "x21","x22","x23",
	 "x24","x25","x26","x27",
	 "v0","v1","v2",
	 "v3","v4","v5",
	 "v6","v7","v8",
	 "v9","v10","v11",
	 "v12","v13","v14",
	 "v15","v16","v17","v18","v19",
	 "v20","v21","v22","v23",
	 "v24","v25","v26","v27",
	 "v28","v29","v30","v31"
	);

	GEMM_UKR_FLUSH_CT( d );
}

// June 2022, removed unused stubs for ancient 4x4 kernels
