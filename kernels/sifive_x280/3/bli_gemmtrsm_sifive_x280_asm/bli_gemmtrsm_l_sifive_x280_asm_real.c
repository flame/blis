/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, SiFive, Inc.

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

// clang-format off
#ifdef GEMMTRSM

GEMMTRSM(GEMMTRSM_L, PRECISION_CHAR, void)
{
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a10 = a10_;
    const DATATYPE* restrict a11 = a11_;
    const DATATYPE* restrict b01 = b01_;
    const DATATYPE* restrict b11 = b11_;
    DATATYPE* restrict c11 = c11_;

    if (!(1 <= m && m <= PACKMR && 1 <= n && n <= PACKNR))
        return;

    dim_t b11_offset, temp;
    size_t vl;
    __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma": "=r"(vl) : "r"(n), "i"(8*FLT_SIZE));
    
    // Multiply step sizes by data size
    __asm__("slli %0, %0, %1": "+r"(rsc) : "I"(LOG_FLT_SIZE));
    __asm__("slli %0, %0, %1": "+r"(csc) : "I"(LOG_FLT_SIZE));
  
    __asm__("addi %0, %1, %2": "=r"(b11_offset): "r"(m), "I"(-1));
    __asm__("li %0, %1": "=r"(temp): "I"(PACKNR * FLT_SIZE));
    __asm__("mul %0, %0, %1": "+r"(b11_offset): "r"(temp));
    // b11_offset = (m-1)*PACKNR*FLT_SIZE

    __asm__("add %0, %0, %1": "+r"(b11): "r"(b11_offset));
    __asm__(FLT_LOAD " f0, (%0)" : : "r"(alpha));  // TO DO: optimize alpha = 1 case
    switch (m){ // Vector loads from b11 with Duff device, multiplying by alpha
        case 7: __asm__(VLE "  v0, (%0)": : "r"(b11)); 
                __asm__("vfmul.vf  v0,  v0, f0");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(-PACKNR * FLT_SIZE));
        case 6: __asm__(VLE "  v4, (%0)": : "r"(b11));
                __asm__("vfmul.vf  v4,  v4, f0");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(-PACKNR * FLT_SIZE));
        case 5: __asm__(VLE "  v8, (%0)": : "r"(b11));
                __asm__("vfmul.vf  v8,  v8, f0");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(-PACKNR * FLT_SIZE));
        case 4: __asm__(VLE " v12, (%0)": : "r"(b11));
                 __asm__("vfmul.vf v12, v12, f0");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(-PACKNR * FLT_SIZE));
        case 3: __asm__(VLE " v16, (%0)": : "r"(b11));
                __asm__("vfmul.vf v16, v16, f0");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(-PACKNR * FLT_SIZE));
        case 2: __asm__(VLE " v20, (%0)": : "r"(b11));
                __asm__("vfmul.vf v20, v20, f0");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(-PACKNR * FLT_SIZE));
        case 1: __asm__(VLE " v24, (%0)": : "r"(b11));
                __asm__("vfmul.vf v24, v24, f0");
                // no sub of b11 on final entry
    }
    // b11 now reset to original value
    //  v0 = row 6 of b11
    //  v4 = row 5 of b11
    //  v8 = row 4 of b11
    // v12 = row 3 of b11
    // v16 = row 2 of b11
    // v20 = row 1 of b11
    // v24 = row 0 of b11

    // GEMM: B11 := alpha * B11 - A10 * B01
    for (dim_t i = 0; i < k; i++){
        __asm__(VLE " v28, (%0)": : "r"(b01)); // kth row of b01
        switch (m){
            case 7: __asm__(FLT_LOAD " f6, %0(%1)" : : "I"(6*FLT_SIZE), "r"(a10));
                    __asm__("vfnmsac.vf  v0, f6, v28");
            case 6: __asm__(FLT_LOAD " f5, %0(%1)" : : "I"(5*FLT_SIZE), "r"(a10));
                    __asm__("vfnmsac.vf  v4, f5, v28");
            case 5: __asm__(FLT_LOAD " f4, %0(%1)" : : "I"(4*FLT_SIZE), "r"(a10));
                    __asm__("vfnmsac.vf  v8, f4, v28");
            case 4: __asm__(FLT_LOAD " f3, %0(%1)" : : "I"(3*FLT_SIZE), "r"(a10));
                    __asm__("vfnmsac.vf v12, f3, v28");
            case 3: __asm__(FLT_LOAD " f2, %0(%1)" : : "I"(2*FLT_SIZE), "r"(a10));
                    __asm__("vfnmsac.vf v16, f2, v28");
            case 2: __asm__(FLT_LOAD " f1, %0(%1)" : : "I"(1*FLT_SIZE), "r"(a10));
                    __asm__("vfnmsac.vf v20, f1, v28");
            case 1: __asm__(FLT_LOAD " f0, %0(%1)" : : "I"(0*FLT_SIZE), "r"(a10));
                 __asm__("vfnmsac.vf v24, f0, v28");
        }
        __asm__("addi %0, %0, %1": "+r"(a10): "I"(PACKMR * FLT_SIZE));
        __asm__("addi %0, %0, %1": "+r"(b01): "I"(PACKNR * FLT_SIZE));
    }
    // TRSM: B11 := inv(A11) * B11
    // TO DO: Investigate code size reduction (loop rerolling)

    // Row 0
    __asm__(FLT_LOAD " f0,  %0(%1)": : "I"(0*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v24, v24, f0");
    __asm__(VSE " v24, (%0)": : "r"(b11));
    __asm__(VSSE " v24, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 1) return;

    switch (m){
        case 7: __asm__(FLT_LOAD " f6, %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v0, f6, v24");
        case 6: __asm__(FLT_LOAD " f5, %0(%1)": : "I"(5*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v4, f5, v24");
        case 5: __asm__(FLT_LOAD " f4, %0(%1)": : "I"(4*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v8, f4, v24");
        case 4: __asm__(FLT_LOAD " f3, %0(%1)": : "I"(3*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf v12, f3, v24");
        case 3: __asm__(FLT_LOAD " f2, %0(%1)": : "I"(2*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf v16, f2, v24");
        case 2: __asm__(FLT_LOAD " f1, %0(%1)": : "I"(1*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf v20, f1, v24");
    }
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(PACKNR * FLT_SIZE));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(rsc));

    // Row 1
    __asm__(FLT_LOAD " f1,  %0(%1)": : "I"(1*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v20, v20, f1");
    __asm__(VSE " v20, (%0)": : "r"(b11));
    __asm__(VSSE " v20, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 2) return;
    
    switch (m){
        case 7: __asm__(FLT_LOAD " f6, %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v0, f6, v20");
        case 6: __asm__(FLT_LOAD " f5, %0(%1)": : "I"(5*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v4, f5, v20");
        case 5: __asm__(FLT_LOAD " f4, %0(%1)": : "I"(4*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v8, f4, v20");
        case 4: __asm__(FLT_LOAD " f3, %0(%1)": : "I"(3*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf v12, f3, v20");
        case 3: __asm__(FLT_LOAD " f2, %0(%1)": : "I"(2*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf v16, f2, v20");
    }
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(PACKNR * FLT_SIZE));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(rsc));

    // Row 2
    __asm__(FLT_LOAD " f2,  %0(%1)": : "I"(2*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v16, v16, f2");
    __asm__(VSE " v16, (%0)": : "r"(b11));
    __asm__(VSSE " v16, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 3) return;
    
    switch (m){
        case 7: __asm__(FLT_LOAD " f6, %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v0, f6, v16");
        case 6: __asm__(FLT_LOAD " f5, %0(%1)": : "I"(5*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v4, f5, v16");
        case 5: __asm__(FLT_LOAD " f4, %0(%1)": : "I"(4*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v8, f4, v16");
        case 4: __asm__(FLT_LOAD " f3, %0(%1)": : "I"(3*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf v12, f3, v16");
    }
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(PACKNR * FLT_SIZE));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(rsc));
    
    // Row 3
    __asm__(FLT_LOAD " f3,  %0(%1)": : "I"(3*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v12, v12, f3");
    __asm__(VSE " v12, (%0)": : "r"(b11));
    __asm__(VSSE " v12, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 4) return;
  
    switch (m){
        case 7: __asm__(FLT_LOAD " f6, %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v0, f6, v12");
        case 6: __asm__(FLT_LOAD " f5, %0(%1)": : "I"(5*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v4, f5, v12");
        case 5: __asm__(FLT_LOAD " f4, %0(%1)": : "I"(4*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v8, f4, v12");
    }
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(PACKNR * FLT_SIZE));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(rsc));
    
    // Row 4
    __asm__(FLT_LOAD " f4,  %0(%1)": : "I"(4*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v8, v8, f4");
    __asm__(VSE " v8, (%0)": : "r"(b11));
    __asm__(VSSE " v8, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 5) return;
    
    switch (m){
        case 7: __asm__(FLT_LOAD " f6, %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v0, f6, v8");
        case 6: __asm__(FLT_LOAD " f5, %0(%1)": : "I"(5*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf  v4, f5, v8");
    }
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(PACKNR * FLT_SIZE));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(rsc));
    
    // Row 5
    __asm__(FLT_LOAD " f5,  %0(%1)": : "I"(5*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v4, v4, f5");
    __asm__(VSE " v4, (%0)": : "r"(b11));
    __asm__(VSSE " v4, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 6) return;
    
    __asm__(FLT_LOAD " f6,  %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
    __asm__("vfnmsac.vf v0, f6, v4");
    
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(PACKNR * FLT_SIZE));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(rsc));
    
    // Row 6
    __asm__(FLT_LOAD " f6,  %0(%1)": : "I"(6*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf v0, v0, f6");
    __asm__(VSE " v0, (%0)": : "r"(b11));
    __asm__(VSSE " v0, (%0), %1": : "r"(c11), "r"(csc));
}
#endif
