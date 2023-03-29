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

GEMMTRSM(GEMMTRSM_U, PRECISION_CHAR, void)
{
    const DATATYPE* restrict alpha = alpha_;
    const DATATYPE* restrict a12 = a12_;
    const DATATYPE* restrict a11 = a11_;
    const DATATYPE* restrict b21 = b21_;
    const DATATYPE* restrict b11 = b11_;
    DATATYPE* restrict c11 = c11_;

    if (!(1 <= m && m <= PACKMR && 1 <= n && n <= PACKNR))
        return;
    
    dim_t m_2sz, a11_offset, c11_offset, temp;
    size_t vl;
    __asm__ volatile("vsetvli %0, %1, e%2, m4, ta, ma": "=r"(vl) : "r"(n), "i"(8*FLT_SIZE));

    // Multiply step sizes by data size
    __asm__("slli %0, %0, %1": "+r"(rsc) : "I"(LOG_FLT_SIZE+1));
    __asm__("slli %0, %0, %1": "+r"(csc) : "I"(LOG_FLT_SIZE+1));
    __asm__("slli %0, %1, %2": "=r"(m_2sz) : "r"(m), "I"(LOG_FLT_SIZE+1));
    
    __asm__("li %0, %1": "=r"(temp): "I"(2*(PACKMR+1)*FLT_SIZE)); 
    __asm__("mul %0, %1, %2": "=r"(a11_offset) : "r"(m), "r"(temp));
    __asm__("addi %0, %0, %1": "+r"(a11_offset) : "I"(-2*PACKMR * FLT_SIZE)); 
    __asm__("mul %0, %1, %2": "=r"(c11_offset): "r"(m), "r"(rsc));
    __asm__("sub %0, %0, %1": "+r"(c11_offset): "r"(rsc));   
    // a11_offset = 2*(PACKMR*(m-1)+m)*sz = m*2*(PACKMR+1)*FLT_SIZE - 2*PACKMR*FLT_SIZE
    // c11_offset = rsc*(m-1)*sz
    
    __asm__(FLT_LOAD " f0, %0(%1)" : : "I"(0*FLT_SIZE), "r"(alpha)); // f0 = a
    __asm__(FLT_LOAD " f1, %0(%1)" : : "I"(1*FLT_SIZE), "r"(alpha)); // f1 = b
    switch (m){ // Vector loads from b11 with Duff device, multiplying by alpha
        case 3: __asm__(VLSEG "     v24, (%0)": : "r"(b11)); // v24 = c, v28 = d
                __asm__("vfmul.vf    v0, v24,  f0"); // a*c
                __asm__("vfmul.vf    v4, v28,  f0"); // a*d
                __asm__("vfnmsac.vf  v0,  f1, v28"); // a*c - b*d
                __asm__("vfmacc.vf   v4,  f1, v24"); // a*d + b*c
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(2 * PACKNR * FLT_SIZE));
        case 2: __asm__(VLSEG "     v24, (%0)": : "r"(b11));
                __asm__("vfmul.vf    v8, v24,  f0");
                __asm__("vfmul.vf   v12, v28,  f0");
                __asm__("vfnmsac.vf  v8,  f1, v28");
                __asm__("vfmacc.vf  v12,  f1, v24");
                __asm__("addi %0, %0, %1": "+r"(b11): "I"(2 * PACKNR * FLT_SIZE));
        case 1: __asm__(VLSEG "     v24, (%0)": : "r"(b11));
                __asm__("vfmul.vf   v16, v24,  f0");
                __asm__("vfmul.vf   v20, v28,  f0");
                __asm__("vfnmsac.vf v16,  f1, v28");
                __asm__("vfmacc.vf  v20,  f1, v24");
                // no sub of b11 on final entry
    }
    // b11 now positioned at start of last row
    // v24 = row 0 from bottom (bottom row)
    // v20 = row 1 from bottom
    // v16 = row 2 from bottom
    // v12 = row 3 from bottom
    //  v8 = row 4 from bottom
    //  v4 = row 5 from bottom
    //  v0 = row 6 from bottom

    // GEMM: B11 := alpha * B11 - A12 * B21
    __asm__("add %0, %0, %1": "+r"(a12): "r"(m_2sz));
    for (dim_t i = 0; i < k; i++){
        __asm__(VLSEG " v24, (%0)": : "r"(b21)); // kth row of b01: v24 = c, v28 = d
        switch (m){
            case 3: __asm__(FLT_LOAD " f5, %0(%1)" : : "I"(-6*FLT_SIZE), "r"(a12)); // f5 = a
                    __asm__(FLT_LOAD " f4, %0(%1)" : : "I"(-5*FLT_SIZE), "r"(a12)); // f4 = b
                    __asm__("vfnmsac.vf  v0, f5, v24");               // a*c
                    __asm__("vfnmsac.vf  v4, f5, v28");               // a*d
                    __asm__("vfmacc.vf   v0, f4, v28");               // a*c - b*d
                    __asm__("vfnmsac.vf  v4, f4, v24");               // a*d + b*c
            case 2: __asm__(FLT_LOAD " f3, %0(%1)" : : "I"(-4*FLT_SIZE), "r"(a12));
                    __asm__(FLT_LOAD " f2, %0(%1)" : : "I"(-3*FLT_SIZE), "r"(a12));
                    __asm__("vfnmsac.vf  v8, f3, v24");
                    __asm__("vfnmsac.vf v12, f3, v28");
                    __asm__("vfmacc.vf   v8, f2, v28");
                    __asm__("vfnmsac.vf v12, f2, v24");
            case 1: __asm__(FLT_LOAD " f1, %0(%1)" : : "I"(-2*FLT_SIZE), "r"(a12));
                    __asm__(FLT_LOAD " f0, %0(%1)" : : "I"(-1*FLT_SIZE), "r"(a12));
                    __asm__("vfnmsac.vf v16, f1, v24");
                    __asm__("vfnmsac.vf v20, f1, v28");
                    __asm__("vfmacc.vf  v16, f0, v28");
                    __asm__("vfnmsac.vf v20, f0, v24");
        }
        __asm__("addi %0, %0, %1": "+r"(a12): "I"(2 * PACKMR * FLT_SIZE));
        __asm__("addi %0, %0, %1": "+r"(b21): "I"(2 * PACKNR * FLT_SIZE));
    }

    // TRSM: B11 := inv(A11) * B11
    // Move a11 to end of array and c11 to first entry in last row
    __asm__("add %0, %0, %1": "+r"(a11): "r"(a11_offset));
    __asm__("add %0, %0, %1": "+r"(c11): "r"(c11_offset));

    // Row 0 from bottom (bottom row)
    __asm__(FLT_LOAD " f0,  %0(%1)": : "I"(-2*FLT_SIZE), "r"(a11)); // f0 = a
    __asm__(FLT_LOAD " f1,  %0(%1)": : "I"(-1*FLT_SIZE), "r"(a11)); // f1 = b
    __asm__("vfmul.vf  v24, v20,  f1"); // b*d
    __asm__("vfmul.vf  v28, v16,  f1"); // b*c
    __asm__("vfmsub.vf v16,  f0, v24"); // a*c - b*d
    __asm__("vfmadd.vf v20,  f0, v28"); // a*d + b*c

    __asm__(VSSEG " v16, (%0)": : "r"(b11));
    __asm__(VSSSEG " v16, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 1) return;
    
    switch (m){
        case 3: __asm__(FLT_LOAD " f4, %0(%1)": : "I"(-6*FLT_SIZE), "r"(a11)); // f4 = a
                __asm__(FLT_LOAD " f5, %0(%1)": : "I"(-5*FLT_SIZE), "r"(a11)); // f5 = b
                __asm__("vfnmsac.vf   v0, f4, v16"); // - a*c
                __asm__("vfnmsac.vf   v4, f4, v20"); // - a*d
                __asm__("vfmacc.vf    v0, f5, v20"); // - a*c + b*d
                __asm__("vfnmsac.vf   v4, f5, v16"); // - a*d - b*c
        case 2: __asm__(FLT_LOAD " f2, %0(%1)": : "I"(-4*FLT_SIZE), "r"(a11));
                __asm__(FLT_LOAD " f3, %0(%1)": : "I"(-3*FLT_SIZE), "r"(a11));
                __asm__("vfnmsac.vf   v8, f2, v16");
                __asm__("vfnmsac.vf  v12, f2, v20");
                __asm__("vfmacc.vf    v8, f3, v20");
                __asm__("vfnmsac.vf  v12, f3, v16");
    }
    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(-2 * PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(-2 * PACKNR * FLT_SIZE));
    __asm__("sub %0, %0, %1": "+r"(c11): "r"(rsc));

    // Row 1 from bottom
    __asm__(FLT_LOAD " f2,  %0(%1)": : "I"(-4*FLT_SIZE), "r"(a11)); // f0 = a
    __asm__(FLT_LOAD " f3,  %0(%1)": : "I"(-3*FLT_SIZE), "r"(a11)); // f1 = b
    __asm__("vfmul.vf  v24, v12,  f3"); // b*d
    __asm__("vfmul.vf  v28,  v8,  f3"); // b*c
    __asm__("vfmsub.vf  v8,  f2, v24"); // a*c - b*d
    __asm__("vfmadd.vf v12,  f2, v28"); // a*d + b*c
    
    __asm__(VSSEG " v8, (%0)": : "r"(b11));
    __asm__(VSSSEG " v8, (%0), %1": : "r"(c11), "r"(csc));
    if (m == 2) return;
    
    __asm__(FLT_LOAD " f4, %0(%1)": : "I"(-6*FLT_SIZE), "r"(a11));
    __asm__(FLT_LOAD " f5, %0(%1)": : "I"(-5*FLT_SIZE), "r"(a11));
    __asm__("vfnmsac.vf   v0, f4,  v8");
    __asm__("vfnmsac.vf   v4, f4, v12");
    __asm__("vfmacc.vf    v0, f5, v12");
    __asm__("vfnmsac.vf   v4, f5,  v8");

    // Pointer bumps
    __asm__("addi %0, %0, %1": "+r"(a11): "I"(-2 * PACKMR * FLT_SIZE));
    __asm__("addi %0, %0, %1": "+r"(b11): "I"(-2 * PACKNR * FLT_SIZE));
    __asm__("sub %0, %0, %1": "+r"(c11): "r"(rsc));

    // Row 2 from bottom
    __asm__(FLT_LOAD " f4, %0(%1)": : "I"(-6*FLT_SIZE), "r"(a11));
    __asm__(FLT_LOAD " f5, %0(%1)": : "I"(-5*FLT_SIZE), "r"(a11));
    __asm__("vfmul.vf  v24, v4,  f5"); // b*d
    __asm__("vfmul.vf  v28, v0,  f5"); // b*c
    __asm__("vfmsub.vf  v0, f4, v24"); // a*c - b*d
    __asm__("vfmadd.vf  v4, f4, v28"); // a*d + b*c

    __asm__(VSSEG " v0, (%0)": : "r"(b11));
    __asm__(VSSSEG " v0, (%0), %1": : "r"(c11), "r"(csc));
    
}
#endif
