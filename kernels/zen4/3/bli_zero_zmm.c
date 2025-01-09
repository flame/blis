/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include "bli_x86_asm_macros.h"
void bli_zero_zmm()
{

    BEGIN_ASM()
    VXORPD(ZMM(16), ZMM(16), ZMM(16))
    VXORPD(ZMM(17), ZMM(17), ZMM(17))
    VXORPD(ZMM(18), ZMM(18), ZMM(18))
    VXORPD(ZMM(19), ZMM(19), ZMM(19))
    VXORPD(ZMM(20), ZMM(20), ZMM(20))
    VXORPD(ZMM(21), ZMM(21), ZMM(21))
    VXORPD(ZMM(22), ZMM(22), ZMM(22))
    VXORPD(ZMM(23), ZMM(23), ZMM(23))
    VXORPD(ZMM(24), ZMM(24), ZMM(24))
    VXORPD(ZMM(25), ZMM(25), ZMM(25))
    VXORPD(ZMM(26), ZMM(26), ZMM(26))
    VXORPD(ZMM(27), ZMM(27), ZMM(27))
    VXORPD(ZMM(28), ZMM(28), ZMM(28))
    VXORPD(ZMM(29), ZMM(29), ZMM(29))
    VXORPD(ZMM(30), ZMM(30), ZMM(30))
    VXORPD(ZMM(31), ZMM(31), ZMM(31))

    END_ASM
    (
        : // output operands
        : // input operands
        : // register clobber list
          "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
          "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28",
          "zmm29", "zmm30", "zmm31", "memory"
    )
}
