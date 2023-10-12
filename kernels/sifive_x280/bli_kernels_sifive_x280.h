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

// Level 1
ADDV_KER_PROT(float,        s, addv_sifive_x280_intr)
ADDV_KER_PROT(double,       d, addv_sifive_x280_intr)
ADDV_KER_PROT(scomplex,     c, addv_sifive_x280_intr)
ADDV_KER_PROT(dcomplex,     z, addv_sifive_x280_intr)

AMAXV_KER_PROT(float,       s, amaxv_sifive_x280_asm)
AMAXV_KER_PROT(double,      d, amaxv_sifive_x280_asm)
AMAXV_KER_PROT(scomplex,    c, amaxv_sifive_x280_asm)
AMAXV_KER_PROT(dcomplex,    z, amaxv_sifive_x280_asm)

AXPBYV_KER_PROT(float,      s, axpbyv_sifive_x280_intr)
AXPBYV_KER_PROT(double,     d, axpbyv_sifive_x280_intr)
AXPBYV_KER_PROT(scomplex,   c, axpbyv_sifive_x280_intr)
AXPBYV_KER_PROT(dcomplex,   z, axpbyv_sifive_x280_intr)

AXPYV_KER_PROT(float,       s, axpyv_sifive_x280_intr)
AXPYV_KER_PROT(double,      d, axpyv_sifive_x280_intr)
AXPYV_KER_PROT(scomplex,    c, axpyv_sifive_x280_intr)
AXPYV_KER_PROT(dcomplex,    z, axpyv_sifive_x280_intr)

COPYV_KER_PROT(float,       s, copyv_sifive_x280_asm)
COPYV_KER_PROT(double,      d, copyv_sifive_x280_asm)
COPYV_KER_PROT(scomplex,    c, copyv_sifive_x280_asm)
COPYV_KER_PROT(dcomplex,    z, copyv_sifive_x280_asm)

DOTV_KER_PROT(float,        s, dotv_sifive_x280_intr)
DOTV_KER_PROT(double,       d, dotv_sifive_x280_intr)
DOTV_KER_PROT(scomplex,     c, dotv_sifive_x280_intr)
DOTV_KER_PROT(dcomplex,     z, dotv_sifive_x280_intr)

DOTXV_KER_PROT(float,       s, dotxv_sifive_x280_intr)
DOTXV_KER_PROT(double,      d, dotxv_sifive_x280_intr)
DOTXV_KER_PROT(scomplex,    c, dotxv_sifive_x280_intr)
DOTXV_KER_PROT(dcomplex,    z, dotxv_sifive_x280_intr)

INVERTV_KER_PROT(float,     s, invertv_sifive_x280_asm)
INVERTV_KER_PROT(double,    d, invertv_sifive_x280_asm)
INVERTV_KER_PROT(scomplex,  c, invertv_sifive_x280_asm)
INVERTV_KER_PROT(dcomplex,  z, invertv_sifive_x280_asm)

INVSCALV_KER_PROT(float,    s, invscalv_sifive_x280_asm)
INVSCALV_KER_PROT(double,   d, invscalv_sifive_x280_asm)
INVSCALV_KER_PROT(scomplex, c, invscalv_sifive_x280_asm)
INVSCALV_KER_PROT(dcomplex, z, invscalv_sifive_x280_asm)

SCAL2V_KER_PROT(float,      s, scal2v_sifive_x280_intr)
SCAL2V_KER_PROT(double,     d, scal2v_sifive_x280_intr)
SCAL2V_KER_PROT(scomplex,   c, scal2v_sifive_x280_intr)
SCAL2V_KER_PROT(dcomplex,   z, scal2v_sifive_x280_intr)

SCALV_KER_PROT(float,       s, scalv_sifive_x280_intr)
SCALV_KER_PROT(double,      d, scalv_sifive_x280_intr)
SCALV_KER_PROT(scomplex,    c, scalv_sifive_x280_intr)
SCALV_KER_PROT(dcomplex,    z, scalv_sifive_x280_intr)

SETV_KER_PROT(float,        s, setv_sifive_x280_asm)
SETV_KER_PROT(double,       d, setv_sifive_x280_asm)
SETV_KER_PROT(scomplex,     c, setv_sifive_x280_asm)
SETV_KER_PROT(dcomplex,     z, setv_sifive_x280_asm)

SUBV_KER_PROT(float,        s, subv_sifive_x280_intr)
SUBV_KER_PROT(double,       d, subv_sifive_x280_intr)
SUBV_KER_PROT(scomplex,     c, subv_sifive_x280_intr)
SUBV_KER_PROT(dcomplex,     z, subv_sifive_x280_intr)

SWAPV_KER_PROT(float,       s, swapv_sifive_x280_asm)
SWAPV_KER_PROT(double,      d, swapv_sifive_x280_asm)
SWAPV_KER_PROT(scomplex,    c, swapv_sifive_x280_asm)
SWAPV_KER_PROT(dcomplex,    z, swapv_sifive_x280_asm)

XPBYV_KER_PROT(float,       s, xpbyv_sifive_x280_intr)
XPBYV_KER_PROT(double,      d, xpbyv_sifive_x280_intr)
XPBYV_KER_PROT(scomplex,    c, xpbyv_sifive_x280_intr)
XPBYV_KER_PROT(dcomplex,    z, xpbyv_sifive_x280_intr)

// Level 1f
AXPY2V_KER_PROT(float,      s, axpy2v_sifive_x280_intr)
AXPY2V_KER_PROT(double,     d, axpy2v_sifive_x280_intr)
AXPY2V_KER_PROT(scomplex,   c, axpy2v_sifive_x280_intr)
AXPY2V_KER_PROT(dcomplex,   z, axpy2v_sifive_x280_intr)

AXPYF_KER_PROT(float,       s, axpyf_sifive_x280_asm)
AXPYF_KER_PROT(double,      d, axpyf_sifive_x280_asm)
AXPYF_KER_PROT(scomplex,    c, axpyf_sifive_x280_asm)
AXPYF_KER_PROT(dcomplex,    z, axpyf_sifive_x280_asm)

DOTXF_KER_PROT(float,       s, dotxf_sifive_x280_asm)
DOTXF_KER_PROT(double,      d, dotxf_sifive_x280_asm)
DOTXF_KER_PROT(scomplex,    c, dotxf_sifive_x280_asm)
DOTXF_KER_PROT(dcomplex,    z, dotxf_sifive_x280_asm)

DOTAXPYV_KER_PROT(float,    s, dotaxpyv_sifive_x280_intr)
DOTAXPYV_KER_PROT(double,   d, dotaxpyv_sifive_x280_intr)
DOTAXPYV_KER_PROT(scomplex, c, dotaxpyv_sifive_x280_intr)
DOTAXPYV_KER_PROT(dcomplex, z, dotaxpyv_sifive_x280_intr)

DOTXAXPYF_KER_PROT(float,   s, dotxaxpyf_sifive_x280_asm)
DOTXAXPYF_KER_PROT(double,  d, dotxaxpyf_sifive_x280_asm)
DOTXAXPYF_KER_PROT(scomplex,c, dotxaxpyf_sifive_x280_asm)
DOTXAXPYF_KER_PROT(dcomplex,z, dotxaxpyf_sifive_x280_asm)

// Level 3
GEMM_UKR_PROT(float,        s, gemm_sifive_x280_asm_7m4)
GEMM_UKR_PROT(double,       d, gemm_sifive_x280_asm_7m4)
GEMM_UKR_PROT(scomplex,     c, gemm_sifive_x280_asm_3m4)
GEMM_UKR_PROT(dcomplex,     z, gemm_sifive_x280_asm_3m4)

GEMMTRSM_UKR_PROT(float,    s, gemmtrsm_l_sifive_x280_asm)
GEMMTRSM_UKR_PROT(double,   d, gemmtrsm_l_sifive_x280_asm)
GEMMTRSM_UKR_PROT(scomplex, c, gemmtrsm_l_sifive_x280_asm)
GEMMTRSM_UKR_PROT(dcomplex, z, gemmtrsm_l_sifive_x280_asm)
GEMMTRSM_UKR_PROT(float,    s, gemmtrsm_u_sifive_x280_asm)
GEMMTRSM_UKR_PROT(double,   d, gemmtrsm_u_sifive_x280_asm)
GEMMTRSM_UKR_PROT(scomplex, c, gemmtrsm_u_sifive_x280_asm)
GEMMTRSM_UKR_PROT(dcomplex, z, gemmtrsm_u_sifive_x280_asm)
