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

// macros to emit complex multiplication
// caveat: the destination registers cannot overlap the source registers!
// rd = rs1 * rs2
#define cmul(rd_r, rd_i, rs1_r, rs1_i, rs2_r, rs2_i) \
  \
  __asm__(FMUL#rd_r", "#rs1_r", "#rs2_r);\
  __asm__(FMUL#rd_i", "#rs1_r", "#rs2_i);\
  __asm__(FNMSUB#rd_r", "#rs1_i", "#rs2_i", "#rd_r);\
  __asm__(FMADD#rd_i", "#rs1_i", "#rs2_r", "#rd_i)

// vd = vs2 * f[rs1]
#define vcmul_vf(vd_r, vd_i, vs2_r, vs2_i, rs1_r, rs1_i) \
  \
  __asm__("vfmul.vf "#vd_r", "#vs2_r", "#rs1_r);\
  __asm__("vfmul.vf "#vd_i", "#vs2_r", "#rs1_i);\
  __asm__("vfnmsac.vf "#vd_r", "#rs1_i", "#vs2_i);\
  __asm__("vfmacc.vf "#vd_i", "#rs1_r", "#vs2_i)

#define vcmul_vf2(vd_r, vd_i, vs2_r, vs2_i, rs1_r, rs1_i) \
  \
  __asm__("vfmul.vf "#vd_r", "#vs2_r", %0" : : "f"(rs1_r));\
  __asm__("vfmul.vf "#vd_i", "#vs2_r", %0" : : "f"(rs1_i));\
  __asm__("vfnmsac.vf "#vd_r", %0, "#vs2_i : : "f"(rs1_i));\
  __asm__("vfmacc.vf "#vd_i", %0, "#vs2_i : : "f"(rs1_r))

// vd = conj(vs2) * f[rs1]
#define vcmul_vf_conj(vd_r, vd_i, vs2_r, vs2_i, rs1_r, rs1_i) \
  \
  __asm__("vfmul.vf "#vd_r", "#vs2_r", "#rs1_r);\
  __asm__("vfmul.vf "#vd_i", "#vs2_r", "#rs1_i);\
  __asm__("vfmacc.vf "#vd_r", "#rs1_i", "#vs2_i);\
  __asm__("vfnmsac.vf "#vd_i", "#rs1_r", "#vs2_i)

#define vcmul_vf_conj2(vd_r, vd_i, vs2_r, vs2_i, rs1_r, rs1_i) \
  \
  __asm__("vfmul.vf "#vd_r", "#vs2_r", %0" : : "f"(rs1_r));\
  __asm__("vfmul.vf "#vd_i", "#vs2_r", %0" : : "f"(rs1_i));\
  __asm__("vfmacc.vf "#vd_r", %0, "#vs2_i : : "f"(rs1_i));\
  __asm__("vfnmsac.vf "#vd_i", %0, "#vs2_i : : "f"(rs1_r))

// vd += vs2 * f[rs1]
#define vcmacc_vf(vd_r, vd_i, rs1_r, rs1_i, vs2_r, vs2_i) \
  \
  __asm__("vfmacc.vf "#vd_r", "#rs1_r", "#vs2_r);\
  __asm__("vfmacc.vf "#vd_i", "#rs1_i", "#vs2_r);\
  __asm__("vfnmsac.vf "#vd_r", "#rs1_i", "#vs2_i);\
  __asm__("vfmacc.vf "#vd_i", "#rs1_r", "#vs2_i)

#define vcmacc_vf2(vd_r, vd_i, rs1_r, rs1_i, vs2_r, vs2_i) \
  \
  __asm__("vfmacc.vf "#vd_r", %0, "#vs2_r : : "f"(rs1_r));\
  __asm__("vfmacc.vf "#vd_i", %0, "#vs2_r : : "f"(rs1_i));\
  __asm__("vfnmsac.vf "#vd_r", %0, "#vs2_i : : "f"(rs1_i));\
  __asm__("vfmacc.vf "#vd_i", %0, "#vs2_i : : "f"(rs1_r))

// vd += conj(vs2) * f[rs1]
#define vcmacc_vf_conj(vd_r, vd_i, rs1_r, rs1_i, vs2_r, vs2_i) \
  \
  __asm__("vfmacc.vf "#vd_r", "#rs1_r", "#vs2_r);\
  __asm__("vfmacc.vf "#vd_i", "#rs1_i", "#vs2_r);\
  __asm__("vfmacc.vf "#vd_r", "#rs1_i", "#vs2_i);\
  __asm__("vfnmsac.vf "#vd_i", "#rs1_r", "#vs2_i)

// vd -= vs2 * f[rs1]
#define vcnmsac_vf(vd_r, vd_i, rs1_r, rs1_i, vs2_r, vs2_i) \
  \
  __asm__("vfnmsac.vf "#vd_r", "#rs1_r", "#vs2_r);\
  __asm__("vfnmsac.vf "#vd_i", "#rs1_i", "#vs2_r);\
  __asm__("vfmacc.vf "#vd_r", "#rs1_i", "#vs2_i);\
  __asm__("vfnmsac.vf "#vd_i", "#rs1_r", "#vs2_i)

// vd = vs2 * vs1
#define vcmul_vv(vd_r, vd_i, vs2_r, vs2_i, vs1_r, vs1_i) \
  \
  __asm__("vfmul.vv "#vd_r", "#vs2_r", "#vs1_r);\
  __asm__("vfmul.vv "#vd_i", "#vs2_r", "#vs1_i);\
  __asm__("vfnmsac.vv "#vd_r", "#vs2_i", "#vs1_i);\
  __asm__("vfmacc.vv "#vd_i", "#vs2_i", "#vs1_r)

// vd = vs2 * conj(vs1)
#define vcmul_vv_conj(vd_r, vd_i, vs2_r, vs2_i, vs1_r, vs1_i) \
  \
  __asm__("vfmul.vv "#vd_r", "#vs2_r", "#vs1_r);\
  __asm__("vfmul.vv "#vd_i", "#vs2_r", "#vs1_i);\
  __asm__("vfmacc.vv "#vd_r", "#vs2_i", "#vs1_i);\
  __asm__("vfmsac.vv "#vd_i", "#vs2_i", "#vs1_r)

// vd += vs2 * vs1
#define vcmacc_vv(vd_r, vd_i, vs2_r, vs2_i, vs1_r, vs1_i) \
  \
  __asm__("vfmacc.vv "#vd_r", "#vs2_r", "#vs1_r);\
  __asm__("vfmacc.vv "#vd_i", "#vs2_r", "#vs1_i);\
  __asm__("vfnmsac.vv "#vd_r", "#vs2_i", "#vs1_i);\
  __asm__("vfmacc.vv "#vd_i", "#vs2_i", "#vs1_r)

// vd += vs2 * conj(vs1)
#define vcmacc_vv_conj(vd_r, vd_i, vs2_r, vs2_i, vs1_r, vs1_i) \
  \
  __asm__("vfmacc.vv "#vd_r", "#vs2_r", "#vs1_r);\
  __asm__("vfnmsac.vv "#vd_i", "#vs2_r", "#vs1_i);\
  __asm__("vfmacc.vv "#vd_r", "#vs2_i", "#vs1_i);\
  __asm__("vfmacc.vv "#vd_i", "#vs2_i", "#vs1_r)

