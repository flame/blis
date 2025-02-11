/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023 - 2024, Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LPGEMM_BLKSZ_MAP_H
#define LPGEMM_BLKSZ_MAP_H

// The XMACRO follows the format ID,MC,NC,KC,MR,NR,PACKA_RS,PACKA_CS,PACKB_RS,PACKB_CS:
// ID = One of the AOCL_OPERATION_TYPE enum.

#define LPGEMM_BLKSZ_MAP_ZEN4 \
	XMACRO(U8S8S32OS32, 144, 1024, 2048, 6, 64, 4, 24, 4*64, 64) \
	XMACRO(F32F32F32OF32, 192, 8064, 512, 6, 64, 1, 6, 64, 1) \
	XMACRO(BF16BF16F32OF32, 144, 1024, 4096, 6, 64, 0, 0, 2*64, 64/2) \
	XMACRO(BF16S4F32OF32, 144, 1024, 4096, 6, 64, 0, 0, 2*64, 64/2) \
	XMACRO(S8S8S32OS32, 144, 1024, 2048, 6, 64, 4, 24, 4*64, 64) \
	XMACRO(U8S4S32OS32, 144, 1024, 2048, 6, 64, 4, 24, 4*64, 64) \
	XMACRO(F32OBF16, 144, 1024, 4096, 6, 64, 0, 0, 2*64, 64/2) \

#define LPGEMM_BLKSZ_MAP_ZEN \
	XMACRO(U8S8S32OS32, 144, 1024, 2048, 6, 64, 4, 24, 4*64, 64) \
	XMACRO(F32F32F32OF32, 144, 8160, 512, 6, 16, 1, 6, 16, 1) \
	XMACRO(BF16BF16F32OF32, 144, 1024, 2048, 6, 64, 0, 0, 2*64, 64/2) \
	XMACRO(S8S8S32OS32, 144, 1024, 2048, 6, 64, 4, 24, 4*64, 64) \
	XMACRO(U8S4S32OS32, 144, 1024, 2048, 6, 64, 4, 24, 4*64, 64) \
	XMACRO(BF16S4F32OF32, 144, 1024, 2048, 6, 64, 0, 0, 2*64, 64/2) \
	XMACRO(F32OBF16, 144, 1024, 2048, 6, 64, 0, 0, 2*64, 64/2) \

#define LPGEMM_BLKSZ_UPD_MAP_ZEN4_TO_ZEN \
	XMACRO(F32F32F32OF32, 144, 8160, 512, 6, 16, 1, 6, 16, 1) \

// The STMACRO follows the format MT, NT, KT which are SUP switch thresholds.
// ID = One of the AOCL_OPERATION_TYPE enum.
#define LPGEMM_SUP_THRES_MAP_ZEN4 \
	STMACRO(F32F32F32OF32, 682, 512, 240) \

#define LPGEMM_SUP_THRES_MAP_ZEN \
	STMACRO(F32F32F32OF32, 512, 200, 240) \

#define LPGEMM_SUP_THRES_UPD_MAP_ZEN4_TO_ZEN \
	STMACRO(F32F32F32OF32, 512, 200, 240) \

// Block sizes used only elementwise ops APIs
#define LPGEMM_ELTWISE_OPS_BLKSZ_MAP_ZEN4 \
	XMACRO(BF16OF32, 144, 1024, 2048, 6, 64) \
	XMACRO(F32OF32, 144, 1024, 2048, 6, 64) \

#define LPGEMM_ELTWISE_OPS_BLKSZ_MAP_ZEN

#endif //LPGEMM_BLKSZ_MAP_H
