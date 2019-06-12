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


#define VSZEROOUT \
 "xxlxor           %%vs0, %%vs0, %%vs0              \n\t" \
 "xxlxor           %%vs1, %%vs1, %%vs1              \n\t" \
 "xxlxor           %%vs2, %%vs2, %%vs2              \n\t" \
 "xxlxor           %%vs3, %%vs3, %%vs3              \n\t" \
 "xxlxor           %%vs4, %%vs4, %%vs4              \n\t" \
 "xxlxor           %%vs5, %%vs5, %%vs5              \n\t" \
 "xxlxor           %%vs6, %%vs6, %%vs6              \n\t" \
 "xxlxor           %%vs7, %%vs7, %%vs7              \n\t" \
 "xxlxor           %%vs8, %%vs8, %%vs8              \n\t" \
 "xxlxor           %%vs9, %%vs9, %%vs9              \n\t" \
 "xxlxor           %%vs10, %%vs10, %%vs10           \n\t" \
 "xxlxor           %%vs11, %%vs11, %%vs11           \n\t" \
 "xxlxor           %%vs12, %%vs12, %%vs12           \n\t" \
 "xxlxor           %%vs13, %%vs13, %%vs13           \n\t" \
 "xxlxor           %%vs14, %%vs14, %%vs14           \n\t" \
 "xxlxor           %%vs15, %%vs15, %%vs15           \n\t" \
 "xxlxor           %%vs16, %%vs16, %%vs16           \n\t" \
 "xxlxor           %%vs17, %%vs17, %%vs17           \n\t" \
 "xxlxor           %%vs18, %%vs18, %%vs18           \n\t" \
 "xxlxor           %%vs19, %%vs19, %%vs19           \n\t" \
 "xxlxor           %%vs20, %%vs20, %%vs20           \n\t" \
 "xxlxor           %%vs21, %%vs21, %%vs21           \n\t" \
 "xxlxor           %%vs22, %%vs22, %%vs22           \n\t" \
 "xxlxor           %%vs23, %%vs23, %%vs23           \n\t" \
 "xxlxor           %%vs24, %%vs24, %%vs24           \n\t" \
 "xxlxor           %%vs25, %%vs25, %%vs25           \n\t" \
 "xxlxor           %%vs26, %%vs26, %%vs26           \n\t" \
 "xxlxor           %%vs27, %%vs27, %%vs27           \n\t" \
 "xxlxor           %%vs28, %%vs28, %%vs28           \n\t" \
 "xxlxor           %%vs29, %%vs29, %%vs29           \n\t" \
 "xxlxor           %%vs30, %%vs30, %%vs30           \n\t" \
 "xxlxor           %%vs31, %%vs31, %%vs31           \n\t" \
 "xxlxor           %%vs32, %%vs32, %%vs32           \n\t" \
 "xxlxor           %%vs33, %%vs33, %%vs33           \n\t" \
 "xxlxor           %%vs34, %%vs34, %%vs34           \n\t" \
 "xxlxor           %%vs35, %%vs35, %%vs35           \n\t" \
 "xxlxor           %%vs36, %%vs36, %%vs36           \n\t" \
 "xxlxor           %%vs37, %%vs37, %%vs37           \n\t" \
 "xxlxor           %%vs38, %%vs38, %%vs38           \n\t" \
 "xxlxor           %%vs39, %%vs39, %%vs39           \n\t" \
 "xxlxor           %%vs40, %%vs40, %%vs40           \n\t" \
 "xxlxor           %%vs41, %%vs41, %%vs41           \n\t" \
 "xxlxor           %%vs42, %%vs42, %%vs42           \n\t" \
 "xxlxor           %%vs43, %%vs43, %%vs43           \n\t" \
 "xxlxor           %%vs44, %%vs44, %%vs44           \n\t" \
 "xxlxor           %%vs45, %%vs45, %%vs45           \n\t" \
 "xxlxor           %%vs46, %%vs46, %%vs46           \n\t" \
 "xxlxor           %%vs47, %%vs47, %%vs47           \n\t" \
 "xxlxor           %%vs48, %%vs48, %%vs48           \n\t" \
 "xxlxor           %%vs49, %%vs49, %%vs49           \n\t" \
 "xxlxor           %%vs50, %%vs50, %%vs50           \n\t" \
 "xxlxor           %%vs51, %%vs51, %%vs51           \n\t" \
 "xxlxor           %%vs52, %%vs52, %%vs52           \n\t" \
 "xxlxor           %%vs53, %%vs53, %%vs53           \n\t" \
 "xxlxor           %%vs54, %%vs54, %%vs54           \n\t" \
 "xxlxor           %%vs55, %%vs55, %%vs55           \n\t" \
 "xxlxor           %%vs56, %%vs56, %%vs56           \n\t" \
 "xxlxor           %%vs57, %%vs57, %%vs57           \n\t" \
 "xxlxor           %%vs58, %%vs58, %%vs58           \n\t" \
 "xxlxor           %%vs59, %%vs59, %%vs59           \n\t" \
 "xxlxor           %%vs60, %%vs60, %%vs60           \n\t" \
 "xxlxor           %%vs61, %%vs61, %%vs61           \n\t" \
 "xxlxor           %%vs62, %%vs62, %%vs62           \n\t" \
 "xxlxor           %%vs63, %%vs63, %%vs63           \n\t"   

#if 1
#define LOADANDUPDATE \
  "                                               \n\t" \
  "lxv           %%vs36, 0(%%r4)                  \n\t" \
  "lxv           %%vs37, 16(%%r4)                 \n\t" \
  "lxv           %%vs38, 32(%%r4)                 \n\t" \
  "lxv           %%vs39, 48(%%r4)                 \n\t" \
  "lxv           %%vs40, 64(%%r4)                 \n\t" \
  "lxv           %%vs41, 80(%%r4)                 \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs0, %%vs36, %%vs48         \n\t" \
  "xvmaddadp        %%vs6, %%vs36, %%vs49         \n\t" \
  "xvmaddadp        %%vs12, %%vs36, %%vs50        \n\t" \
  "xvmaddadp        %%vs18, %%vs36, %%vs51        \n\t" \
  "xvmaddadp        %%vs24, %%vs36, %%vs52        \n\t" \
  "xvmaddadp        %%vs30, %%vs36, %%vs53        \n\t" \

  "xvmaddadp        %%vs1, %%vs37, %%vs48         \n\t" \
  "xvmaddadp        %%vs7, %%vs37, %%vs49         \n\t" \
  "xvmaddadp        %%vs13, %%vs37, %%vs50        \n\t" \
  "xvmaddadp        %%vs19, %%vs37, %%vs51        \n\t" \
  "xvmaddadp        %%vs25, %%vs37, %%vs52        \n\t" \
  "xvmaddadp        %%vs31, %%vs37, %%vs53        \n\t" \

  "xvmaddadp        %%vs2, %%vs38, %%vs48         \n\t" \
  "xvmaddadp        %%vs8, %%vs38, %%vs49         \n\t" \
  "xvmaddadp        %%vs14, %%vs38, %%vs50        \n\t" \
  "xvmaddadp        %%vs20, %%vs38, %%vs51        \n\t" \
  "xvmaddadp        %%vs26, %%vs38, %%vs52        \n\t" \
  "xvmaddadp        %%vs32, %%vs38, %%vs53        \n\t" \

  "xvmaddadp        %%vs3, %%vs39, %%vs48         \n\t" \
  "xvmaddadp        %%vs9, %%vs39, %%vs49         \n\t" \
  "xvmaddadp        %%vs15, %%vs39, %%vs50        \n\t" \
  "xvmaddadp        %%vs21, %%vs39, %%vs51        \n\t" \
  "xvmaddadp        %%vs27, %%vs39, %%vs52        \n\t" \
  "xvmaddadp        %%vs33, %%vs39, %%vs53        \n\t" \

  "xvmaddadp        %%vs4, %%vs40, %%vs48         \n\t" \
  "xvmaddadp        %%vs10, %%vs40, %%vs49        \n\t" \
  "xvmaddadp        %%vs16, %%vs40, %%vs50        \n\t" \
  "xvmaddadp        %%vs22, %%vs40, %%vs51        \n\t" \
  "xvmaddadp        %%vs28, %%vs40, %%vs52        \n\t" \
  "xvmaddadp        %%vs34, %%vs40, %%vs53        \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs5, %%vs41, %%vs48         \n\t" \
  "xvmaddadp        %%vs11, %%vs41, %%vs49        \n\t" \
  "xvmaddadp        %%vs17, %%vs41, %%vs50        \n\t" \
  "xvmaddadp        %%vs23, %%vs41, %%vs51        \n\t" \
  "xvmaddadp        %%vs29, %%vs41, %%vs52        \n\t" \
  "xvmaddadp        %%vs35, %%vs41, %%vs53        \n\t" \
  "                                               \n\t" \
  "addi             %%r4, %%r4, 96                \n\t" \
  "addi             %%r3, %%r3, 48                \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "lxvdsx       %%vs48, %%r22, %%r3               \n\t" \
  "lxvdsx       %%vs49, %%r23, %%r3               \n\t" \
  "lxvdsx       %%vs50, %%r24, %%r3               \n\t" \
  "lxvdsx       %%vs51, %%r25, %%r3               \n\t" \
  "lxvdsx       %%vs52, %%r26, %%r3               \n\t" \
  "lxvdsx       %%vs53, %%r27, %%r3               \n\t" \
  "                                               \n\t" \
  "                                               \n\t" 

#else
#define LOADANDUPDATE \
  "                                               \n\t" \
  "lxv           %%vs36, 0(%%r4)                  \n\t" \
  "lxv           %%vs37, 16(%%r4)                 \n\t" \
  "lxv           %%vs38, 32(%%r4)                 \n\t" \
  "lxv           %%vs39, 48(%%r4)                 \n\t" \
  "lxv           %%vs40, 64(%%r4)                 \n\t" \
  "lxv           %%vs41, 80(%%r4)                 \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs0, %%vs36, %%vs48         \n\t" \
  "xvmaddadp        %%vs1, %%vs37, %%vs48         \n\t" \
  "xvmaddadp        %%vs2, %%vs38, %%vs48         \n\t" \
  "xvmaddadp        %%vs3, %%vs39, %%vs48         \n\t" \
  "xvmaddadp        %%vs4, %%vs40, %%vs48         \n\t" \
  "xvmaddadp        %%vs5, %%vs41, %%vs48         \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs6, %%vs36, %%vs49         \n\t" \
  "xvmaddadp        %%vs7, %%vs37, %%vs49         \n\t" \
  "xvmaddadp        %%vs8, %%vs38, %%vs49         \n\t" \
  "xvmaddadp        %%vs9, %%vs39, %%vs49         \n\t" \
  "xvmaddadp        %%vs10, %%vs40, %%vs49        \n\t" \
  "xvmaddadp        %%vs11, %%vs41, %%vs49        \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs12, %%vs36, %%vs50        \n\t" \
  "xvmaddadp        %%vs13, %%vs37, %%vs50        \n\t" \
  "xvmaddadp        %%vs14, %%vs38, %%vs50        \n\t" \
  "xvmaddadp        %%vs15, %%vs39, %%vs50        \n\t" \
  "xvmaddadp        %%vs16, %%vs40, %%vs50        \n\t" \
  "xvmaddadp        %%vs17, %%vs41, %%vs50        \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs18, %%vs36, %%vs51        \n\t" \
  "xvmaddadp        %%vs19, %%vs37, %%vs51        \n\t" \
  "xvmaddadp        %%vs20, %%vs38, %%vs51        \n\t" \
  "xvmaddadp        %%vs21, %%vs39, %%vs51        \n\t" \
  "xvmaddadp        %%vs22, %%vs40, %%vs51        \n\t" \
  "xvmaddadp        %%vs23, %%vs41, %%vs51        \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs24, %%vs36, %%vs52        \n\t" \
  "xvmaddadp        %%vs25, %%vs37, %%vs52        \n\t" \
  "xvmaddadp        %%vs26, %%vs38, %%vs52        \n\t" \
  "xvmaddadp        %%vs27, %%vs39, %%vs52        \n\t" \
  "xvmaddadp        %%vs28, %%vs40, %%vs52        \n\t" \
  "xvmaddadp        %%vs29, %%vs41, %%vs52        \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "xvmaddadp        %%vs30, %%vs36, %%vs53        \n\t" \
  "xvmaddadp        %%vs31, %%vs37, %%vs53        \n\t" \
  "xvmaddadp        %%vs32, %%vs38, %%vs53        \n\t" \
  "xvmaddadp        %%vs33, %%vs39, %%vs53        \n\t" \
  "xvmaddadp        %%vs34, %%vs40, %%vs53        \n\t" \
  "xvmaddadp        %%vs35, %%vs41, %%vs53        \n\t" \
  "                                               \n\t" \
  "addi             %%r4, %%r4, 96                \n\t" \
  "addi             %%r3, %%r3, 48                \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
  "lxvdsx       %%vs48, %%r22, %%r3               \n\t" \
  "lxvdsx       %%vs49, %%r23, %%r3               \n\t" \
  "lxvdsx       %%vs50, %%r24, %%r3               \n\t" \
  "lxvdsx       %%vs51, %%r25, %%r3               \n\t" \
  "lxvdsx       %%vs52, %%r26, %%r3               \n\t" \
  "lxvdsx       %%vs53, %%r27, %%r3               \n\t" \
  "                                               \n\t" \
  "                                               \n\t" \
#endif

#define SCALEBYALPHA \
 "xvmuldp          %%vs0, %%vs0, %%vs48   	 \n\t" \
 "xvmuldp          %%vs1, %%vs1, %%vs48   	 \n\t" \
 "xvmuldp          %%vs2, %%vs2, %%vs48   	 \n\t" \
 "xvmuldp          %%vs3, %%vs3, %%vs48   	 \n\t" \
 "xvmuldp          %%vs4, %%vs4, %%vs48   	 \n\t" \
 "xvmuldp          %%vs5, %%vs5, %%vs48   	 \n\t" \
 "xvmuldp          %%vs6, %%vs6, %%vs48   	 \n\t" \
 "xvmuldp          %%vs7, %%vs7, %%vs48   	 \n\t" \
 "xvmuldp          %%vs8, %%vs8, %%vs48   	 \n\t" \
 "xvmuldp          %%vs9, %%vs9, %%vs48   	 \n\t" \
 "xvmuldp          %%vs10, %%vs10, %%vs48   	 \n\t" \
 "xvmuldp          %%vs11, %%vs11, %%vs48   	 \n\t" \
 "xvmuldp          %%vs12, %%vs12, %%vs48   	 \n\t" \
 "xvmuldp          %%vs13, %%vs13, %%vs48   	 \n\t" \
 "xvmuldp          %%vs14, %%vs14, %%vs48   	 \n\t" \
 "xvmuldp          %%vs15, %%vs15, %%vs48   	 \n\t" \
 "xvmuldp          %%vs16, %%vs16, %%vs48   	 \n\t" \
 "xvmuldp          %%vs17, %%vs17, %%vs48   	 \n\t" \
 "xvmuldp          %%vs18, %%vs18, %%vs48   	 \n\t" \
 "xvmuldp          %%vs19, %%vs19, %%vs48   	 \n\t" \
 "xvmuldp          %%vs20, %%vs20, %%vs48   	 \n\t" \
 "xvmuldp          %%vs21, %%vs21, %%vs48   	 \n\t" \
 "xvmuldp          %%vs22, %%vs22, %%vs48   	 \n\t" \
 "xvmuldp          %%vs23, %%vs23, %%vs48   	 \n\t" \
 "xvmuldp          %%vs24, %%vs24, %%vs48   	 \n\t" \
 "xvmuldp          %%vs25, %%vs25, %%vs48   	 \n\t" \
 "xvmuldp          %%vs26, %%vs26, %%vs48   	 \n\t" \
 "xvmuldp          %%vs27, %%vs27, %%vs48   	 \n\t" \
 "xvmuldp          %%vs28, %%vs28, %%vs48   	 \n\t" \
 "xvmuldp          %%vs29, %%vs29, %%vs48   	 \n\t" \
 "xvmuldp          %%vs30, %%vs30, %%vs48   	 \n\t" \
 "xvmuldp          %%vs31, %%vs31, %%vs48   	 \n\t" \
 "xvmuldp          %%vs32, %%vs32, %%vs48   	 \n\t" \
 "xvmuldp          %%vs33, %%vs33, %%vs48   	 \n\t" \
 "xvmuldp          %%vs34, %%vs34, %%vs48   	 \n\t" \
 "xvmuldp          %%vs35, %%vs35, %%vs48   	 \n\t" 

#define SCALECMATRIX \
  "xvmuldp          %%vs36, %%vs36, %%vs63        \n\t"  \
  "xvmuldp          %%vs37, %%vs37, %%vs63        \n\t"  \
  "xvmuldp          %%vs38, %%vs38, %%vs63        \n\t"  \
  "xvmuldp          %%vs39, %%vs39, %%vs63        \n\t"  \
  "xvmuldp          %%vs40, %%vs40, %%vs63        \n\t"  \
  "xvmuldp          %%vs41, %%vs41, %%vs63        \n\t"  \
  "xvmuldp          %%vs42, %%vs42, %%vs63        \n\t"  \
  "xvmuldp          %%vs43, %%vs43, %%vs63        \n\t"  \
  "xvmuldp          %%vs44, %%vs44, %%vs63        \n\t"  \
  "xvmuldp          %%vs45, %%vs45, %%vs63        \n\t"  \
  "xvmuldp          %%vs46, %%vs46, %%vs63        \n\t"  \
  "xvmuldp          %%vs47, %%vs47, %%vs63        \n\t"  \
  "xvmuldp          %%vs48, %%vs48, %%vs63        \n\t"  \
  "xvmuldp          %%vs49, %%vs49, %%vs63        \n\t"  \
  "xvmuldp          %%vs50, %%vs50, %%vs63        \n\t"  \
  "xvmuldp          %%vs51, %%vs51, %%vs63        \n\t"  \
  "xvmuldp          %%vs52, %%vs52, %%vs63        \n\t"  \
  "xvmuldp          %%vs53, %%vs53, %%vs63        \n\t"
 
#define LOADCMATRIX \
  "lxv              %%vs36, 0(%%r22)              \n\t" \
  "lxv              %%vs37, 16(%%r22)             \n\t" \
  "lxv              %%vs38, 32(%%r22)             \n\t" \
  "lxv              %%vs39, 48(%%r22)             \n\t" \
  "lxv              %%vs40, 64(%%r22)             \n\t" \
  "lxv              %%vs41, 80(%%r22)             \n\t" \
  "lxv              %%vs42, 0(%%r23)             \n\t" \
  "lxv              %%vs43, 16(%%r23)            \n\t" \
  "lxv              %%vs44, 32(%%r23)            \n\t" \
  "lxv              %%vs45, 48(%%r23)            \n\t" \
  "lxv              %%vs46, 64(%%r23)            \n\t" \
  "lxv              %%vs47, 80(%%r23)            \n\t" \
  "lxv              %%vs48, 0(%%r24)             \n\t" \
  "lxv              %%vs49, 16(%%r24)            \n\t" \
  "lxv              %%vs50, 32(%%r24)            \n\t" \
  "lxv              %%vs51, 48(%%r24)            \n\t" \
  "lxv              %%vs52, 64(%%r24)            \n\t" \
  "lxv              %%vs53, 80(%%r24)            \n\t"

#define STORECMATRIX \
  "stxv              %%vs0, 0(%%r16)    \n\t" \
  "stxv              %%vs1, 16(%%r16)    \n\t" \
  "stxv              %%vs2, 32(%%r16)    \n\t" \
  "stxv              %%vs3, 48(%%r16)    \n\t" \
  "stxv              %%vs4, 64(%%r16)    \n\t" \
  "stxv              %%vs5, 80(%%r16)    \n\t" \
  "stxv              %%vs6, 0(%%r17)    \n\t" \
  "stxv              %%vs7, 16(%%r17)    \n\t" \
  "stxv              %%vs8, 32(%%r17)    \n\t" \
  "stxv              %%vs9, 48(%%r17)    \n\t" \
  "stxv              %%vs10, 64(%%r17)    \n\t" \
  "stxv              %%vs11, 80(%%r17)    \n\t" \
  "stxv              %%vs12, 0(%%r18)    \n\t" \
  "stxv              %%vs13, 16(%%r18)    \n\t" \
  "stxv              %%vs14, 32(%%r18)    \n\t" \
  "stxv              %%vs15, 48(%%r18)    \n\t" \
  "stxv              %%vs16, 64(%%r18)    \n\t" \
  "stxv              %%vs17, 80(%%r18)    \n\t" \
  "stxv              %%vs18, 0(%%r19)    \n\t" \
  "stxv              %%vs19, 16(%%r19)    \n\t" \
  "stxv              %%vs20, 32(%%r19)    \n\t" \
  "stxv              %%vs21, 48(%%r19)    \n\t" \
  "stxv              %%vs22, 64(%%r19)    \n\t" \
  "stxv              %%vs23, 80(%%r19)    \n\t" \
  "stxv              %%vs24, 0(%%r20)    \n\t" \
  "stxv              %%vs25, 16(%%r20)    \n\t" \
  "stxv              %%vs26, 32(%%r20)    \n\t" \
  "stxv              %%vs27, 48(%%r20)    \n\t" \
  "stxv              %%vs28, 64(%%r20)    \n\t" \
  "stxv              %%vs29, 80(%%r20)    \n\t" \
  "stxv              %%vs30, 0(%%r21)    \n\t" \
  "stxv              %%vs31, 16(%%r21)    \n\t" \
  "stxv              %%vs32, 32(%%r21)    \n\t" \
  "stxv              %%vs33, 48(%%r21)    \n\t" \
  "stxv              %%vs34, 64(%%r21)    \n\t" \
  "stxv              %%vs35, 80(%%r21)    \n\t"



void bli_dgemm_power9_asm_12x6
     (
       dim_t               k0,
       double*    restrict alpha,
       double*    restrict a,
       double*    restrict b,
       double*    restrict beta,
       double*    restrict c, inc_t rs_c0, inc_t cs_c0,
       auxinfo_t* restrict data,
       cntx_t*    restrict cntx
     )
{
	//void*   a_next = bli_auxinfo_next_a( data );
	//void*   b_next = bli_auxinfo_next_b( data );

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
	uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

  if(rs_c0 != 1)
  {
    printf("rs_c0 = %ld | cs_c0 = %ld\n", rs_c0, cs_c0);
    bli_check_error_code(BLIS_NOT_YET_IMPLEMENTED);
  }


	__asm__ volatile
	(
	"                                               \n\t"
  VSZEROOUT                                             // Zero out vec regs
  "                                               \n\t"
  "ld               %%r4, %2                      \n\t" // load ptr of A
  "ld               %%r3, %3                      \n\t" // load ptr of B
  "ld               %%r16, %6                      \n\t" // load ptr for C
  "                                               \n\t" 
  "                                               \n\t" 
  "ld               %%r6, %8                      \n\t" // load cs_c
  "                                               \n\t"
  "                                               \n\t" // Offsets for B
  "li               %%r22,0                       \n\t" // 0
  "li               %%r23,8                       \n\t" // 1
  "li               %%r24,16                      \n\t" // 2
  "li               %%r25,24                      \n\t" // 3
  "li               %%r26,32                      \n\t" // 4
  "li               %%r27,40                      \n\t" // 5
  "                                               \n\t"
  "                                               \n\t"
  "ld               %%r9, %0                      \n\t" // Set k_iter to be loop counter
  "mtctr            %%r9                          \n\t"
  "                                               \n\t"
  "lxvdsx       %%vs48, %%r22, %%r3               \n\t" // load first row of B
  "lxvdsx       %%vs49, %%r23, %%r3               \n\t" 
  "lxvdsx       %%vs50, %%r24, %%r3               \n\t" 
  "lxvdsx       %%vs51, %%r25, %%r3               \n\t" 
  "lxvdsx       %%vs52, %%r26, %%r3               \n\t" 
  "lxvdsx       %%vs53, %%r27, %%r3               \n\t" 
  "                                               \n\t" // k_iter loop does A*B 
  "DLOOPKITER:                                    \n\t" // Begin k_iter loop
  "                                               \n\t"
  LOADANDUPDATE
  LOADANDUPDATE
  LOADANDUPDATE
  LOADANDUPDATE
  "                                               \n\t"
  "bdnz             DLOOPKITER                    \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "ld               %%r9, %1                      \n\t" // edge case
  "cmpwi            %%r0, %%r9, 0                 \n\t"
  "beq              %%r0, DPOSTACCUM              \n\t"
  "mtctr            %%r9                          \n\t"
  "                                               \n\t"
  "DLOOPKLEFT:                                    \n\t" // EDGE LOOP
  LOADANDUPDATE
  "bdnz             DLOOPKLEFT                    \n\t"
  "                                               \n\t"
  "DPOSTACCUM:                                    \n\t"
  "                                               \n\t"
  "ld               %%r8, %4                      \n\t" // load ptr for alpha
  "ld               %%r5, %5                      \n\t" // load ptr for beta
  "                                               \n\t"
  "lxvdsx           %%vs48, 0, %%r8               \n\t" // splat alpha
  "lxvdsx           %%vs63, 0, %%r5               \n\t" // splat beta
  "                                               \n\t"
  SCALEBYALPHA
  "                                               \n\t"
  "slwi             %%r6, %%r6, 3                 \n\t" // mul by size of elem
  "add              %%r17, %%r16, %%r6             \n\t" // c + cs_c
  "add              %%r18, %%r17, %%r6             \n\t" // c + cs_c * 2
  "add              %%r19, %%r18, %%r6             \n\t" // c + cs_c * 3
  "add              %%r20, %%r19, %%r6             \n\t" // c + cs_c * 4
  "add              %%r21, %%r20, %%r6             \n\t" // c + cs_c * 5
  "                                               \n\t"
  "cmpwi            %%r0, %%r5, 0                 \n\t"
  "beq              %%r0, DBETAZERO               \n\t"
  "                                               \n\t"
  "ld               %%r22, %6                     \n\t" // load ptr for C (used as offset)
  "add              %%r23, %%r22, %%r6            \n\t" // load ptr for C (used as offset)
  "add              %%r24, %%r24, %%r6            \n\t" // load ptr for C (used as offset)
  "                                               \n\t"
  "ADDTOC:                                        \n\t" // C = beta*C + alpha*(AB)
  "                                               \n\t"
  LOADCMATRIX
  "add             %%r22, %%r24, %%r6             \n\t" // Move C-ptrs
  "add             %%r23, %%r22, %%r6             \n\t" // Move C-ptrs
  "add             %%r24, %%r23, %%r6             \n\t" 
  SCALECMATRIX
  "                                               \n\t"
  "xvadddp          %%vs0, %%vs0, %%vs36          \n\t"  
  "xvadddp          %%vs1, %%vs1, %%vs37          \n\t"  
  "xvadddp          %%vs2, %%vs2, %%vs38          \n\t"  
  "xvadddp          %%vs3, %%vs3, %%vs39          \n\t"  
  "xvadddp          %%vs4, %%vs4, %%vs40          \n\t"  
  "xvadddp          %%vs5, %%vs5, %%vs41          \n\t"  
  "xvadddp          %%vs6, %%vs6, %%vs42          \n\t"  
  "xvadddp          %%vs7, %%vs7, %%vs43          \n\t"  
  "xvadddp          %%vs8, %%vs8, %%vs44          \n\t"  
  "xvadddp          %%vs9, %%vs9, %%vs45          \n\t"  
  "xvadddp          %%vs10, %%vs10, %%vs46        \n\t"  
  "xvadddp          %%vs11, %%vs11, %%vs47        \n\t"
  "xvadddp          %%vs12, %%vs12, %%vs48        \n\t"  
  "xvadddp          %%vs13, %%vs13, %%vs49        \n\t"  
  "xvadddp          %%vs14, %%vs14, %%vs50        \n\t"  
  "xvadddp          %%vs15, %%vs15, %%vs51        \n\t"  
  "xvadddp          %%vs16, %%vs16, %%vs52        \n\t"  
  "xvadddp          %%vs17, %%vs17, %%vs53        \n\t" 
  "                                               \n\t"
  LOADCMATRIX
  SCALECMATRIX
  "                                               \n\t"   
  "                                               \n\t"
  "xvadddp          %%vs18, %%vs18, %%vs36        \n\t"  
  "xvadddp          %%vs19, %%vs19, %%vs37        \n\t"  
  "xvadddp          %%vs20, %%vs20, %%vs38        \n\t"  
  "xvadddp          %%vs21, %%vs21, %%vs39        \n\t"  
  "xvadddp          %%vs22, %%vs22, %%vs40        \n\t"  
  "xvadddp          %%vs23, %%vs23, %%vs41        \n\t"
  "xvadddp          %%vs24, %%vs24, %%vs42   	    \n\t"  
  "xvadddp          %%vs25, %%vs25, %%vs43   	    \n\t"  
  "xvadddp          %%vs26, %%vs26, %%vs44   	    \n\t"  
  "xvadddp          %%vs27, %%vs27, %%vs45   	    \n\t"  
  "xvadddp          %%vs28, %%vs28, %%vs46   	    \n\t"  
  "xvadddp          %%vs29, %%vs29, %%vs47   	    \n\t"  
  "xvadddp          %%vs30, %%vs30, %%vs48   	    \n\t"  
  "xvadddp          %%vs31, %%vs31, %%vs49   	    \n\t"  
  "xvadddp          %%vs32, %%vs32, %%vs50   	    \n\t"  
  "xvadddp          %%vs33, %%vs33, %%vs51   	    \n\t"  
  "xvadddp          %%vs34, %%vs34, %%vs52   	    \n\t"  
  "xvadddp          %%vs35, %%vs35, %%vs53   	    \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "                                               \n\t"
  "DBETAZERO:                                     \n\t"
  "                                               \n\t" 
  STORECMATRIX 
  "                                               \n\t"
	: // output operands (none)
	: // input operands
	  "m" (k_iter), // 0
	  "m" (k_left), // 1
	  "m" (a),      // 2
	  "m" (b),      // 3
	  "m" (alpha),  // 4
	  "m" (beta),   // 5
	  "m" (c),      // 6
	  "m" (rs_c),   // 7
	  "m" (cs_c)/*,   // 8
	  "m" (b_next), // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
  /* unclobberable regs: r2(PIC reg), */
  "r0", "r3", "r4", "r5", "r6", "r8", "r9", 
  "r10", "r11", "r15", "r16", "r17", "r18", "r19",
  "r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r29", 

  "vs0", "vs1", "vs2", "vs3", "vs4", "vs5", "vs6", "vs7", "vs8", "vs9", "vs10",
  "vs11", "vs12", "vs13", "vs14", "vs15", "vs16", "vs17", "vs18", "vs19", "vs20",
  "vs21", "vs22", "vs23", "vs24", "vs25", "vs26", "vs27", "vs28", "vs29", "vs30",
  "vs31", "vs32", "vs33", "vs34", "vs35", "vs36", "vs37", "vs38", "vs39", "vs40",
  "vs41", "vs42", "vs43", "vs44", "vs45", "vs46", "vs47", "vs48", "vs49", "vs50",
  "vs51", "vs52", "vs53"
  );
}
