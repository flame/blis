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

#define XLC 0 

#define VSZEROOUT \
 "xxlxor           %%vs0, %%vs0, %%vs0          \n\t" \
 "xxlxor           %%vs1, %%vs1, %%vs1          \n\t" \
 "xxlxor           %%vs2, %%vs2, %%vs2          \n\t" \
 "xxlxor           %%vs3, %%vs3, %%vs3          \n\t" \
 "xxlxor           %%vs4, %%vs4, %%vs4          \n\t" \
 "xxlxor           %%vs5, %%vs5, %%vs5          \n\t" \
 "xxlxor           %%vs6, %%vs6, %%vs6          \n\t" \
 "xxlxor           %%vs7, %%vs7, %%vs7          \n\t" \
 "xxlxor           %%vs8, %%vs8, %%vs8          \n\t" \
 "xxlxor           %%vs9, %%vs9, %%vs9          \n\t" \
 "xxlxor           %%vs10, %%vs10, %%vs10       \n\t" \
 "xxlxor           %%vs11, %%vs11, %%vs11       \n\t" \
 "xxlxor           %%vs12, %%vs12, %%vs12       \n\t" \
 "xxlxor           %%vs13, %%vs13, %%vs13       \n\t" \
 "xxlxor           %%vs14, %%vs14, %%vs14       \n\t" \
 "xxlxor           %%vs15, %%vs15, %%vs15       \n\t" \
 "xxlxor           %%vs16, %%vs16, %%vs16       \n\t" \
 "xxlxor           %%vs17, %%vs17, %%vs17       \n\t" \
 "xxlxor           %%vs18, %%vs18, %%vs18       \n\t" \
 "xxlxor           %%vs19, %%vs19, %%vs19       \n\t" \
 "xxlxor           %%vs20, %%vs20, %%vs20       \n\t" \
 "xxlxor           %%vs21, %%vs21, %%vs21       \n\t" \
 "xxlxor           %%vs22, %%vs22, %%vs22       \n\t" \
 "xxlxor           %%vs23, %%vs23, %%vs23       \n\t" \
 "xxlxor           %%vs24, %%vs24, %%vs24       \n\t" \
 "xxlxor           %%vs25, %%vs25, %%vs25       \n\t" \
 "xxlxor           %%vs26, %%vs26, %%vs26       \n\t" \
 "xxlxor           %%vs27, %%vs27, %%vs27       \n\t" \
 "xxlxor           %%vs28, %%vs28, %%vs28       \n\t" \
 "xxlxor           %%vs29, %%vs29, %%vs29       \n\t" \
 "xxlxor           %%vs30, %%vs30, %%vs30       \n\t" \
 "xxlxor           %%vs31, %%vs31, %%vs31       \n\t" \
 "xxlxor           %%vs32, %%vs32, %%vs32       \n\t" \
 "xxlxor           %%vs33, %%vs33, %%vs33       \n\t" \
 "xxlxor           %%vs34, %%vs34, %%vs34       \n\t" \
 "xxlxor           %%vs35, %%vs35, %%vs35       \n\t" \
 "xxlxor           %%vs36, %%vs36, %%vs36       \n\t" \
 "xxlxor           %%vs37, %%vs37, %%vs37       \n\t" \
 "xxlxor           %%vs38, %%vs38, %%vs38       \n\t" \
 "xxlxor           %%vs39, %%vs39, %%vs39       \n\t" \
 "xxlxor           %%vs40, %%vs40, %%vs40       \n\t" \
 "xxlxor           %%vs41, %%vs41, %%vs41       \n\t" \
 "xxlxor           %%vs42, %%vs42, %%vs42       \n\t" \
 "xxlxor           %%vs43, %%vs43, %%vs43       \n\t" \
 "xxlxor           %%vs44, %%vs44, %%vs44       \n\t" \
 "xxlxor           %%vs45, %%vs45, %%vs45       \n\t" \
 "xxlxor           %%vs46, %%vs46, %%vs46       \n\t" \
 "xxlxor           %%vs47, %%vs47, %%vs47       \n\t" \
 "xxlxor           %%vs48, %%vs48, %%vs48       \n\t" \
 "xxlxor           %%vs49, %%vs49, %%vs49       \n\t" \
 "xxlxor           %%vs50, %%vs50, %%vs50       \n\t" \
 "xxlxor           %%vs51, %%vs51, %%vs51       \n\t" \
 "xxlxor           %%vs52, %%vs52, %%vs52       \n\t" \
 "xxlxor           %%vs53, %%vs53, %%vs53       \n\t" \
 "xxlxor           %%vs54, %%vs54, %%vs54       \n\t" \
 "xxlxor           %%vs55, %%vs55, %%vs55       \n\t" \
 "xxlxor           %%vs56, %%vs56, %%vs56       \n\t" \
 "xxlxor           %%vs57, %%vs57, %%vs57       \n\t" \
 "xxlxor           %%vs58, %%vs58, %%vs58       \n\t" \
 "xxlxor           %%vs59, %%vs59, %%vs59       \n\t" \
 "xxlxor           %%vs60, %%vs60, %%vs60       \n\t" \
 "xxlxor           %%vs61, %%vs61, %%vs61       \n\t" \
 "xxlxor           %%vs62, %%vs62, %%vs62       \n\t" \
 "xxlxor           %%vs63, %%vs63, %%vs63       \n\t" 

#define SCALEBYALPHA \
"xvmuldp         %%vs0, %%vs0, %%vs60           \n\t" \
"xvmuldp         %%vs1, %%vs1, %%vs60           \n\t" \
"xvmuldp         %%vs2, %%vs2, %%vs60           \n\t" \
"xvmuldp         %%vs3, %%vs3, %%vs60           \n\t" \
"xvmuldp         %%vs4, %%vs4, %%vs60           \n\t" \
"xvmuldp         %%vs5, %%vs5, %%vs60           \n\t" \
"xvmuldp         %%vs6, %%vs6, %%vs60           \n\t" \
"xvmuldp         %%vs7, %%vs7, %%vs60           \n\t" \
"xvmuldp         %%vs8, %%vs8, %%vs60           \n\t" \
"xvmuldp         %%vs9, %%vs9, %%vs60           \n\t" \
"xvmuldp         %%vs10, %%vs10, %%vs60         \n\t" \
"xvmuldp         %%vs11, %%vs11, %%vs60         \n\t" \
"xvmuldp         %%vs12, %%vs12, %%vs60         \n\t" \
"xvmuldp         %%vs13, %%vs13, %%vs60         \n\t" \
"xvmuldp         %%vs14, %%vs14, %%vs60         \n\t" \
"xvmuldp         %%vs15, %%vs15, %%vs60         \n\t" \
"xvmuldp         %%vs16, %%vs16, %%vs60         \n\t" \
"xvmuldp         %%vs17, %%vs17, %%vs60         \n\t" \
"xvmuldp         %%vs18, %%vs18, %%vs60         \n\t" \
"xvmuldp         %%vs19, %%vs19, %%vs60         \n\t" \
"xvmuldp         %%vs20, %%vs20, %%vs60         \n\t" \
"xvmuldp         %%vs21, %%vs21, %%vs60         \n\t" \
"xvmuldp         %%vs22, %%vs22, %%vs60         \n\t" \
"xvmuldp         %%vs23, %%vs23, %%vs60         \n\t" \
"xvmuldp         %%vs24, %%vs24, %%vs60         \n\t" \
"xvmuldp         %%vs25, %%vs25, %%vs60         \n\t" \
"xvmuldp         %%vs26, %%vs26, %%vs60         \n\t" \
"xvmuldp         %%vs27, %%vs27, %%vs60         \n\t" \
"xvmuldp         %%vs28, %%vs28, %%vs60         \n\t" \
"xvmuldp         %%vs29, %%vs29, %%vs60         \n\t" \
"xvmuldp         %%vs30, %%vs30, %%vs60         \n\t" \
"xvmuldp         %%vs31, %%vs31, %%vs60         \n\t"

#define PRELOAD_A_B \
"lxv              %%vs32, 0(%%r7)               \n\t" \
"lxv              %%vs33, 16(%%r7)              \n\t" \
"lxv              %%vs34, 32(%%r7)              \n\t" \
"lxv              %%vs35, 48(%%r7)              \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 0(%%r8)               \n\t" \
"lxv              %%vs50, 16(%%r8)              \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 64(%%r7)              \n\t" \
"lxv              %%vs37, 80(%%r7)              \n\t" \
"lxv              %%vs38, 96(%%r7)              \n\t" \
"lxv              %%vs39, 112(%%r7)             \n\t" 


#define LOADANDUPDATE2 \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs40, 0(%%r7)               \n\t" \
"lxv              %%vs41, 16(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"lxv              %%vs42, 32(%%r7)              \n\t" \
"lxv              %%vs43, 48(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs52, 0(%%r8)               \n\t" \
"lxv              %%vs54, 16(%%r8)              \n\t" \
"xxpermdi         %%vs53, %%vs52, %%vs52, 2     \n\t" \
"xxpermdi         %%vs55, %%vs54, %%vs54, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"lxv              %%vs44, 64(%%r7)              \n\t" \
"lxv              %%vs45, 80(%%r7)              \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"                                               \n\t" \
"lxv              %%vs46, 96(%%r7)              \n\t" \
"lxv              %%vs47, 112(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 32(%%r8)              \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs40, %%vs52         \n\t" \
"xvmaddadp        %%vs1, %%vs41, %%vs52         \n\t" \
"xvmaddadp        %%vs2, %%vs42, %%vs52         \n\t" \
"xvmaddadp        %%vs3, %%vs43, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs50, 48(%%r8)              \n\t" \
"xxpermdi         %%vs51, %%vs51, %%vs51, 2     \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs4, %%vs44, %%vs52         \n\t" \
"xvmaddadp        %%vs5, %%vs45, %%vs52         \n\t" \
"xvmaddadp        %%vs6, %%vs46, %%vs52         \n\t" \
"xvmaddadp        %%vs7, %%vs47, %%vs52         \n\t" \
"                                               \n\t" \
"lxv              %%vs32, 128(%%r7)             \n\t" \
"lxv              %%vs33, 144(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs40, %%vs53         \n\t" \
"xvmaddadp        %%vs9, %%vs41, %%vs53         \n\t" \
"xvmaddadp        %%vs10, %%vs42, %%vs53        \n\t" \
"xvmaddadp        %%vs11, %%vs43, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs34, 160(%%r7)             \n\t" \
"lxv              %%vs35, 176(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs12, %%vs44, %%vs53        \n\t" \
"xvmaddadp        %%vs13, %%vs45, %%vs53        \n\t" \
"xvmaddadp        %%vs14, %%vs46, %%vs53        \n\t" \
"xvmaddadp        %%vs15, %%vs47, %%vs53        \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 192(%%r7)             \n\t" \
"lxv              %%vs37, 208(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs40, %%vs54        \n\t" \
"xvmaddadp        %%vs17, %%vs41, %%vs54        \n\t" \
"xvmaddadp        %%vs18, %%vs42, %%vs54        \n\t" \
"xvmaddadp        %%vs19, %%vs43, %%vs54        \n\t" \
"                                               \n\t" \
"lxv              %%vs38, 224(%%r7)             \n\t" \
"lxv              %%vs39, 240(%%r7)             \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs20, %%vs44, %%vs54        \n\t" \
"xvmaddadp        %%vs21, %%vs45, %%vs54        \n\t" \
"xvmaddadp        %%vs22, %%vs46, %%vs54        \n\t" \
"xvmaddadp        %%vs23, %%vs47, %%vs54        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs40, %%vs55        \n\t" \
"xvmaddadp        %%vs25, %%vs41, %%vs55        \n\t" \
"xvmaddadp        %%vs26, %%vs42, %%vs55        \n\t" \
"xvmaddadp        %%vs27, %%vs43, %%vs55        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs28, %%vs44, %%vs55        \n\t" \
"xvmaddadp        %%vs29, %%vs45, %%vs55        \n\t" \
"xvmaddadp        %%vs30, %%vs46, %%vs55        \n\t" \
"xvmaddadp        %%vs31, %%vs47, %%vs55        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t"

#define LOADANDUPDATE1 \
"lxv              %%vs32, 0(%%r7)               \n\t" \
"lxv              %%vs33, 16(%%r7)              \n\t" \
"lxv              %%vs34, 32(%%r7)              \n\t" \
"lxv              %%vs35, 48(%%r7)              \n\t" \
"                                               \n\t" \
"lxv              %%vs48, 0(%%r8)               \n\t" \
"lxv              %%vs50, 16(%%r8)              \n\t" \
"xxpermdi         %%vs49, %%vs48, %%vs48, 2     \n\t" \
"xxpermdi         %%vs51, %%vs50, %%vs50, 2     \n\t" \
"                                               \n\t" \
"lxv              %%vs36, 64(%%r7)              \n\t" \
"lxv              %%vs37, 80(%%r7)              \n\t" \
"lxv              %%vs38, 96(%%r7)              \n\t" \
"lxv              %%vs39, 112(%%r7)             \n\t" \
"                                               \n\t" \
"addi             %%r8, %%r8, 32                \n\t" \
"addi             %%r7, %%r7, 128               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs0, %%vs32, %%vs48         \n\t" \
"xvmaddadp        %%vs1, %%vs33, %%vs48         \n\t" \
"xvmaddadp        %%vs2, %%vs34, %%vs48         \n\t" \
"xvmaddadp        %%vs3, %%vs35, %%vs48         \n\t" \
"xvmaddadp        %%vs4, %%vs36, %%vs48         \n\t" \
"xvmaddadp        %%vs5, %%vs37, %%vs48         \n\t" \
"xvmaddadp        %%vs6, %%vs38, %%vs48         \n\t" \
"xvmaddadp        %%vs7, %%vs39, %%vs48         \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs8, %%vs32, %%vs49         \n\t" \
"xvmaddadp        %%vs9, %%vs33, %%vs49         \n\t" \
"xvmaddadp        %%vs10, %%vs34, %%vs49        \n\t" \
"xvmaddadp        %%vs11, %%vs35, %%vs49        \n\t" \
"xvmaddadp        %%vs12, %%vs36, %%vs49        \n\t" \
"xvmaddadp        %%vs13, %%vs37, %%vs49        \n\t" \
"xvmaddadp        %%vs14, %%vs38, %%vs49        \n\t" \
"xvmaddadp        %%vs15, %%vs39, %%vs49        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs16, %%vs32, %%vs50        \n\t" \
"xvmaddadp        %%vs17, %%vs33, %%vs50        \n\t" \
"xvmaddadp        %%vs18, %%vs34, %%vs50        \n\t" \
"xvmaddadp        %%vs19, %%vs35, %%vs50        \n\t" \
"xvmaddadp        %%vs20, %%vs36, %%vs50        \n\t" \
"xvmaddadp        %%vs21, %%vs37, %%vs50        \n\t" \
"xvmaddadp        %%vs22, %%vs38, %%vs50        \n\t" \
"xvmaddadp        %%vs23, %%vs39, %%vs50        \n\t" \
"                                               \n\t" \
"xvmaddadp        %%vs24, %%vs32, %%vs51        \n\t" \
"xvmaddadp        %%vs25, %%vs33, %%vs51        \n\t" \
"xvmaddadp        %%vs26, %%vs34, %%vs51        \n\t" \
"xvmaddadp        %%vs27, %%vs35, %%vs51        \n\t" \
"xvmaddadp        %%vs28, %%vs36, %%vs51        \n\t" \
"xvmaddadp        %%vs29, %%vs37, %%vs51        \n\t" \
"xvmaddadp        %%vs30, %%vs38, %%vs51        \n\t" \
"xvmaddadp        %%vs31, %%vs39, %%vs51        \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" \
"                                               \n\t" 



#define COL_ADD_STORE \
"xvadddp          %%vs48, %%vs48, %%vs32   	    \n\t" \
"xvadddp          %%vs49, %%vs49, %%vs33   	    \n\t" \
"xvadddp          %%vs50, %%vs50, %%vs34   	    \n\t" \
"xvadddp          %%vs51, %%vs51, %%vs35        \n\t" \
"xvadddp          %%vs52, %%vs52, %%vs36 	      \n\t" \
"xvadddp          %%vs53, %%vs53, %%vs37    	  \n\t" \
"xvadddp          %%vs54, %%vs54, %%vs38   	    \n\t" \
"xvadddp          %%vs55, %%vs55, %%vs39     	  \n\t" \
"xvadddp          %%vs56, %%vs56, %%vs40   	    \n\t" \
"xvadddp          %%vs57, %%vs57, %%vs41   	    \n\t" \
"xvadddp          %%vs58, %%vs58, %%vs42   	    \n\t" \
"xvadddp          %%vs59, %%vs59, %%vs43        \n\t" \
"xvadddp          %%vs60, %%vs60, %%vs44 	      \n\t" \
"xvadddp          %%vs61, %%vs61, %%vs45    	  \n\t" \
"xvadddp          %%vs62, %%vs62, %%vs46   	    \n\t" \
"xvadddp          %%vs63, %%vs63, %%vs47     	  \n\t" \
"            	                                  \n\t" \
"            	                                  \n\t" \
"stxv              %%vs48, 0(%%r24)             \n\t" \
"stxv              %%vs49, 16(%%r24)            \n\t" \
"stxv              %%vs50, 32(%%r24)            \n\t" \
"stxv              %%vs51, 48(%%r24)            \n\t" \
"stxv              %%vs52, 64(%%r24)            \n\t" \
"stxv              %%vs53, 80(%%r24)            \n\t" \
"stxv              %%vs54, 96(%%r24)            \n\t" \
"stxv              %%vs55, 112(%%r24)           \n\t" \
"stxv              %%vs56, 0(%%r25)             \n\t" \
"stxv              %%vs57, 16(%%r25)            \n\t" \
"stxv              %%vs58, 32(%%r25)            \n\t" \
"stxv              %%vs59, 48(%%r25)            \n\t" \
"stxv              %%vs60, 64(%%r25)            \n\t" \
"stxv              %%vs61, 80(%%r25)            \n\t" \
"stxv              %%vs62, 96(%%r25)            \n\t" \
"stxv              %%vs63, 112(%%r25)           \n\t" 




#define SCALECOL_CMATRIX \
"xvmuldp         %%vs32, %%vs32, %%vs59         \n\t" \
"xvmuldp         %%vs33, %%vs33, %%vs59         \n\t" \
"xvmuldp         %%vs34, %%vs34, %%vs59         \n\t" \
"xvmuldp         %%vs35, %%vs35, %%vs59         \n\t" \
"xvmuldp         %%vs36, %%vs36, %%vs59         \n\t" \
"xvmuldp         %%vs37, %%vs37, %%vs59         \n\t" \
"xvmuldp         %%vs38, %%vs38, %%vs59         \n\t" \
"xvmuldp         %%vs39, %%vs39, %%vs59         \n\t" \
"xvmuldp         %%vs40, %%vs40, %%vs59         \n\t" \
"xvmuldp         %%vs41, %%vs41, %%vs59         \n\t" \
"xvmuldp         %%vs42, %%vs42, %%vs59         \n\t" \
"xvmuldp         %%vs43, %%vs43, %%vs59         \n\t" \
"xvmuldp         %%vs44, %%vs44, %%vs59         \n\t" \
"xvmuldp         %%vs45, %%vs45, %%vs59         \n\t" \
"xvmuldp         %%vs46, %%vs46, %%vs59         \n\t" \
"xvmuldp         %%vs47, %%vs47, %%vs59         \n\t"  

#define SCALEGEN_CMATRIX \
"xvmuldp          %%vs32, %%vs32, %%vs59   	    \n\t" \
"xvmuldp          %%vs33, %%vs33, %%vs59   	    \n\t" \
"xvmuldp          %%vs34, %%vs34, %%vs59   	    \n\t" \
"xvmuldp          %%vs35, %%vs35, %%vs59   	    \n\t" \
"xvmuldp          %%vs36, %%vs36, %%vs59   	    \n\t" \
"xvmuldp          %%vs37, %%vs37, %%vs59   	    \n\t" \
"xvmuldp          %%vs38, %%vs38, %%vs59   	    \n\t" \
"xvmuldp          %%vs39, %%vs39, %%vs59   	    \n\t" 

#define LOADGEN_CMATRIX \
"lxsdx     %%vs32, %%r9, %%r22                  \n\t" \
"lxsdx     %%vs33, 0, %%r22                     \n\t" \
"lxsdx     %%vs34, %%r9, %%r23                  \n\t" \
"lxsdx     %%vs35, 0, %%r23                     \n\t" \
"lxsdx     %%vs36, %%r9, %%r24                  \n\t" \
"lxsdx     %%vs37, 0, %%r24                     \n\t" \
"lxsdx     %%vs38, %%r9, %%r25                  \n\t" \
"lxsdx     %%vs39, 0, %%r25                     \n\t" \
"lxsdx     %%vs40, %%r9, %%r26                  \n\t" \
"lxsdx     %%vs41, 0, %%r26                     \n\t" \
"lxsdx     %%vs42, %%r9, %%r27                  \n\t" \
"lxsdx     %%vs43, 0, %%r27                     \n\t" \
"lxsdx     %%vs44, %%r9, %%r28                  \n\t" \
"lxsdx     %%vs45, 0, %%r28                     \n\t" \
"lxsdx     %%vs46, %%r9, %%r29                  \n\t" \
"lxsdx     %%vs47, 0, %%r29                     \n\t" \
"xxpermdi    %%vs32, %%vs32, %%vs33, 0          \n\t" \
"xxpermdi    %%vs33, %%vs34, %%vs35, 0          \n\t" \
"xxpermdi    %%vs34, %%vs36, %%vs37, 0          \n\t" \
"xxpermdi    %%vs35, %%vs38, %%vs39, 0          \n\t" \
"xxpermdi    %%vs36, %%vs40, %%vs41, 0          \n\t" \
"xxpermdi    %%vs37, %%vs42, %%vs43, 0          \n\t" \
"xxpermdi    %%vs38, %%vs44, %%vs45, 0          \n\t" \
"xxpermdi    %%vs39, %%vs46, %%vs47, 0          \n\t" 


#define GEN_NEXT_COL_CMATRIX \
"add             %%r22, %%r22, %%r10            \n\t" \
"add             %%r23, %%r23, %%r10            \n\t" \
"add             %%r24, %%r24, %%r10            \n\t" \
"add             %%r25, %%r25, %%r10            \n\t" \
"add             %%r26, %%r26, %%r10            \n\t" \
"add             %%r27, %%r27, %%r10            \n\t" \
"add             %%r28, %%r28, %%r10            \n\t" \
"add             %%r29, %%r29, %%r10            \n\t"

#define GENADD_STORE \
"xvadddp      %%vs40, %%vs40, %%vs32            \n\t" \
"xvadddp      %%vs41, %%vs41, %%vs33            \n\t" \
"xvadddp      %%vs42, %%vs42, %%vs34            \n\t" \
"xvadddp      %%vs43, %%vs43, %%vs35            \n\t" \
"xvadddp      %%vs44, %%vs44, %%vs36            \n\t" \
"xvadddp      %%vs45, %%vs45, %%vs37            \n\t" \
"xvadddp      %%vs46, %%vs46, %%vs38            \n\t" \
"xvadddp      %%vs47, %%vs47, %%vs39            \n\t" \
"                                              	\n\t" \
"stxsdx       %%vs40, %%r9, %%r22               \n\t" \
"xxswapd      %%vs40, %%vs40                    \n\t" \
"stxsdx       %%vs41, %%r9, %%r23               \n\t" \
"xxswapd      %%vs41, %%vs41                    \n\t" \
"stxsdx       %%vs42, %%r9, %%r24               \n\t" \
"xxswapd      %%vs42, %%vs42                    \n\t" \
"stxsdx       %%vs43, %%r9, %%r25               \n\t" \
"xxswapd      %%vs43, %%vs43                    \n\t" \
"stxsdx       %%vs44, %%r9, %%r26               \n\t" \
"xxswapd      %%vs44, %%vs44                    \n\t" \
"stxsdx       %%vs45, %%r9, %%r27               \n\t" \
"xxswapd      %%vs45, %%vs45                    \n\t" \
"stxsdx       %%vs46, %%r9, %%r28               \n\t" \
"xxswapd      %%vs46, %%vs46                    \n\t" \
"stxsdx       %%vs47, %%r9, %%r29               \n\t" \
"xxswapd      %%vs47, %%vs47                    \n\t" \
"stxsdx       %%vs40, 0, %%r22                  \n\t" \
"stxsdx       %%vs41, 0, %%r23                  \n\t" \
"stxsdx       %%vs42, 0, %%r24                  \n\t" \
"stxsdx       %%vs43, 0, %%r25                  \n\t" \
"stxsdx       %%vs44, 0, %%r26                  \n\t" \
"stxsdx       %%vs45, 0, %%r27                  \n\t" \
"stxsdx       %%vs46, 0, %%r28                  \n\t" \
"stxsdx       %%vs47, 0, %%r29                  \n\t" \

#define GENLOAD_SCALE_UPDATE \
  LOADGEN_CMATRIX   \
  SCALEGEN_CMATRIX

#define PERMUTEALLVREG \
  "xxpermdi   %%vs32, %%vs8, %%vs0, 1           \n\t" \
  "xxpermdi   %%vs33, %%vs9, %%vs1, 1           \n\t" \
  "xxpermdi   %%vs34, %%vs10, %%vs2, 1          \n\t" \
  "xxpermdi   %%vs35, %%vs11, %%vs3, 1          \n\t" \
  "xxpermdi   %%vs36, %%vs12, %%vs4, 1          \n\t" \
  "xxpermdi   %%vs37, %%vs13, %%vs5, 1          \n\t" \
  "xxpermdi   %%vs38, %%vs14, %%vs6, 1          \n\t" \
  "xxpermdi   %%vs39, %%vs15, %%vs7, 1          \n\t" \
  "xxpermdi   %%vs40, %%vs0, %%vs8, 1           \n\t" \
  "xxpermdi   %%vs41, %%vs1, %%vs9, 1           \n\t" \
  "xxpermdi   %%vs42, %%vs2, %%vs10, 1          \n\t" \
  "xxpermdi   %%vs43, %%vs3, %%vs11, 1          \n\t" \
  "xxpermdi   %%vs44, %%vs4, %%vs12, 1          \n\t" \
  "xxpermdi   %%vs45, %%vs5, %%vs13, 1          \n\t" \
  "xxpermdi   %%vs46, %%vs6, %%vs14, 1          \n\t" \
  "xxpermdi   %%vs47, %%vs7, %%vs15, 1          \n\t" \
  "xxpermdi   %%vs48, %%vs24, %%vs16, 1         \n\t" \
  "xxpermdi   %%vs49, %%vs25, %%vs17, 1         \n\t" \
  "xxpermdi   %%vs50, %%vs26, %%vs18, 1         \n\t" \
  "xxpermdi   %%vs51, %%vs27, %%vs19, 1         \n\t" \
  "xxpermdi   %%vs52, %%vs28, %%vs20, 1         \n\t" \
  "xxpermdi   %%vs53, %%vs29, %%vs21, 1         \n\t" \
  "xxpermdi   %%vs54, %%vs30, %%vs22, 1         \n\t" \
  "xxpermdi   %%vs55, %%vs31, %%vs23, 1         \n\t" \
  "xxpermdi   %%vs56, %%vs16, %%vs24, 1         \n\t" \
  "xxpermdi   %%vs57, %%vs17, %%vs25, 1         \n\t" \
  "xxpermdi   %%vs58, %%vs18, %%vs26, 1         \n\t" \
  "xxpermdi   %%vs59, %%vs19, %%vs27, 1         \n\t" \
  "xxpermdi   %%vs60, %%vs20, %%vs28, 1         \n\t" \
  "xxpermdi   %%vs61, %%vs21, %%vs29, 1         \n\t" \
  "xxpermdi   %%vs62, %%vs22, %%vs30, 1         \n\t" \
  "xxpermdi   %%vs63, %%vs23, %%vs31, 1         \n\t"

#define COLSTORE_CMATRIX \
  "stxv              %%vs32, 0(%%r16)           \n\t" \
  "stxv              %%vs33, 16(%%r16)          \n\t" \
  "stxv              %%vs34, 32(%%r16)          \n\t" \
  "stxv              %%vs35, 48(%%r16)          \n\t" \
  "stxv              %%vs36, 64(%%r16)          \n\t" \
  "stxv              %%vs37, 80(%%r16)          \n\t" \
  "stxv              %%vs38, 96(%%r16)          \n\t" \
  "stxv              %%vs39, 112(%%r16)         \n\t" \
  "stxv              %%vs40, 0(%%r17)           \n\t" \
  "stxv              %%vs41, 16(%%r17)          \n\t" \
  "stxv              %%vs42, 32(%%r17)          \n\t" \
  "stxv              %%vs43, 48(%%r17)          \n\t" \
  "stxv              %%vs44, 64(%%r17)          \n\t" \
  "stxv              %%vs45, 80(%%r17)          \n\t" \
  "stxv              %%vs46, 96(%%r17)          \n\t" \
  "stxv              %%vs47, 112(%%r17)         \n\t" \
  "stxv              %%vs48, 0(%%r18)           \n\t" \
  "stxv              %%vs49, 16(%%r18)          \n\t" \
  "stxv              %%vs50, 32(%%r18)          \n\t" \
  "stxv              %%vs51, 48(%%r18)          \n\t" \
  "stxv              %%vs52, 64(%%r18)          \n\t" \
  "stxv              %%vs53, 80(%%r18)          \n\t" \
  "stxv              %%vs54, 96(%%r18)          \n\t" \
  "stxv              %%vs55, 112(%%r18)         \n\t" \
  "stxv              %%vs56, 0(%%r19)           \n\t" \
  "stxv              %%vs57, 16(%%r19)          \n\t" \
  "stxv              %%vs58, 32(%%r19)          \n\t" \
  "stxv              %%vs59, 48(%%r19)          \n\t" \
  "stxv              %%vs60, 64(%%r19)          \n\t" \
  "stxv              %%vs61, 80(%%r19)          \n\t" \
  "stxv              %%vs62, 96(%%r19)          \n\t" \
  "stxv              %%vs63, 112(%%r19)         \n\t" 



void bli_dgemm_power9_asm_16x4
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

  int *debug = malloc(sizeof(int));
  *debug = 0;

	// Typecast local copies of integers in case dim_t and inc_t are a
	// different size than is expected by load instructions.
  #if 1
	uint64_t k_iter = k0 / 4;
	uint64_t k_left = k0 % 4;
  #else
  uint64_t k_iter = 0;
	uint64_t k_left = k0;
  #endif
  uint64_t rs_c   = rs_c0;
	uint64_t cs_c   = cs_c0;

  #if 0
  printf("[Entering ukernel] k_iter = %ld | k_left = %ld | alpha = %lf | beta = %lf | rs_c = %ld | cs_c = %ld\n",
                                      k_iter, k_left, *alpha, *beta, rs_c, cs_c);
  #elif 0
  printf("[Entering kernel]\n");
  #endif

	__asm__ volatile
	(
	  "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
    "ld               %%r10, %8                     \n\t" // load cs_c
  	"ld               %%r9, %7                      \n\t" // load rs_c
    "                                               \n\t"
  	"ld               %%r17, %0                     \n\t" // load k_iter
  	"ld               %%r18, %1                     \n\t" // load k_left
  	"                                               \n\t"
  	"                                               \n\t"
  	"slwi             %%r10, %%r10, 3               \n\t" // mul by size of elem
  	"slwi             %%r9, %%r9, 3                 \n\t" // mul by size of elem
  	"                                               \n\t"
  	"                                               \n\t"
  	"ld               %%r7, %2                      \n\t" // load ptr of A
  	"ld               %%r8, %3                      \n\t" // load ptr of B
  	"ld               %%r16, %6                     \n\t" // load ptr for C
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    VSZEROOUT                                                 // Zero out vec regs
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "cmpwi            %%r0, %%r17, 0                \n\t"
  	"beq              %%r0, DPRELOOPKLEFT           \n\t"
  	"mtctr            %%r17                         \n\t"
  	"                                               \n\t"
  	"                                               \n\t"  
  	"DLOOPKITER:                                    \n\t" // Begin k_iter loop
    "                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
    LOADANDUPDATE1
    LOADANDUPDATE1
    LOADANDUPDATE1
    LOADANDUPDATE1
  	"                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
  	"bdnz             DLOOPKITER                    \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "DPRELOOPKLEFT:                                 \n\t"
  	"                                               \n\t"
  	"cmpwi            %%r0, %%r18, 0                \n\t"
  	"beq              %%r0, DPOSTACCUM              \n\t"
  	"mtctr            %%r18                         \n\t"
  	"                                               \n\t"
  	"DLOOPKLEFT:                                    \n\t" // EDGE LOOP
    "                                               \n\t"
    LOADANDUPDATE1
    "                                               \n\t"
  	"bdnz             DLOOPKLEFT                    \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"DPOSTACCUM:                                    \n\t"
  	"                                               \n\t"
  	"ld               %%r0, %4                      \n\t" // load ptr for alpha
  	"ld               %%r28, %5                     \n\t" // load ptr for beta
    "ld               %%r26, 0(%%r28)               \n\t" // load val of beta
  	"                                               \n\t"
  	"lxvdsx           %%vs60, 0, %%r0               \n\t" // splat alpha
  	"lxvdsx           %%vs59, 0, %%r28              \n\t" // splat beta
    "                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
    SCALEBYALPHA
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"cmpdi            %%r0, %%r26, 0                \n\t"
  	"beq              %%r0, DBETAZERO               \n\t" // jump to BZ case if beta = 0
  	"                                               \n\t"
  	"ld               %%r22, %6                     \n\t" // load ptr for C (used as offset)
  	"                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
  	"cmpwi            %%r0, %%r9, 8                 \n\t"
  	"beq              %%r0, DCOLSTOREDBNZ           \n\t" // jump to COLstore case, if rs_c = 8
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"DGENSTOREDBNZ:                                 \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t" // create offset regs
  	"slwi            %%r12, %%r9, 1                 \n\t"
  	"add             %%r23, %%r22, %%r12            \n\t" // c + rs_c * 2
  	"add             %%r24, %%r23, %%r12            \n\t" // c + rs_c * 4
  	"add             %%r25, %%r24, %%r12            \n\t" // c + rs_c * 6 
  	"add             %%r26, %%r25, %%r12            \n\t" // c + rs_c * 8
  	"add             %%r27, %%r26, %%r12            \n\t" // c + rs_c * 10
    "add             %%r28, %%r27, %%r12            \n\t" // c + rs_c * 12
    "add             %%r29, %%r28, %%r12            \n\t" // c + rs_c * 14
  	"                                               \n\t"
    GENLOAD_SCALE_UPDATE                                  // (1) load, scale, increment offsets
  	"                                              	\n\t"
    "xxpermdi     %%vs40, %%vs8, %%vs0, 1           \n\t" // permute
    "xxpermdi     %%vs41, %%vs9, %%vs1, 1           \n\t"
    "xxpermdi     %%vs42, %%vs10, %%vs2, 1          \n\t"
    "xxpermdi     %%vs43, %%vs11, %%vs3, 1          \n\t"
    "xxpermdi     %%vs44, %%vs12, %%vs4, 1          \n\t"
    "xxpermdi     %%vs45, %%vs13, %%vs5, 1          \n\t"
    "xxpermdi     %%vs46, %%vs14, %%vs6, 1          \n\t"
    "xxpermdi     %%vs47, %%vs15, %%vs7, 1          \n\t"
    "                                              	\n\t" 
	  GENADD_STORE                                          // add and store
    GEN_NEXT_COL_CMATRIX 
  	"                                               \n\t"
  	"                                               \n\t"
    GENLOAD_SCALE_UPDATE                                  // (2) load, scale, increment offsets
  	"                                               \n\t"
    "                                               \n\t"
  	"xxpermdi     %%vs40, %%vs0, %%vs8, 1           \n\t" // permute
    "xxpermdi     %%vs41, %%vs1, %%vs9, 1           \n\t"
    "xxpermdi     %%vs42, %%vs2, %%vs10, 1          \n\t"
    "xxpermdi     %%vs43, %%vs3, %%vs11, 1          \n\t"
    "xxpermdi     %%vs44, %%vs4, %%vs12, 1          \n\t"
    "xxpermdi     %%vs45, %%vs5, %%vs13, 1          \n\t"
    "xxpermdi     %%vs46, %%vs6, %%vs14, 1          \n\t"
    "xxpermdi     %%vs47, %%vs7, %%vs15, 1          \n\t"
    "                                              	\n\t" 
	  GENADD_STORE                                          // add and store
  	GEN_NEXT_COL_CMATRIX
    "                                               \n\t"
  	"                                               \n\t"
    GENLOAD_SCALE_UPDATE                                  // (3) load, scale, increment offsets
  	"                                               \n\t"
  	"xxpermdi     %%vs40, %%vs24, %%vs16, 1         \n\t" // permute
    "xxpermdi     %%vs41, %%vs25, %%vs17, 1         \n\t"
    "xxpermdi     %%vs42, %%vs26, %%vs18, 1         \n\t"
    "xxpermdi     %%vs43, %%vs27, %%vs19, 1         \n\t"
    "xxpermdi     %%vs44, %%vs28, %%vs20, 1         \n\t"
    "xxpermdi     %%vs45, %%vs29, %%vs21, 1         \n\t"
    "xxpermdi     %%vs46, %%vs30, %%vs22, 1         \n\t"
    "xxpermdi     %%vs47, %%vs31, %%vs23, 1         \n\t"
    "                                              	\n\t" 
	  GENADD_STORE                                          // add and store
  	GEN_NEXT_COL_CMATRIX
    "                                               \n\t"
  	"                                          	    \n\t"
    GENLOAD_SCALE_UPDATE                                  // (4) load, scale, increment offsets
  	"                                               \n\t"
    "xxpermdi     %%vs40, %%vs16, %%vs24, 1         \n\t" // permute
    "xxpermdi     %%vs41, %%vs17, %%vs25, 1         \n\t"
    "xxpermdi     %%vs42, %%vs18, %%vs26, 1         \n\t"
    "xxpermdi     %%vs43, %%vs19, %%vs27, 1         \n\t"
    "xxpermdi     %%vs44, %%vs20, %%vs28, 1         \n\t"
    "xxpermdi     %%vs45, %%vs21, %%vs29, 1         \n\t"
    "xxpermdi     %%vs46, %%vs22, %%vs30, 1         \n\t"
    "xxpermdi     %%vs47, %%vs23, %%vs31, 1         \n\t" 
    "                                              	\n\t"
	  GENADD_STORE                                          // add and store
  	"                                              	\n\t"
  	"                                              	\n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"b                DDONE                       	\n\t"
  	"                                              	\n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                              	\n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"DCOLSTOREDBNZ:                                	\n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                              	\n\t" // column offsets
  	"add              %%r23, %%r22, %%r10           \n\t" // c + cs_c
    "add              %%r24, %%r23, %%r10           \n\t" // c + cs_c * 2
  	"add              %%r25, %%r24, %%r10           \n\t" // c + cs_c * 3
  	"                                              	\n\t"
  	"DADDTOC:                                      	\n\t" // L.S.P.A.S.
  	"                                              	\n\t"
    "            	                                  \n\t"
    "            	                                  \n\t"
    "lxv              %%vs32, 0(%%r22)              \n\t" // load (1)
    "lxv              %%vs33, 16(%%r22)             \n\t" 
    "lxv              %%vs34, 32(%%r22)             \n\t" 
    "lxv              %%vs35, 48(%%r22)             \n\t" 
    "lxv              %%vs36, 64(%%r22)             \n\t" 
    "lxv              %%vs37, 80(%%r22)             \n\t" 
    "lxv              %%vs38, 96(%%r22)             \n\t" 
    "lxv              %%vs39, 112(%%r22)            \n\t" 
    "lxv              %%vs40, 0(%%r23)              \n\t" 
    "lxv              %%vs41, 16(%%r23)             \n\t" 
    "lxv              %%vs42, 32(%%r23)             \n\t" 
    "lxv              %%vs43, 48(%%r23)             \n\t" 
    "lxv              %%vs44, 64(%%r23)             \n\t" 
    "lxv              %%vs45, 80(%%r23)             \n\t" 
    "lxv              %%vs46, 96(%%r23)             \n\t" 
    "lxv              %%vs47, 112(%%r23)            \n\t"
    "                                               \n\t"
 SCALECOL_CMATRIX                                         // scale (1)
  	"            	                                  \n\t"
    "xxpermdi         %%vs48, %%vs8, %%vs0, 1   	  \n\t" // permute (1)
    "xxpermdi         %%vs49, %%vs9, %%vs1, 1   	  \n\t"
    "xxpermdi         %%vs50, %%vs10, %%vs2, 1   	  \n\t"
    "xxpermdi         %%vs51, %%vs11, %%vs3, 1   	  \n\t"
    "xxpermdi         %%vs52, %%vs12, %%vs4, 1   	  \n\t"
    "xxpermdi         %%vs53, %%vs13, %%vs5, 1   	  \n\t"
    "xxpermdi         %%vs54, %%vs14, %%vs6, 1   	  \n\t"
    "xxpermdi         %%vs55, %%vs15, %%vs7, 1   	  \n\t"
    "            	                                  \n\t"
    "xxpermdi         %%vs56, %%vs0, %%vs8, 1   	  \n\t"
    "xxpermdi         %%vs57, %%vs1, %%vs9, 1   	  \n\t"
    "xxpermdi         %%vs58, %%vs2, %%vs10, 1   	  \n\t"
    "xxpermdi         %%vs59, %%vs3, %%vs11, 1   	  \n\t"
    "xxpermdi         %%vs60, %%vs4, %%vs12, 1   	  \n\t"
    "xxpermdi         %%vs61, %%vs5, %%vs13, 1   	  \n\t"
    "xxpermdi         %%vs62, %%vs6, %%vs14, 1   	  \n\t"
    "xxpermdi         %%vs63, %%vs7, %%vs15, 1   	  \n\t"
    "            	                                  \n\t"
    "            	                                  \n\t"
  	COL_ADD_STORE                                         // add then store (1)
    "            	                                  \n\t"
    "lxvdsx           %%vs59, 0, %%r28              \n\t" // resplat beta 
    "                                               \n\t"
  	"                                               \n\t"
    "lxv              %%vs32, 0(%%r24)              \n\t" // load (2)
    "lxv              %%vs33, 16(%%r24)             \n\t" 
    "lxv              %%vs34, 32(%%r24)             \n\t" 
    "lxv              %%vs35, 48(%%r24)             \n\t" 
    "lxv              %%vs36, 64(%%r24)             \n\t" 
    "lxv              %%vs37, 80(%%r24)             \n\t" 
    "lxv              %%vs38, 96(%%r24)             \n\t" 
    "lxv              %%vs39, 112(%%r24)            \n\t" 
    "lxv              %%vs40, 0(%%r25)              \n\t" 
    "lxv              %%vs41, 16(%%r25)             \n\t" 
    "lxv              %%vs42, 32(%%r25)             \n\t" 
    "lxv              %%vs43, 48(%%r25)             \n\t" 
    "lxv              %%vs44, 64(%%r25)             \n\t" 
    "lxv              %%vs45, 80(%%r25)             \n\t" 
    "lxv              %%vs46, 96(%%r25)             \n\t" 
    "lxv              %%vs47, 112(%%r25)            \n\t"
  	"                                               \n\t"
    SCALECOL_CMATRIX                                      // scale (2)
    "                                               \n\t"
  	"                                               \n\t"
  	"xxpermdi         %%vs48, %%vs24, %%vs16, 1  	  \n\t" // permute (2)
    "xxpermdi         %%vs49, %%vs25, %%vs17, 1  	  \n\t"
    "xxpermdi         %%vs50, %%vs26, %%vs18, 1  	  \n\t"
    "xxpermdi         %%vs51, %%vs27, %%vs19, 1  	  \n\t"
    "xxpermdi         %%vs52, %%vs28, %%vs20, 1  	  \n\t"
    "xxpermdi         %%vs53, %%vs29, %%vs21, 1     \n\t"
    "xxpermdi         %%vs54, %%vs30, %%vs22, 1   	\n\t"
    "xxpermdi         %%vs55, %%vs31, %%vs23, 1   	\n\t"
    "            	                                  \n\t"
    "xxpermdi         %%vs56, %%vs16, %%vs24, 1  	  \n\t"
    "xxpermdi         %%vs57, %%vs17, %%vs25, 1  	  \n\t"
    "xxpermdi         %%vs58, %%vs18, %%vs26, 1  	  \n\t"
    "xxpermdi         %%vs59, %%vs19, %%vs27, 1  	  \n\t"
    "xxpermdi         %%vs60, %%vs20, %%vs28, 1  	  \n\t"
    "xxpermdi         %%vs61, %%vs21, %%vs29, 1  	  \n\t"
    "xxpermdi         %%vs62, %%vs22, %%vs30, 1     \n\t"
    "xxpermdi         %%vs63, %%vs23, %%vs31, 1   	\n\t"
    "            	                                  \n\t"
    "            	                                  \n\t"
  	COL_ADD_STORE                                         // add then store (2)
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"b                DDONE                         \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"DBETAZERO:                                     \n\t" // beta=0 case
  	"                                               \n\t" 
    PERMUTEALLVREG
    "                                               \n\t"
    "                                               \n\t"
    "                                               \n\t"
  	"cmpwi            %%r0, %%r9, 8                 \n\t" // if rs_c == 8,
  	"beq              DCOLSTORED                    \n\t" // C is col stored
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"                                               \n\t"
  	"DGENSTORED:                                    \n\t"
  	"                                               \n\t"
  	"ld              %%r22, %6                      \n\t" // load c
  	"slwi            %%r12, %%r9, 1                 \n\t"
  	"add             %%r23, %%r22, %%r12            \n\t" // c + rs_c * 2
  	"add             %%r24, %%r23, %%r12            \n\t" // c + rs_c * 4
  	"add             %%r25, %%r24, %%r12            \n\t" // c + rs_c * 6 
  	"add             %%r26, %%r25, %%r12            \n\t" // c + rs_c * 8
  	"add             %%r27, %%r26, %%r12            \n\t" // c + rs_c * 10
    "add             %%r28, %%r27, %%r12            \n\t" // c + rs_c * 12
    "add             %%r29, %%r28, %%r12            \n\t" // c + rs_c * 14
  	"                                               \n\t"
  	"                                               \n\t"
    "stxsdx          %%vs32, %%r9, %%r22            \n\t" 
    "xxswapd         %%vs32, %%vs32		              \n\t" 
    "stxsdx          %%vs32, 0, %%r22               \n\t" 
    "stxsdx          %%vs33, %%r9, %%r23            \n\t" 
    "xxswapd         %%vs33, %%vs33		              \n\t" 
    "stxsdx          %%vs33, 0, %%r23               \n\t" 
    "stxsdx          %%vs34, %%r9, %%r24            \n\t" 
    "xxswapd         %%vs34, %%vs34		              \n\t" 
    "stxsdx          %%vs34, 0, %%r24               \n\t" 
    "stxsdx          %%vs35, %%r9, %%r25            \n\t" 
    "xxswapd         %%vs35, %%vs35		              \n\t" 
    "stxsdx          %%vs35, 0, %%r25               \n\t" 
    "stxsdx          %%vs36, %%r9, %%r26            \n\t" 
    "xxswapd         %%vs36, %%vs36		              \n\t" 
    "stxsdx          %%vs36, 0, %%r26               \n\t" 
    "stxsdx          %%vs37, %%r9, %%r27            \n\t" 
    "xxswapd         %%vs37, %%vs37		              \n\t" 
    "stxsdx          %%vs37, 0, %%r27               \n\t" 
    "stxsdx          %%vs38, %%r9, %%r28            \n\t" 
    "xxswapd         %%vs38, %%vs38		              \n\t" 
    "stxsdx          %%vs38, 0, %%r28               \n\t" 
    "stxsdx          %%vs39, %%r9, %%r29            \n\t" 
    "xxswapd         %%vs39, %%vs39		              \n\t" 
    "stxsdx          %%vs39, 0, %%r29               \n\t" 
    "                                               \n\t"
    GEN_NEXT_COL_CMATRIX 
    "                                               \n\t"
    "stxsdx          %%vs40, %%r9, %%r22            \n\t" 
    "xxswapd         %%vs40, %%vs40		              \n\t" 
    "stxsdx          %%vs40, 0, %%r22               \n\t" 
    "stxsdx          %%vs41, %%r9, %%r23            \n\t" 
    "xxswapd         %%vs41, %%vs41		              \n\t" 
    "stxsdx          %%vs41, 0, %%r23               \n\t" 
    "stxsdx          %%vs42, %%r9, %%r24            \n\t" 
    "xxswapd         %%vs42, %%vs42		              \n\t" 
    "stxsdx          %%vs42, 0, %%r24               \n\t" 
    "stxsdx          %%vs43, %%r9, %%r25            \n\t" 
    "xxswapd         %%vs43, %%vs43		              \n\t" 
    "stxsdx          %%vs43, 0, %%r25               \n\t" 
    "stxsdx          %%vs44, %%r9, %%r26            \n\t" 
    "xxswapd         %%vs44, %%vs44		              \n\t" 
    "stxsdx          %%vs44, 0, %%r26               \n\t" 
    "stxsdx          %%vs45, %%r9, %%r27            \n\t" 
    "xxswapd         %%vs45, %%vs45		              \n\t" 
    "stxsdx          %%vs45, 0, %%r27               \n\t" 
    "stxsdx          %%vs46, %%r9, %%r28            \n\t" 
    "xxswapd         %%vs46, %%vs46		              \n\t" 
    "stxsdx          %%vs46, 0, %%r28               \n\t" 
    "stxsdx          %%vs47, %%r9, %%r29            \n\t" 
    "xxswapd         %%vs47, %%vs47		              \n\t" 
    "stxsdx          %%vs47, 0, %%r29               \n\t" 
    "                                               \n\t"
    GEN_NEXT_COL_CMATRIX 
    "                                               \n\t"
    "stxsdx          %%vs48, %%r9, %%r22            \n\t" 
    "xxswapd         %%vs48, %%vs48		              \n\t" 
    "stxsdx          %%vs48, 0, %%r22               \n\t" 
    "stxsdx          %%vs49, %%r9, %%r23            \n\t" 
    "xxswapd         %%vs49, %%vs49		              \n\t" 
    "stxsdx          %%vs49, 0, %%r23               \n\t" 
    "stxsdx          %%vs50, %%r9, %%r24            \n\t" 
    "xxswapd         %%vs50, %%vs50		              \n\t" 
    "stxsdx          %%vs50, 0, %%r24               \n\t" 
    "stxsdx          %%vs51, %%r9, %%r25            \n\t" 
    "xxswapd         %%vs51, %%vs51		              \n\t" 
    "stxsdx          %%vs51, 0, %%r25               \n\t" 
    "stxsdx          %%vs52, %%r9, %%r26            \n\t" 
    "xxswapd         %%vs52, %%vs52		              \n\t" 
    "stxsdx          %%vs52, 0, %%r26               \n\t" 
    "stxsdx          %%vs53, %%r9, %%r27            \n\t" 
    "xxswapd         %%vs53, %%vs53		              \n\t" 
    "stxsdx          %%vs53, 0, %%r27               \n\t" 
    "stxsdx          %%vs54, %%r9, %%r28            \n\t" 
    "xxswapd         %%vs54, %%vs54		              \n\t" 
    "stxsdx          %%vs54, 0, %%r28               \n\t" 
    "stxsdx          %%vs55, %%r9, %%r29            \n\t" 
    "xxswapd         %%vs55, %%vs55		              \n\t" 
    "stxsdx          %%vs55, 0, %%r29               \n\t" 
    "                                               \n\t"
    GEN_NEXT_COL_CMATRIX 
    "                                               \n\t"
    "stxsdx          %%vs56, %%r9, %%r22            \n\t" 
    "xxswapd         %%vs56, %%vs56		              \n\t" 
    "stxsdx          %%vs56, 0, %%r22               \n\t" 
    "stxsdx          %%vs57, %%r9, %%r23            \n\t" 
    "xxswapd         %%vs57, %%vs57		              \n\t" 
    "stxsdx          %%vs57, 0, %%r23               \n\t" 
    "stxsdx          %%vs58, %%r9, %%r24            \n\t" 
    "xxswapd         %%vs58, %%vs58		              \n\t" 
    "stxsdx          %%vs58, 0, %%r24               \n\t" 
    "stxsdx          %%vs59, %%r9, %%r25            \n\t" 
    "xxswapd         %%vs59, %%vs59		              \n\t" 
    "stxsdx          %%vs59, 0, %%r25               \n\t" 
    "stxsdx          %%vs60, %%r9, %%r26            \n\t" 
    "xxswapd         %%vs60, %%vs60		              \n\t" 
    "stxsdx          %%vs60, 0, %%r26               \n\t" 
    "stxsdx          %%vs61, %%r9, %%r27            \n\t" 
    "xxswapd         %%vs61, %%vs61		              \n\t" 
    "stxsdx          %%vs61, 0, %%r27               \n\t" 
    "stxsdx          %%vs62, %%r9, %%r28            \n\t" 
    "xxswapd         %%vs62, %%vs62		              \n\t" 
    "stxsdx          %%vs62, 0, %%r28               \n\t" 
    "stxsdx          %%vs63, %%r9, %%r29            \n\t" 
    "xxswapd         %%vs63, %%vs63		              \n\t" 
    "stxsdx          %%vs63, 0, %%r29               \n\t" 
  	"                                               \n\t"
  	"b               DDONE                          \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
    "                                               \n\t"
  	"                                               \n\t"
  	"DCOLSTORED:                                    \n\t"
  	"                                               \n\t" // create offset regs
  	"add              %%r17, %%r16, %%r10           \n\t" // c + cs_c
  	"add              %%r18, %%r17, %%r10           \n\t" // c + cs_c * 2 
  	"add              %%r19, %%r18, %%r10           \n\t" // c + cs_c * 3
  	"                                               \n\t"
    "                                               \n\t"
    COLSTORE_CMATRIX
  	"                                               \n\t"
  	"DDONE:                                         \n\t"  
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
	  "m" (cs_c),   // 8
    "m" (debug)   // 9
                  /*,   
	  "m" (b_next), // 9
	  "m" (a_next)*/  // 10
	: // register clobber list
  /* unclobberable regs: r2, r3, r4, r5, r6, r13, r14, r15, r30, r31 */
  "r0",  "r7", "r8", "r9",
  "r10", "r12","r16", "r17", "r18", "r19", 
  "r20", "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28"

  #if XLC
  ,"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"
  , "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19"
  , "f20" ,"f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29"
  , "f30" ,"f31"
  , "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"
  , "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19"
  , "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"
  , "v30", "v31"
  #else
  , "vs0", "vs1", "vs2", "vs3", "vs4", "vs5", "vs6", "vs7", "vs8", "vs9"
  , "vs10", "vs11", "vs12", "vs13", "vs14", "vs15", "vs16", "vs17", "vs18", "vs19"
  , "vs20", "vs21", "vs22", "vs23", "vs24", "vs25", "vs26", "vs27", "vs28", "vs29"
  , "vs30", "vs31", "vs32", "vs33", "vs34", "vs35", "vs36", "vs37", "vs38", "vs39"
  , "vs40", "vs41", "vs42", "vs43", "vs44", "vs45", "vs46", "vs47", "vs48", "vs49"
  , "vs50", "vs51", "vs52", "vs53", "vs54", "vs55", "vs56", "vs57", "vs58", "vs58"
  , "vs60", "vs61", "vs62", "vs63"
  #endif
  );

  if(*debug)
    printf("it worked!\n");
}
