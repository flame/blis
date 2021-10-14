#
#
#  BLIS    
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2020, Advanced Micro Devices, Inc.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# FLAGS that are specific to the 'zen3' architecture are added here.
# FLAGS that are common for all the AMD architectures are present in
# config/zen/amd_config.mk.

# Declare the name of the current configuration and add it to the
# running list of configurations included by common.mk.
THIS_CONFIG    := zen3 
#CONFIGS_INCL   += $(THIS_CONFIG)

#
# --- Determine the C compiler and related flags ---
#

# NOTE: The build system will append these variables with various
# general-purpose/configuration-agnostic flags in common.mk. You
# may specify additional flags here as needed.
CPPROCFLAGS    :=
CMISCFLAGS     :=
CPICFLAGS      :=
CWARNFLAGS     :=

ifneq ($(DEBUG_TYPE),off)
CDBGFLAGS      := -g
endif

ifeq ($(DEBUG_TYPE),noopt)
COPTFLAGS      := -O0
else
#frame pointers are needed to execution tracing
ifeq ($(ETRACE_ENABLE),1)
COPTFLAGS      := -O3
else
COPTFLAGS      := -O3 -fomit-frame-pointer
endif
endif


#
# --- Enable ETRACE across the library if enabled ETRACE_ENABLE=[0,1] -----------------------
#

ifeq ($(ETRACE_ENABLE),1)
CDBGFLAGS += -pg -finstrument-functions -DAOCL_DTL_AUTO_TRACE_ENABLE
LDFLAGS += -ldl
endif

# Flags specific to optimized kernels.
CKOPTFLAGS     := $(COPTFLAGS)
ifeq ($(CC_VENDOR),gcc)
GCC_VERSION := $(strip $(shell gcc -dumpversion | cut -d. -f1))
#gcc or clang version must be atleast 4.0
# gcc 9.0 or later:
ifeq ($(shell test $(GCC_VERSION) -ge 9; echo $$?),0)
CKVECFLAGS     += -march=znver2
else
# If gcc is older than 9.1.0 but at least 6.1.0, then we can use -march=znver1
# as the fallback option.
CRVECFLAGS += -march=znver1 -mno-avx256-split-unaligned-store
CKVECFLAGS += -march=znver1 -mno-avx256-split-unaligned-store
endif
else
ifeq ($(CC_VENDOR),clang)
ifeq ($(strip $(shell clang -v |&head -1 |grep -c 'AOCC.LLVM.2.0.0')),1)
CKVECFLAGS += -march=znver2
else
#if compiling with clang
VENDOR_STRING := $(strip $(shell ${CC_VENDOR} --version | egrep -o '[0-9]+\.[0-9]+\.?[0-9]*'))
CC_MAJOR := $(shell (echo ${VENDOR_STRING} | cut -d. -f1))
#clang 9.0 or later:
ifeq ($(shell test $(CC_MAJOR) -ge 9; echo $$?),0)
CKVECFLAGS += -march=znver2
else
CKVECFLAGS += -march=znver1
endif
endif
endif
endif

# Flags specific to reference kernels.
CROPTFLAGS     := $(CKOPTFLAGS)
CRVECFLAGS     := $(CKVECFLAGS)

# Store all of the variables here to new variables containing the
# configuration name.
$(eval $(call store-make-defs,$(THIS_CONFIG)))

