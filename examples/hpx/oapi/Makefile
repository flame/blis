#   BLIS
#   An object-based framework for developing high-performance BLAS-like
#   libraries.
#
#   Copyright (C) 2014, The University of Texas at Austin
#   Copyright (C) 2022 Tactical Computing Laboratories, LLC
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are
#   met:
#    - Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    - Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    - Neither the name of The University of Texas nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
-include $(SHARE_PATH)/common.mk

all:
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=00obj_basic -o 00obj_basic 00obj_basic.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=01obj_attach -o 01obj_attach 01obj_attach.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=02obj_ij -o 02obj_ij 02obj_ij.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=03obj_view -o 03obj_view 03obj_view.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=04level0 -o 04level0 04level0.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=05level1v -o 05level1v 05level1v.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=06level1m -o 06level1m 06level1m.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=07level1m_diag -o 07level1m_diag 07level1m_diag.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=08level2 -o 08level2 08level2.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application` -DHPX_APPLICATION_NAME=09level3 -o09level3 09level3.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application` -lblis

clean:
	rm 00obj_basic 02obj_ij 04level0 06level1m 08level2 01obj_attach 03obj_view 05level1v 07level1m_diag 09level3

