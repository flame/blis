CXX=g++
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

