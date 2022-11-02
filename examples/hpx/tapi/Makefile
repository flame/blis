-include $(SHARE_PATH)/common.mk

all:
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application_debug` -DHPX_APPLICATION_NAME=00level1v -o 00level1v 00level1v.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application_debug` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application_debug` -DHPX_APPLICATION_NAME=01level1m  -o 01level1m 01level1m.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application_debug` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application_debug` -DHPX_APPLICATION_NAME=02level1m_diag -o 02level1m_diag 02level1m_diag.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application_debug` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application_debug` -DHPX_APPLICATION_NAME=03level2 -o 03level2 03level2.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application_debug` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application_debug` -DHPX_APPLICATION_NAME=04level3 -o 04level3 04level3.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application_debug` -lblis
	$(CXX) -O3 -I../../../include/x86_64 `pkg-config --cflags hpx_application_debug` -DHPX_APPLICATION_NAME=05util -o 05util 05util.cpp -L../../../lib/x86_64 `pkg-config --libs hpx_application_debug` -lblis

clean:
	rm 00level1v 02level1m_diag  03level2  04level3  05util
