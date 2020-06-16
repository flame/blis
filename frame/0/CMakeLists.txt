##Copyright (C) 2020, Advanced Micro Devices, Inc ##

target_sources("${PROJECT_NAME}"
    PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0.h
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_check.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_check.h
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_fpa.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_fpa.h
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_ft.h
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_oapi.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_oapi.h
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_tapi.c
	${CMAKE_CURRENT_SOURCE_DIR}/bli_l0_tapi.h
	)

	add_subdirectory(copysc)
