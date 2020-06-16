##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources("${PROJECT_NAME}"
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_addv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_amaxv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpbyv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_axpyv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_copyv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_dotxv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_invertv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scal2v_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_scalv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_setv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_subv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_swapv_ref.c
    ${CMAKE_CURRENT_SOURCE_DIR}/bli_xpbyv_ref.c
    )
