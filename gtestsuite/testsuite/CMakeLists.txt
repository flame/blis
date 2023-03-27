# Fetch and Build GTest at configure time
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Return the list of the subdirectories in the directory curdir.
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(DIRS ${CMAKE_CURRENT_SOURCE_DIR})

set(target_name "testsuite")
foreach(dir ${DIRS})
    add_custom_target(${target_name}.${dir})
    SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    foreach(subdir ${SUBDIRS})
      file(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${subdir}/*.cpp)
      if(files)
        add_executable(${target_name}.${dir}.${subdir} ${files})
        set_target_properties(${target_name}.${dir}.${subdir} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
        set_target_properties(${target_name}.${dir}.${subdir} PROPERTIES OUTPUT_NAME ${target_name}.${dir}.${subdir})
        target_include_directories(${target_name}.${dir}.${subdir} PUBLIC ${BLIS_INCLUDE} ${CMAKE_SOURCE_DIR}/testinghelpers/inc ${CMAKE_SOURCE_DIR}/testsuite/)
        target_link_libraries(${target_name}.${dir}.${subdir} gtest gtest_main testinghelpers ${Blis_LIBRARY} ${COMMON_LIBS})
        # if we test serial BLIS, but MKL is used as a reference we still need to set up OpenMP.
        if( (ENABLE_THREADING STREQUAL "openmp") OR (REF_CBLAS STREQUAL "MKL"))
            if(LINUX)
                if(OpenMP_LIBRARY STREQUAL "GNU")
                    target_link_libraries(${target_name}.${dir}.${subdir} -fopenmp)
                else()
                    target_link_libraries(${target_name}.${dir}.${subdir} iomp5)
                endif()
            endif()
        endif()
        if(ENABLE_ASAN)
          target_link_libraries(${target_name}.${dir}.${subdir} -fsanitize=address)
        endif()
        if(TEST_INTERFACE STREQUAL "BLAS")
          target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC TEST_BLAS)
        elseif(TEST_INTERFACE STREQUAL "CBLAS")
          target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC TEST_CBLAS)
        else() # BLIS_TYPED option
          target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC TEST_BLIS_TYPED)
        endif()
        target_compile_definitions(${target_name}.${dir}.${subdir} PUBLIC ELEMENT_TYPE='${ELEMENT_TYPE}')
        add_test(NAME ${target_name}.${dir}.${subdir} COMMAND ${target_name}.${dir}.${subdir})
        add_dependencies(${target_name}.${dir} ${target_name}.${dir}.${subdir})
      endif()
    endforeach()
endforeach()


