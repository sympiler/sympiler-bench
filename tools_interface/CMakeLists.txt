cmake_minimum_required(VERSION 2.8)



include_directories(includes
        )

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
)

project(tools_demo)

if (mkl_exist)
set(SPTRSVD_SOURCE_FILES
#        ${SPARSE_UTIL_SRC}/sparse_utilities.cpp
#        ${SPARSE_UTIL_SRC}/sparse_inspector.cpp
#        ${SPARSE_UTIL_SRC}/sparse_io.cpp
#        ${SPARSE_UTIL_SRC}/test_utils.cpp
#        ${SPARSE_UTIL_SRC}/utils.cpp
#        ${SPARSE_EXTN_SRC}/mmio.cpp
        ${METIS_SOURCE_FILES}
        ${ALL_SOURCE_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/../sym_interface/FusionDemo.cpp
        )

add_executable(sptrsv_demo ${SPTRSVD_SOURCE_FILES}
        sptrsv_demo.cpp )

target_link_libraries(sptrsv_demo
        lbc
        ${METIS_LIBRARY} ${METIS_LIBRARIES})

endif()


if (suitesparse_exist)

endif()