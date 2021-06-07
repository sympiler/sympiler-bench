cmake_minimum_required(VERSION 2.8)


find_package(METIS OPTIONAL_COMPONENTS)
if(NOT METIS_FOUND)
    set(METIS_INCLUDE_DIR ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include/)
    set(METIS_INCLUDES ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include/)
    set(METIS_LIBRARY ${SYMPILER_LBC_ROOT}/lib/)
    set(METIS_LIBRARIES ${SYMPILER_LBC_ROOT}/lib/)
    set(METIS_LIB metis)
    set(METIS_FOUND TRUE)
endif()
add_definitions(-DMETIS)

include_directories(includes
        ${SYMPILER_UTIL_INC}
        ${SYMPILER_UTILE_INC}
        ${SYMPILER_SPARSE_BLAS_INC}
        ${SYMPILER_SPARSE_BLASP_INC}
        ${SYMPILER_LBC_ROOT}/src/include/
        ${METIS_INCLUDE_DIR}
        ${METIS_INCLUDES}
        )

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
        ${METIS_LIBRARY} ${METIS_LIBRARIES}
)

set(ALL_LIBS
        ${MKL_LIB_NAME}
        ${PAPI_LIB_NAME}
        ${PAPI_LIB_NAME2}
        ${METIS_LIB_NAME}
        )

link_libraries(
        ${METIS_LIB}
        lbc
        sym_sparse_blas
        sym_dense_blas
        sympiler_utils
)


project(sympiler_demo)

add_executable(sym_sptrsv
        sptrsv_demo.cpp FusionDemo.cpp)

add_executable(sym_cholesky
        cholesky_demo.cpp FusionDemo.cpp)


