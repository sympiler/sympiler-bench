cmake_minimum_required(VERSION 2.8)



include_directories(includes
        ${SYMPILER_UTIL_INC}
        ${SYMPILER_UTILE_INC}
        ${SYMPILER_SPARSE_BLAS_INC}
        ${SYMPILER_SPARSE_BLASP_INC}
        ${SYMPILER_LBC_INC}
        ${METIS_INCLUDE_DIR}
        ${METIS_INCLUDES}
        )

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
)

link_libraries(
        lbc
        sym_sparse_blas
        sym_dense_blas
        sympiler_utils
        ${METIS_LIBRARY} ${METIS_LIBRARIES})

project(sympiler_demo)
set(SYMPILERD_SOURCE_FILES
        ${METIS_SOURCE_FILES}
        ${ALL_SOURCE_FILES}
        FusionDemo.cpp
        )

add_executable(sym_sptrsv ${SYMPILERD_SOURCE_FILES}
        sptrsv_demo.cpp )

add_executable(sym_cholesky ${SYMPILERD_SOURCE_FILES}
        cholesky_demo.cpp )
