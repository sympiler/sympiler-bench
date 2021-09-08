cmake_minimum_required(VERSION 2.8)



find_package(METIS OPTIONAL_COMPONENTS)

if(NOT METIS_FOUND)
    set(METIS_INCLUDE_DIR ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include/)
    set(METIS_INCLUDES ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include/)
    set(METIS_LIBRARY ${SYMPILER_LBC_ROOT}/lib/libmetis.so)
    set(METIS_LIBRARIES ${SYMPILER_LBC_ROOT}/lib/libmetis.so )
    set(METIS_LIB metis)
    set(METIS_FOUND  TRUE)
else()
    set(METIS_LIB ${METIS_LIBRARY})
endif()
add_definitions(-DMETIS)

include_directories(includes
        ${SYMPILER_INC}
        ${SYMPILER_LBC_INC}
        ${METIS_INCLUDE_DIR}
        ${METIS_INCLUDES}
        )


link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
 #       ${METIS_LIBRARY} ${METIS_LIBRARIES}
)

set(ALL_LIBS
        ${MKL_LIB_NAME}
        ${PAPI_LIB_NAME}
        ${PAPI_LIB_NAME2}
        ${METIS_LIB}
        )



project(sympiler_demo)

add_executable(sym_sptrsv
        sptrsv_demo.cpp ../common/FusionDemo.cpp)

target_link_libraries(sym_sptrsv PRIVATE
        lbc
        sym_sparse_blas
        ${METIS_LIB}
        )

add_executable(sym_cholesky
        cholesky_demo.cpp ../common/FusionDemo.cpp)
target_link_libraries(sym_cholesky PRIVATE
        lbc
        sym_sparse_blas
        ${METIS_LIB}
)

