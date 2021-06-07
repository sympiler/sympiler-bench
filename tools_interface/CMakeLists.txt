cmake_minimum_required(VERSION 2.8)



include_directories(includes
        ${CMAKE_CURRENT_LIST_DIR}/../common/
        )

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
)

set(fusion_demo_src  "${CMAKE_CURRENT_LIST_DIR}/../common/FusionDemo.cpp" )

project(tools_demo)

if (mkl_exist)

add_executable(sptrsv_demo ${SPTRSVD_SOURCE_FILES}
        sptrsv_demo.cpp )

target_link_libraries(sptrsv_demo
        lbc
        ${METIS_LIBRARY} ${METIS_LIBRARIES})

endif()


if(EXISTS "$ENV{SUITEROOT}/include/" AND EXISTS "$ENV{SUITEROOT}/lib/")
    include_directories($ENV{SUITEROOT}/include/)
    link_directories($ENV{SUITEROOT}/lib/)

    add_executable(suite_cholmod
            suitesparse_cholesky.cpp
            ${fusion_demo_src}
            )
    target_link_libraries(suite_cholmod
            cholmod
            sympiler_utils
            lbc)
endif()



find_package(MKL REQUIRED)
if(MKL_FOUND)
include_directories(
        ${MKL_INCLUDE_DIR}
)
add_executable(mkl_pardiso_cholesky
        mkl_pardiso_cholesky.cpp
        ${fusion_demo_src}
        )

    target_link_libraries(mkl_pardiso_cholesky
            ${MKL_LIBRARIES}
            sympiler_utils
            lbc
            sym_sparse_blas  # added for spmv testing
    )

endif()


