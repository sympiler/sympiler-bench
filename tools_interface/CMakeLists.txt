cmake_minimum_required(VERSION 2.8)



include_directories(includes
        ${CMAKE_CURRENT_LIST_DIR}/../common/
        )

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
)

set(fusion_demo_src  "${CMAKE_CURRENT_LIST_DIR}/../common/FusionDemo.cpp" )

project(tools_demo)


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


set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITEROOT}/include)
set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITEROOT}/lib)
find_package(SuiteSparse OPTIONAL_COMPONENTS)
if(SUITESPARSE_FOUND )
    include_directories(${SUITESPARSE_INCLUDE_DIRS})

    add_executable(suite_cholmod
            suitesparse_cholesky.cpp
            )
#       message(" ===> ${SUITESPARSE_LIBRARIES}")
    target_link_libraries(suite_cholmod
            #cholmod
            ${MKL_LIBRARIES}
            ${SUITESPARSE_LIBRARIES}
            )
endif()





set(PARDISO_LIB_DIR $ENV{PARDISO_LIB})
#message("====${PARDISO_LIB_DIR}")
#message("===> ${MKL_LIBRARIES}")

if(EXISTS "${PARDISO_LIB_DIR}" AND MKL_FOUND)
    message(STATUS "PARDISO lib path is set to ${PARDISO_LIB_DIR}")
    project(pardiso6_cholesky)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp ")


    link_directories(${PARDISO_LIB_DIR}
            "/usr/lib/gcc/x86_64-linux-gnu/5/"
           # "/home/kazem/anaconda3/envs/ml_env/lib/"
            )
    add_executable(pardiso6_cholesky ${SOURCE_FILES} pardiso_cholesky.cpp)
    target_link_libraries(pardiso6_cholesky
            gfortran
#            "/home/kazem/anaconda3/envs/ml_env/lib/libgfortran.so"
            pardiso600-GNU720-X86-64
            #pardiso600-GNU800-X86-64
            ${MKL_LIBRARIES}
            )
endif()