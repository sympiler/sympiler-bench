cmake_minimum_required(VERSION 3.15)



include_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../common/
        )

link_directories(
        ${CMAKE_CURRENT_LIST_DIR}/../lib/
)

set(fusion_demo_src  "${CMAKE_CURRENT_LIST_DIR}/../common/FusionDemo.cpp" )

project(tools_demo)

#set(MKL_THREADING OMP)
if (NOT TARGET MKL::MKL)
    set(MKL_INTERFACE "lp64")
    find_package(MKL CONFIG)
endif()

if(TARGET MKL::MKL )
    add_executable(mkl_pardiso_cholesky
            mkl_pardiso_cholesky.cpp
            ${fusion_demo_src}
            )
    target_include_directories(mkl_pardiso_cholesky PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            )
    target_link_libraries(mkl_pardiso_cholesky PRIVATE
            $<LINK_ONLY:MKL::MKL>
            sym_sparse_blas  # added for spmv testing
            aggregation
            )
    target_compile_definitions(mkl_pardiso_cholesky PRIVATE
            "MKL_BLAS")

endif()


set(SUITESPARSE_INCLUDE_DIR_HINTS $ENV{SUITEROOT}/include)
set(SUITESPARSE_LIBRARY_DIR_HINTS $ENV{SUITEROOT}/lib)
find_package(SuiteSparse OPTIONAL_COMPONENTS)
if(SUITESPARSE_FOUND )
    add_executable(suite_cholmod
            suitesparse_cholesky.cpp
            )
    target_include_directories(suite_cholmod PRIVATE
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>
            ${SUITESPARSE_INCLUDE_DIRS}
            )
    target_link_libraries(suite_cholmod PRIVATE
            #cholmod
            $<LINK_ONLY:MKL::MKL>
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
            )
    add_executable(pardiso6_cholesky ${SOURCE_FILES} pardiso_cholesky.cpp)
    target_link_libraries(pardiso6_cholesky
            gfortran
            pardiso600-GNU720-X86-64
            #pardiso600-GNU800-X86-64
            lapack
            ${MKL_LIBRARIES}
            )
endif()