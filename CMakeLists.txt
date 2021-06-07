cmake_minimum_required( VERSION 2.5)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/sympiler/cmake)
include(DownloadProject)


project(sympiler_bench)


set(SYMPILER_UTIL_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/utils/includes")
set(SYMPILER_UTILE_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/utils/external/includes")
set(SYMPILER_UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/utils/")
set(SYMPILER_SPARSE_BLAS_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/sparse_blas/includes")
set(SYMPILER_SPARSE_BLASP_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/sparse_blas/includes/parsy")
set(SYMPILER_LBC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/lbc/")


if(APPLE)
    if(EXISTS "/usr/local/opt/llvm/bin/clang++" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) # where LLVM is selected that has OpenMP
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
        link_directories(
                /usr/local/opt/llvm/lib/
        )
        message("Make sure to set the -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++")
    else()# otherwise, maybe gcc
        find_package(OpenMP REQUIRED)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
else()
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#add_definitions( -DPROFILE)
#add_definitions( -DLOAD)





#find_package(MKL REQUIRED)

### Find Intel MKL if available
#if(EXISTS "$ENV{MKLROOT}" OR EXISTS "$ENV{MKL_ROOT}")
#    if(EXISTS "$ENV{MKLROOT}")
#        set(MKLPATH "$ENV{MKLROOT}")
#    else()
#        set(MKLPATH "$ENV{MKL_ROOT}")
#    endif()
#
#    if(APPLE)
#        #find_package(OpenMP REQUIRED)
#        set(MKL_INC "${MKLPATH}/include")
#        set(MKL_LIB "${MKLPATH}/lib")
#        set(MKL_LIB2 "${MKLPATH}/../compiler/lib")
#    endif()
#    if(UNIX AND NOT APPLE)
#        set(MKL_INC "${MKLPATH}/include")
#        set(MKL_LIB "${MKLPATH}/lib/intel64")
#        # for iomp lib
#        set(MKL_LIB2 "${MKLPATH}/../compiler/lib/intel64")
#    endif()
#    add_definitions(-DMKL)
#    set(MKL 1)
#    set(MKL_LIB_NAME "mkl_intel_lp64"
#            "pthread"
#            "mkl_intel_thread"
#            "iomp5"
#            "mkl_sequential"
#            "mkl_core"
#            #"mkl_sequential"
#                     )
##                     "mkl_intel_ilp64")
#    message(STATUS "Intel MKL found")
#else()
#    message(STATUS "Intel MKL not found, building without MKL.")
#    set(MKL_LIB_NAME "")
#    set(MKL 0)
#endif()

### Find PAPI for Profiling
### Can be cloned directly from https://bitbucket.org/icl/papi.git
#if(EXISTS "$ENV{PAPIROOT}")
#    if(EXISTS "$ENV{PAPIROOT}/src")
#        set(PAPI_LIB "$ENV{PAPIROOT}/src")
#        set(PAPI_LIB2 "${PAPI_LIB}/libpfm4/lib/")
#    elseif(EXISTS "$ENV{PAPIROOT}/lib")
#        set(PAPI_LIB "$ENV{PAPIROOT}/lib")
#    endif()
#    if(EXISTS "$ENV{PAPIROOT}/include")
#        set(PAPI_INC "$ENV{PAPIROOT}/include")
#    else()
#        set(PAPI_INC "$ENV{PAPIROOT}/src")
#    endif()
#
#    add_definitions(-DPAPI)
#    set(PAPI_LIB_NAME "papi")
#    set(PAPI_LIB_NAME2 "pfm")
#    message(STATUS "PAPI found")
#else()
#    message(STATUS "PAPI not found, building without PAPI.")
#    set(PAPI_LIB_NAME "")
#endif()




set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -w -m64")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64   -mavx2")



add_subdirectory(sympiler)
add_subdirectory(sym_interface)
add_subdirectory(tools_interface)


