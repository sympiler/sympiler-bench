cmake_minimum_required( VERSION 2.5)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/sympiler/cmake/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

project(sympiler_bench)


set(SYMPILER_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/includes/")
set(SYMPILER_LBC_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/lbc/includes/")
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


include_directories(includes
        ${SYMPILER_INC}
        ${SYMPILER_LBC_INC}
        )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -w -m64")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=x86-64   -mavx2")



add_subdirectory(sympiler)
add_subdirectory(tools_interface)
add_subdirectory(sym_interface)

