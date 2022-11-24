cmake_minimum_required( VERSION 3.19)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/sympiler/cmake/)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

project(sympiler_bench)


set(SYMPILER_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/include/")
set(SYMPILER_LBC_INC "${CMAKE_CURRENT_SOURCE_DIR}/sympiler/aggregation/include/")
#set(SYMPILER_LBC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sympiler//")


# Find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    if(APPLE) #TODO: set path for brew install libomp
        set(OpenMP_CXX_INCLUDE_DIRS  "/usr/local/include/")
        include_directories(
                ${OpenMP_CXX_INCLUDE_DIRS}
        )
        link_libraries(
                OpenMP::OpenMP_CXX
        )

        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.6")
    endif()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DOPENMP)
endif()



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

