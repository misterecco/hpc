cmake_minimum_required(VERSION 3.5)
project(matrixmul)

set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING
            "Choose the type of build, options are: Debug Release"
            FORCE)

    message(STATUS "No build type selected, default to Debug")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-flto -O3 -fomit-frame-pointer -ffast-math -funsafe-math-optimizations -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if (ENABLE_SANITIZERS)
    message(STATUS "Sanitizers enabled")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()



# FIND DEPENDENCIES

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# set(MKL_USE_sdl on)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIR})
link_directories(${MKL_LIBRARY_DIR})

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})
link_directories(${MPI_CXX_LIBRARIES})


# DEFINE PROJECT FILES

set(SOURCE_FILES ${SOURCE_FILES} errors.h utils.h matrixmul.h mpigroup.h)

add_library(densematgen densematgen.cpp densematgen.h)
add_library(matrix matrix.cpp matrix.h)
add_library(initialization initialization.cpp)
add_library(replication replication.cpp)
add_library(multiplication multiplication.cpp)
add_library(gathering gathering.cpp)
add_library(communication communication.cpp)

target_link_libraries(matrix densematgen)

target_link_libraries(multiplication ${MKL_LIBRARIES})

add_executable(matrixmul matrixmul.cpp)
target_link_libraries(matrixmul
    matrix initialization replication multiplication gathering communication)
target_link_libraries(matrixmul ${MPI_CXX_LIBRARIES})
