cmake_minimum_required(VERSION 3.5)
project(matrixmul)

set(CMAKE_CXX_STANDARD 14)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release"
            FORCE)

    message(STATUS "No build type selected, default to Release")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "-flto -O3 -fomit-frame-pointer -ffast-math -funsafe-math-optimizations -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra -Wall")
  set(CMAKE_CXX_FLAGS_RELEASE "-flto -O3 -fomit-frame-pointer -ffast-math -funsafe-math-optimizations -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Cray")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -h msglevel_3")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -hfp3 -h vector3 -hipa4 -hwp -h pl=...")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -fp-model fast=2 -funroll-loops -std=c++14")
else()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")


# FIND DEPENDENCIES

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIR})
link_directories(${MKL_LIBRARY_DIR})
link_directories(${MKL_LIBRARIES})

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

target_link_libraries(multiplication mkl_intel_lp64 mkl_sequential mkl_core)

add_executable(matrixmul matrixmul.cpp)
target_link_libraries(matrixmul
    matrix initialization replication multiplication gathering communication)
target_link_libraries(matrixmul ${MPI_CXX_LIBRARIES})
