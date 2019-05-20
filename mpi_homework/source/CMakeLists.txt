cmake_minimum_required(VERSION 3.9)
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


set(SOURCE_FILES ${SOURCE_FILES} errors.h matrixmul.cpp)

add_library(densematgen densematgen.cpp densematgen.h)

add_executable(matrixmul matrixmul.cpp)
target_link_libraries(matrixmul densematgen)
