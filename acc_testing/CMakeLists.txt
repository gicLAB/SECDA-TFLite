cmake_minimum_required(VERSION 3.0)

option(BUILD_ARM "Build ARM" OFF)

set(CURRENT_COMPILER "HOST")
set(CROSS_CXX_COMPILER "/home/jude/Workspace/PYNQCC/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-g++")
set(CROSS_CXX_FLAGS "-marm -mfloat-abi=hard -march=armv7-a -mfpu=neon -funsafe-math-optimizations -ftree-vectorize -fPIC")
set(HOST_CXX_COMPILER "/usr/bin/clang++-12")
set(HOST_CXX_FLAGS "")

macro(use_host_compiler)
    if(${CURRENT_COMPILER} STREQUAL "CROSS")
        message("Switching to host compiler")

        # Save current cross flags
        set(CROSS_CXX_FLAGS ${CMAKE_CXX_FLAGS} CACHE STRING "GCC flags for the cross compiler." FORCE)

        # Change compiler
        set(CMAKE_CXX_COMPILER ${HOST_CXX_COMPILER})
        set(CMAKE_CXX_FLAGS ${HOST_CXX_FLAGS})
        set(CURRENT_COMPILER "HOST" CACHE STRING "Which compiler we are using." FORCE)
    endif()
endmacro()

macro(use_cross_compiler)
    if(${CURRENT_COMPILER} STREQUAL "HOST")
        message("Switching to cross compiler")

        # Save current host flags
        set(HOST_CXX_FLAGS ${CMAKE_CXX_FLAGS} CACHE STRING "GCC flags for the host compiler." FORCE)

        # Change compiler
        set(CMAKE_CXX_COMPILER ${CROSS_CXX_COMPILER})
        set(CMAKE_CXX_FLAGS ${CROSS_CXX_FLAGS})
        set(CURRENT_COMPILER "CROSS" CACHE STRING "Which compiler we are using." FORCE)
    endif()
endmacro()

# set(CROSS_CXX_FLAGS "${HOST_CXX_FLAGS}" CACHE STRING "GCC flags for the cross compiler.") # By default, the cross copiler is the same as the host compiler if we're not cross-compiling.
# set(CROSS_CXX_COMPILER "${HOST_CXX_COMPILER}" CACHE STRING "Path to the compiler for the system we are programing for.")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-12")
set(CMAKE_CXX_STANDARD 17)


SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
project(acc_testing)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModules" ${CMAKE_MODULE_PATH})

find_package(SYSC REQUIRED)

add_subdirectory(secda_tflite)
add_subdirectory(toy_add)
add_subdirectory(mm_v4)
