

cmake_minimum_required(VERSION 3.20)

# * CMAKE SCRIPTS *
# This allows for including CMake scripts from ./cmake/ directory.
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Include CMake scripts from /cmake



# * PROJECT *
set(TIP_VERSION_MAJOR 0)
set(TIP_VERSION_MINOR 0)
set(TIP_VERSION_PATCH 1)
set(TIP_VERSION ${TIP_VERSION_MAJOR}.${TIP_VERSION_MINOR}.${TIP_VERSION_PATCH})

project(TsImageProcessing
	VERSION ${TIP_VERSION}
    DESCRIPTION "TIP - Timie Image Processing Library"
    LANGUAGES C CXX
)


# * Compiler stuff *
message(STATUS "C++ Compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler is: ${CMAKE_CXX_COMPILER}")

# Set default standard to C++20
set(TIP_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD ${TIP_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add reasonable compile-time definitions.
add_compile_definitions(_USE_MATH_DEFINES)

# * Configuration *
option(TIP_LIBS_SHARED
       "Enable building data_processing_library as shared library"
       ON)
if(TIP_LIBS_SHARED)
    set(TIP_LIB_TYPE SHARED)
else()
    set(TIP_LIB_TYPE STATIC)
endif()


# * 3rd Party Libraries *
# External
# https://opencv.org
find_package(OpenCV REQUIRED)

# https://eigen.tuxfamily.org/
find_package(Eigen3 REQUIRED)
# add_compile_definitions(EIGEN_VECTORIZE_AVX512=1)

# Internal
# https://nemequ.github.io/hedley/
set(HEDLEY_include_dir "${CMAKE_CURRENT_LIST_DIR}/third_party/hedley_v15/include")

# https://github.com/ocornut/imgui
# find_package(imgui REQUIRED)



# **** TIP modules
# Libraries
add_subdirectory(tip_core)

add_subdirectory(tip_img_enhance)

# Applications
add_subdirectory(tip_img_enhance_demo)


# Unit tests
# TODO

# Performance tests
# TODO

# Evaluations
# TODO
