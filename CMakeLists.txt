cmake_minimum_required(VERSION 3.26)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build.")
project(CatDB LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG} -O3")

# silence a warning about DEPFILE path transformations (used in LLVM)
cmake_policy(SET CMP0116 OLD)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

# *****************************************************************************
# Project-specific include directories
# *****************************************************************************

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

# *****************************************************************************
# Descend to subdirectories
# *****************************************************************************

add_subdirectory(src/api)
add_subdirectory(src/parser)
add_subdirectory(src/runtime/io)
add_subdirectory(src/test)
add_subdirectory(src/util)
add_subdirectory(src/util/data)

# prepare data
add_executable(CatDB
        experiments/DataPrepare.cpp
        ${srcIO}
        ${srcUtil}
        ${srcSerialization}
        ${srcInPlace}
        ${srcProtobuf}
        ${srcProtobufcc}
        ${srcFlatbuf}
        src/test/run_tests.cpp
        src/test/run_tests.h
        src/runtime/util/data/Frame.cpp
        src/runtime/util/data/Frame.h
        src/runtime/util/data/Matrix.cpp
        src/runtime/util/data/Matrix.h
        src/runtime/util/data/Frame.h
        src/runtime/util/data/Structure.h
)
target_link_libraries(DataPrepare PRIVATE mongo::bsoncxx_shared)
target_link_libraries(DataPrepare PRIVATE mongo::mongocxx_shared)