cmake_minimum_required(VERSION 3.22)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build.")
project(CatDB LANGUAGES CXX C)

set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_FIND_DEBUG_MODE 1)

# check the operating system and add the appropriate flags
if (APPLE)
    SET(CMAKE_CXX_FLAGS " -std=c++11 -g -fPIC -Wno-deprecated-declarations")
    SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -undefined dynamic_lookup")
elseif (UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS " -std=c++11 -g -march=native -O3 -fPIC -Wno-deprecated-declarations -pthread -lprotobuf -lboost_serialization")
endif ()


#set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
#set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_CXX_FLAGS_DEBUG="${CMAKE_CXX_FLAGS_DEBUG} -O3")

message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
# silence a warning about DEPFILE path transformations (used in LLVM)
#cmake_policy(SET CMP0116 OLD)

#find_package(Arrow REQUIRED)
#find_package(Parquet REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

# *****************************************************************************
# Project-specific include directories
# *****************************************************************************

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

# *****************************************************************************
# Descend to subdirectories
# *****************************************************************************

#add_subdirectory(src/api)
#add_subdirectory(src/parser)
#add_subdirectory(src/runtime/io)
#add_subdirectory(src/runtime/util/data)
#add_subdirectory(src/test)
#add_subdirectory(src/util)

# set the directories with the common header files
include_directories("${PROJECT_SOURCE_DIR}/runtime/header")
include_directories("${PROJECT_SOURCE_DIR}/api")
include_directories("${PROJECT_SOURCE_DIR}/parser")
include_directories("${PROJECT_SOURCE_DIR}/runtime/io")
include_directories("${PROJECT_SOURCE_DIR}/runtime/util/data")
include_directories("${PROJECT_SOURCE_DIR}/test")
include_directories("${PROJECT_SOURCE_DIR}/util")

## prepare data
#add_executable(CatDB
#        src/test/run_tests.cpp
#        src/test/run_tests.h
#        src/runtime/util/data/Frame.cpp
#        src/runtime/util/data/Frame.h
#        src/runtime/util/data/Matrix.cpp
#        src/runtime/util/data/Matrix.h
#        src/runtime/util/data/Frame.h
#        src/runtime/util/data/Structure.h
#        src/runtime/util/data/ValueTypeCode.h
#        src/runtime/util/data/ValueTypeUtils.cpp
#        src/runtime/util/data/ValueTypeUtils.h
#        src/runtime/util/data/DataObjectFactory.h
#        src/runtime/util/data/Range.h
#        src/runtime/util/data/MetaDataObject.cpp
#        src/runtime/util/data/MetaDataObject.h
#        src/runtime/util/data/DataPlacement.cpp
#        src/runtime/util/data/DataPlacement.h
#        src/runtime/util/data/IAllocationDescriptor.h
#        src/runtime/util/data/DenseMatrix.cpp
#        src/runtime/util/data/DenseMatrix.h
#        src/runtime/util/data/AllocationDescriptorHost.h
#        src/runtime/io/File.h
#        src/test/runtime/io/ReadCsvTest.cpp
#)
