cmake_minimum_required(VERSION 3.26)
project(CatDB)

set(CMAKE_CXX_STANDARD 11)

add_executable(cpp main.cpp)




project(cpp)

# set clang as the default compiler
set(CMAKE_C_COMPILER "clang")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_FIND_DEBUG_MODE 1)

find_package(mongocxx CONFIG REQUIRED)
find_package(bsoncxx CONFIG REQUIRED)

include_directories(${LIBMONGOCXX_INCLUDE_DIR})
include_directories(${LIBBSONCXX_INCLUDE_DIR})


# set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/libraries")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/tools/cmake")
set(CTEST_BINARY_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

# check the operating system and add the appropriate flags
if (APPLE)
    SET(CMAKE_CXX_FLAGS " -std=c++11 -g -fPIC -Wno-deprecated-declarations")
    SET(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -undefined dynamic_lookup")
elseif (UNIX AND NOT APPLE)
    SET(CMAKE_CXX_FLAGS " -std=c++11 -g -march=native -O3 -fPIC -Wno-deprecated-declarations -pthread -lprotobuf -lboost_serialization")
endif ()


# set the directories with the common header files
include_directories("${PROJECT_SOURCE_DIR}/runtime/header")
include_directories("${PROJECT_SOURCE_DIR}/util/header")
include_directories("${PROJECT_SOURCE_DIR}/serialization/header")
include_directories("${PROJECT_SOURCE_DIR}/serialization/header/inplace")
include_directories("${PROJECT_SOURCE_DIR}/serialization/header/proto")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/proto/*.h")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/complexobject/header")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/complexobject/header/inplace")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/flatbuffers/*.h")
include_directories("${PROJECT_SOURCE_DIR}/serialization/header/flatbuffers")
include_directories("${PROJECT_SOURCE_DIR}/serialization/header/flatbuffers/flatbuffers")

# set the directories with the common source files
include_directories("${PROJECT_SOURCE_DIR}/runtime/source")
include_directories("${PROJECT_SOURCE_DIR}/util/source")
include_directories("${PROJECT_SOURCE_DIR}/serialization/source/")
include_directories("${PROJECT_SOURCE_DIR}/serialization/source/inplace/")
include_directories("${PROJECT_SOURCE_DIR}/serialization/source/proto/")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/complexobject/source/")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/complexobject/source/inplace/")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/proto/*.cc")
include_directories("${PROJECT_SOURCE_DIR}/benchmark/flatbuffers/*.cc")
include_directories("${PROJECT_SOURCE_DIR}/serialization/source/flatbuffers/")
include_directories("${PROJECT_SOURCE_DIR}/experiments/")

# set serialization source files
file(GLOB srcIO "${PROJECT_SOURCE_DIR}/runtime/source/*.cpp")
file(GLOB srcUtil "${PROJECT_SOURCE_DIR}/util/source/*.cpp")
file(GLOB srcSerialization "${PROJECT_SOURCE_DIR}/serialization/source/*.cpp")
file(GLOB srcInPlace "${PROJECT_SOURCE_DIR}/serialization/source/inplace/*.cpp")
file(GLOB srcProtobuf "${PROJECT_SOURCE_DIR}/serialization/source/proto/*.cpp")
file(GLOB srcProtobufcc "${PROJECT_SOURCE_DIR}/serialization/source/proto/*.cc")
file(GLOB srcFlatbuf "${PROJECT_SOURCE_DIR}/serialization/source/flatbuffers/*.cpp")
file(GLOB srcBenchmarkCCP "${PROJECT_SOURCE_DIR}/benchmark/proto/*.cc")
file(GLOB srcBenchmarkCCF "${PROJECT_SOURCE_DIR}/benchmark/flatbuffers/*.cc")
file(GLOB srcComplexObject "${PROJECT_SOURCE_DIR}/benchmark/complexobject/source/*.cpp")
file(GLOB srcComplexObjectIP "${PROJECT_SOURCE_DIR}/benchmark/complexobject/source/inplace/*.cpp")


# prepare data
add_executable(DataPrepare
        experiments/DataPrepare.cpp
        ${srcIO}
        ${srcUtil}
        ${srcSerialization}
        ${srcInPlace}
        ${srcProtobuf}
        ${srcProtobufcc}
        ${srcFlatbuf}
)
